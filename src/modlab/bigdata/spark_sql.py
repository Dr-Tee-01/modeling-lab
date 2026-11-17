"""Spark SQL in plain Python: join strategies, the Catalyst optimiser and table file layout.

These are simulations of Spark's behaviour, not Spark: no PySpark runs on the server. Each part
re-implements the documented rule closely enough that the intermediate state can be shown:

* **Joins** (join-strategies deck): the physical planner's choice between broadcast hash, shuffle
  hash, sort-merge, co-partitioned (bucketed) sort-merge, broadcast nested-loop and Cartesian
  joins, made the way Spark 3's ``JoinSelection`` makes it, and each strategy executed partition
  by partition on the course tables (or a sample of a synthetic scale-up).
* **SQL** (SQL intro and DataFrame decks): a small analyser, three optimiser rules applied one at a
  time (constant folding, predicate pushdown, column pruning), a physical planner and an executor,
  so the rows and columns read can be counted with and without each rule.
* **Tables** (DDL/DML deck): a ``PARTITIONED BY ... CLUSTERED BY ... INTO n BUCKETS`` Parquet write,
  partition and bucket pruning, row-group min/max skipping, managed versus external ``DROP TABLE``,
  and how ``DISTRIBUTE BY`` / ``SORT BY`` / ``CLUSTER BY`` / ``ORDER BY`` place rows.

Spark hashes with Murmur3 (x86, 32-bit, seed 42) and takes ``pmod(hash, n)`` both for shuffle
partitions and for bucket ids; that function is reproduced bit for bit (``murmur3_int(1) ==
-559580957``, as ``SELECT hash(1)`` returns in Spark).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .datasets import EMP8, EMP30_NODES, MARKS19, MR_CUSTOMERS, MR_ORDERS, SPARK_CUSTOMERS, SPARK_ORDERS

MB = 1024 * 1024
GB = 1024 * MB

# =========================================================================== Spark's hash
_M = 0xFFFFFFFF
_C1, _C2 = 0xCC9E2D51, 0x1B873593


def _s32(x: int) -> int:
    x &= _M
    return x - (1 << 32) if x & 0x80000000 else x


def _rotl(x: int, r: int) -> int:
    x &= _M
    return ((x << r) | (x >> (32 - r))) & _M


def _mix_k1(k: int) -> int:
    return (_rotl((k * _C1) & _M, 15) * _C2) & _M


def _mix_h1(h: int, k: int) -> int:
    return (_rotl(h ^ k, 13) * 5 + 0xE6546B64) & _M


def _fmix(h: int, n: int) -> int:
    h ^= n
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & _M
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & _M
    return h ^ (h >> 16)


def murmur3_int(v: int, seed: int = 42) -> int:
    """Spark's ``Murmur3_x86_32.hashInt``: the hash of an INT column."""
    return _s32(_fmix(_mix_h1(seed & _M, _mix_k1(v & _M)), 4))


def murmur3_str(s: str, seed: int = 42) -> int:
    """Spark's ``hashUnsafeBytes`` for a UTF-8 string (4-byte words, then one byte at a time)."""
    b = s.encode("utf-8")
    h, n = seed & _M, len(b)
    a = n - n % 4
    for i in range(0, a, 4):
        h = _mix_h1(h, _mix_k1(int.from_bytes(b[i:i + 4], "little")))
    for i in range(a, n):
        h = _mix_h1(h, _mix_k1((b[i] - 256 if b[i] > 127 else b[i]) & _M))
    return _s32(_fmix(h, n))


def murmur3_np(v: np.ndarray, seed: int = 42) -> np.ndarray:
    """``murmur3_int`` over an int array (uint64 arithmetic keeps the low 32 bits exact)."""
    k = np.asarray(v, dtype=np.int64).astype(np.uint64) & np.uint64(_M)
    m = np.uint64(_M)
    k = (k * np.uint64(_C1)) & m
    k = ((k << np.uint64(15)) | (k >> np.uint64(17))) & m
    k = (k * np.uint64(_C2)) & m
    h = np.uint64(seed & _M) ^ k
    h = ((h << np.uint64(13)) | (h >> np.uint64(19))) & m
    h = (h * np.uint64(5) + np.uint64(0xE6546B64)) & m
    h ^= np.uint64(4)
    h ^= h >> np.uint64(16)
    h = (h * np.uint64(0x85EBCA6B)) & m
    h ^= h >> np.uint64(13)
    h = (h * np.uint64(0xC2B2AE35)) & m
    h ^= h >> np.uint64(16)
    out = h.astype(np.int64)
    return np.where(out >= 2 ** 31, out - 2 ** 32, out)


def pmod(h, n):
    """Spark's ``pmod``: a non-negative remainder (Python's % already behaves this way)."""
    return h % n


def partition_of(keys, n: int) -> np.ndarray:
    """``pmod(murmur3(key), n)``: the shuffle partition (or bucket) of each integer key."""
    return pmod(murmur3_np(np.asarray(keys, dtype=np.int64)), n)


def fmt_bytes(b: float) -> str:
    b = float(b)
    for unit, size in (("TB", 1024 ** 4), ("GB", GB), ("MB", MB), ("KB", 1024)):
        if abs(b) >= size:
            v = b / size
            return f"{v:.1f} {unit}" if v < 100 else f"{v:.0f} {unit}"
    return f"{b:.0f} B"


def fmt_rows(n: float) -> str:
    n = float(n)
    for unit, size in (("billion", 1e9), ("million", 1e6)):
        if n >= size:
            return f"{n / size:.3g} {unit}"
    return f"{n:,.0f}"


def fmt_seconds(s: float) -> str:
    if s < 1:
        return f"{s * 1000:.0f} ms"
    if s < 120:
        return f"{s:.1f} s"
    if s < 7200:
        return f"{s / 60:.0f} min"
    if s < 2 * 86400:
        return f"{s / 3600:.1f} h"
    if s < 2 * 365 * 86400:
        return f"{s / 86400:.0f} days"
    return f"{s / (365 * 86400):.0f} years"


# =========================================================================== Joins
# Spark 3 constants used by the planner (documented defaults; beyond the slides).
AUTO_BROADCAST_DEFAULT_MB = 10            # spark.sql.autoBroadcastJoinThreshold = 10 MB
BROADCAST_HARD_LIMIT = 8 * GB             # a broadcast table larger than 8 GB is refused
MAX_PARTITION_BYTES = 128 * MB            # spark.sql.files.maxPartitionBytes: input split size
SKEW_FACTOR, SKEW_MIN_BYTES, ADVISORY_BYTES = 5, 256 * MB, 64 * MB   # AQE OptimizeSkewedJoin
# A deliberately simple cost model for the time estimate (beyond the slides; order of magnitude only).
NET_BPS = 125e6           # 1 Gbit/s of network per executor
HASH_RATE = 20e6          # rows hashed/probed per second per task
SORT_RATE = 50e6          # comparisons per second per task
NL_RATE = 100e6           # nested-loop comparisons per second per task

JOIN_TYPES = {"inner": "Inner", "left": "LeftOuter", "full": "FullOuter"}
JOIN_SQL = {"inner": "INNER JOIN", "left": "LEFT OUTER JOIN", "full": "FULL OUTER JOIN"}
STRATEGIES = ["bhj", "shj", "smj", "copart", "bnlj", "cartesian"]
STRATEGY = {
    "bhj": {"label": "Broadcast hash join", "exec": "BroadcastHashJoin", "short": "Broadcast hash"},
    "shj": {"label": "Shuffle hash join", "exec": "ShuffledHashJoin", "short": "Shuffle hash"},
    "smj": {"label": "Shuffle sort-merge join", "exec": "SortMergeJoin", "short": "Sort-merge"},
    "copart": {"label": "Co-partitioned (bucketed) sort-merge join", "exec": "SortMergeJoin", "short": "Co-partitioned"},
    "bnlj": {"label": "Broadcast nested-loop join", "exec": "BroadcastNestedLoopJoin", "short": "Nested loop"},
    "cartesian": {"label": "Cartesian product (shuffle-and-replicate nested loop)", "exec": "CartesianProduct", "short": "Cartesian"},
}
# Which side each strategy may build (hash or broadcast) for each join type, as in Spark 3.1+.
_BUILD = {
    "bhj": {"inner": ("left", "right"), "left": ("right",), "full": ()},
    "shj": {"inner": ("left", "right"), "left": ("right",), "full": ("left", "right")},
    # A broadcast nested loop chosen by size or hint obeys the broadcast rules (as "bhj"); the last-resort
    # fallback builds the smaller side whatever the join type, which BroadcastNestedLoopJoinExec supports.
    "bnlj": {"inner": ("left", "right"), "left": ("left", "right"), "full": ("left", "right")},
}
SIDE_NAME = {"left": "customers", "right": "orders"}


def a_join(join: str) -> str:
    """'an INNER JOIN', 'a LEFT OUTER JOIN', 'a FULL OUTER JOIN'."""
    return ("an " if join == "inner" else "a ") + JOIN_SQL[join]


HINTS = {"none": None, "broadcast": "BROADCAST", "shuffle_hash": "SHUFFLE_HASH", "merge": "MERGE",
         "shuffle_replicate_nl": "SHUFFLE_REPLICATE_NL"}

_COL_WIDTH = {"int": 4, "string": 20}


def _row_width(types) -> int:
    """Spark's per-row size estimate for an in-memory relation: 8 bytes of row header plus the
    default size of each column (4 for INT, 20 for STRING)."""
    return 8 + sum(_COL_WIDTH[t] for t in types)


COURSE_JOIN_SOURCES = {
    "course": {
        "label": "Customers and orders (RDD deck p40–41)",
        "left": {"name": "customers", "alias": "c", "cols": ["cust_id", "state"], "types": ["int", "string"],
                 "rows": SPARK_CUSTOMERS, "key": 0},
        "right": {"name": "orders", "alias": "o", "cols": ["order_id", "cust_id", "amount"], "types": ["int", "int", "int"],
                  "rows": SPARK_ORDERS, "key": 1},
    },
    "mr": {
        "label": "Customers and orders (MapReduce deck 2 p12/p17)",
        "left": {"name": "customers", "alias": "c", "cols": ["cust_id", "country", "state"], "types": ["int", "string", "string"],
                 "rows": MR_CUSTOMERS, "key": 0},
        "right": {"name": "orders", "alias": "o", "cols": ["order_id", "cust_id", "amount"], "types": ["int", "int", "int"],
                  "rows": MR_ORDERS, "key": 1},
    },
}
_SYN_STATES = ["Gujarat", "Texas", "Maharashtra", "Scotland", "Washington", "Karnataka"]
SAMPLE_LEFT, SAMPLE_RIGHT = 240, 2400         # rows actually joined for a synthetic scale-up
_TOP_KEYS = 4000                               # Zipf keys whose partition is computed exactly


def _zipf_weights(n_keys: int, s: float, top: int) -> tuple[np.ndarray, float]:
    """Probabilities of the `top` most frequent keys of a Zipf(s) law over n_keys keys, and the
    probability mass of all the others (spread evenly over partitions)."""
    k = np.arange(1, min(n_keys, top) + 1, dtype=float)
    head = k ** -s
    if n_keys <= top:
        h = head.sum()
        return head / h, 0.0
    a, b = top + 0.5, n_keys + 0.5
    tail = math.log(b / a) if abs(s - 1) < 1e-9 else (b ** (1 - s) - a ** (1 - s)) / (1 - s)
    h = head.sum() + tail
    return head / h, tail / h


def _synthetic_tables(R: int, L: int, skew: float, seed: int = 7):
    """A sample of the synthetic scale-up: Rs customers (unique ids 1..Rs) and Ls orders whose
    customer id follows Zipf(skew) over those ids; 1% of orders point to customers that do not
    exist (so a FULL OUTER join has something to add)."""
    rng = np.random.default_rng(seed)
    Rs, Ls = min(R, SAMPLE_LEFT), min(L, SAMPLE_RIGHT)
    cust = [(i, _SYN_STATES[i % len(_SYN_STATES)]) for i in range(1, Rs + 1)]
    p = np.arange(1, Rs + 1, dtype=float) ** -skew
    p /= p.sum()
    keys = rng.choice(np.arange(1, Rs + 1), size=Ls, p=p)
    orphan = rng.random(Ls) < 0.01
    keys = np.where(orphan, Rs + 1 + rng.integers(0, 5, Ls), keys)
    amounts = rng.integers(100, 1000, Ls)
    orders = [(i + 1, int(k), int(a)) for i, (k, a) in enumerate(zip(keys, amounts))]
    return cust, orders


def _sizes(R, L, lw, rw):
    return {"left_bytes": R * lw, "right_bytes": L * rw}


def _feasible(strategy: str, join: str, cond: str) -> tuple[bool, str]:
    if strategy in ("bhj", "shj", "smj", "copart") and cond != "eq":
        return False, "needs an equality (=) join condition: hashing and sorting only find equal keys"
    if strategy in _BUILD and not _BUILD[strategy][join]:
        return False, f"cannot build either side for {a_join(join)} (the build side's unmatched rows would be lost)"
    if strategy == "cartesian" and join != "inner":
        return False, "a Cartesian product only implements INNER joins"
    return True, ""


def _pick_build(strategy, join, sizes) -> str | None:
    sides = _BUILD.get(strategy, {}).get(join, ())
    if not sides:
        return None
    return min(sides, key=lambda s: sizes[f"{s}_bytes"])


def plan_join(*, join: str, cond: str, hint: str, threshold_mb: float, partitions: int, prefer_smj: bool,
              bucketed: bool, sizes: dict) -> dict:
    """Choose a physical join the way Spark 3's JoinSelection does, recording every check.

    Equi-joins: hinted strategies first (BROADCAST, SHUFFLE_HASH, MERGE, SHUFFLE_REPLICATE_NL), then
    broadcast hash if a buildable side is at most autoBroadcastJoinThreshold, then shuffle hash if
    preferSortMergeJoin is false and one side is small enough per partition and 3x smaller, then
    sort-merge. Non-equi joins: broadcast nested loop if a side can be broadcast, else Cartesian
    product (inner only), else broadcast nested loop on the smaller buildable side anyway."""
    thr = -1 if threshold_mb < 0 else threshold_mb * MB
    checks: list[dict] = []
    lb, rb = sizes["left_bytes"], sizes["right_bytes"]

    def can_broadcast(side):
        b = sizes[f"{side}_bytes"]
        return thr >= 0 and b <= thr

    def done(strategy, build, why):
        exchange = strategy in ("shj", "smj") and not bucketed
        if strategy in ("shj", "smj") and bucketed:
            strategy = "copart" if strategy == "smj" else strategy
            checks.append({"rule": "EnsureRequirements", "ok": True,
                           "detail": f"both tables are bucketed by cust_id into {partitions} buckets with the same hash, "
                                     "so rows with equal keys are already in the same partition: no Exchange is added"})
        return {"strategy": strategy, "build": build, "exchange": exchange, "checks": checks, "why": why,
                "label": STRATEGY[strategy]["label"], "exec": STRATEGY[strategy]["exec"]}

    hinted = HINTS.get(hint)
    if hinted:
        checks.append({"rule": f"Hint /*+ {hinted} */", "ok": True, "detail": "hints are tried before any size rule"})
    if cond == "eq":
        if hint == "broadcast":
            side = _pick_build("bhj", join, sizes)
            if side:
                if sizes[f"{side}_bytes"] > BROADCAST_HARD_LIMIT:
                    checks.append({"rule": "BROADCAST hint", "ok": False,
                                   "detail": f"{side} is {fmt_bytes(sizes[side + '_bytes'])}, above the 8 GB broadcast limit; the hint fails at run time"})
                checks.append({"rule": "BROADCAST hint", "ok": True, "detail": f"{a_join(join)} can build the {side} side"})
                return done("bhj", side, f"the BROADCAST hint asks for it and {SIDE_NAME[side]} (the {side} side) can be built for {a_join(join)}")
            checks.append({"rule": "BROADCAST hint", "ok": False, "detail": f"{a_join(join)} cannot build either side: the hint is ignored"})
        if hint == "shuffle_hash":
            side = _pick_build("shj", join, sizes)
            if side:
                checks.append({"rule": "SHUFFLE_HASH hint", "ok": True, "detail": f"builds a hash table of the {side} side in every partition"})
                return done("shj", side, "the SHUFFLE_HASH hint asks for it")
        if hint == "merge":
            checks.append({"rule": "MERGE hint", "ok": True, "detail": "the join keys are sortable"})
            return done("smj", None, "the MERGE hint asks for it")
        if hint == "shuffle_replicate_nl":
            if join == "inner":
                checks.append({"rule": "SHUFFLE_REPLICATE_NL hint", "ok": True, "detail": "inner join: a Cartesian product is allowed"})
                return done("cartesian", None, "the SHUFFLE_REPLICATE_NL hint asks for it")
            checks.append({"rule": "SHUFFLE_REPLICATE_NL hint", "ok": False, "detail": "only for inner joins: the hint is ignored"})
        # 1. broadcast hash join by size
        cands = [s for s in _BUILD["bhj"][join] if can_broadcast(s)]
        for s in ("left", "right"):
            b = sizes[f"{s}_bytes"]
            if thr < 0:
                detail = f"{SIDE_NAME[s]} ({s}) is {fmt_bytes(b)}; auto-broadcast is disabled (threshold −1)"
            else:
                detail = f"{SIDE_NAME[s]} ({s}) is {fmt_bytes(b)} {'≤' if b <= thr else '>'} threshold {fmt_bytes(thr)}"
            if s not in _BUILD["bhj"][join] and b <= max(thr, 0):
                detail += f", but {a_join(join)} cannot build the {s} side"
            checks.append({"rule": f"Broadcast {s}?", "ok": s in cands, "detail": detail})
        if cands:
            side = min(cands, key=lambda s: sizes[f"{s}_bytes"])
            return done("bhj", side, f"{SIDE_NAME[side]} (the {side} side) is under spark.sql.autoBroadcastJoinThreshold and can be built for {a_join(join)}")
        # 2. shuffle hash join (only when preferSortMergeJoin is false)
        if prefer_smj:
            checks.append({"rule": "Shuffle hash?", "ok": False, "detail": "spark.sql.join.preferSortMergeJoin = true (the default) skips this rule"})
        else:
            ok_sides = []
            for s in _BUILD["shj"][join]:
                o = "right" if s == "left" else "left"
                small = thr > 0 and sizes[f"{s}_bytes"] < thr * partitions
                much = sizes[f"{s}_bytes"] * 3 <= sizes[f"{o}_bytes"]
                checks.append({"rule": f"Shuffle hash, build {s}?", "ok": small and much,
                               "detail": f"{fmt_bytes(sizes[s + '_bytes'])} {'<' if small else '≥'} threshold × {partitions} partitions; "
                                         f"{'at least' if much else 'less than'} 3× smaller than the {o} side"})
                if small and much:
                    ok_sides.append(s)
            if ok_sides:
                side = min(ok_sides, key=lambda s: sizes[f"{s}_bytes"])
                return done("shj", side, f"preferSortMergeJoin is off and {SIDE_NAME[side]} (the {side} side) fits a hash map per partition")
        checks.append({"rule": "Sort-merge?", "ok": True, "detail": "the key (cust_id, an INT) is sortable"})
        return done("smj", None, "no side can be broadcast, so both are shuffled by cust_id and sorted")
    # non-equi join
    checks.append({"rule": "Equi-join keys?", "ok": False, "detail": "the condition is a range (BETWEEN), so hash and sort-merge joins are impossible"})
    sides = [s for s in _BUILD["bhj"][join] if hint == "broadcast" or can_broadcast(s)]
    if sides:
        side = min(sides, key=lambda s: sizes[f"{s}_bytes"])
        how = "the BROADCAST hint names it" if hint == "broadcast" else f"{fmt_bytes(sizes[side + '_bytes'])} is under the threshold"
        checks.append({"rule": "Broadcast nested loop?", "ok": True, "detail": f"the {side} side can be broadcast: {how}"})
        return done("bnlj", side, "a non-equi join with a side that may be broadcast")
    checks.append({"rule": "Broadcast nested loop?", "ok": False,
                   "detail": (f"{a_join(join)} may not broadcast either side" if not _BUILD["bhj"][join] else
                              f"no side {a_join(join)} may broadcast is under the threshold")})
    if join == "inner":
        checks.append({"rule": "Cartesian product?", "ok": True, "detail": "inner join: pair every partition of one side with every partition of the other"})
        return done("cartesian", None, "a non-equi inner join with no side small enough to broadcast")
    side = _pick_build("bnlj", join, sizes)
    checks.append({"rule": "Fallback", "ok": True, "detail": f"outer non-equi join: Spark broadcasts the smaller ({side}) side anyway, even if it is large or the join type would not normally allow it"})
    return done("bnlj", side, "the last resort for an outer non-equi join")


# --------------------------------------------------------------------------- execution
def _matches(stream_keys: np.ndarray, build_keys: np.ndarray, cond: str) -> list[tuple[int, int]]:
    """(i, j) positions where stream key i matches build key j."""
    if not len(stream_keys) or not len(build_keys):
        return []
    if cond == "eq":
        index = defaultdict(list)
        for j, k in enumerate(build_keys.tolist()):
            index[k].append(j)
        return [(i, j) for i, k in enumerate(stream_keys.tolist()) for j in index.get(k, ())]
    hit = np.abs(stream_keys[:, None] - build_keys[None, :]) <= 1
    ii, jj = np.nonzero(hit)
    return list(zip(ii.tolist(), jj.tolist()))


def _merge_join(lrows: list[int], rrows: list[int], lk, rk) -> tuple[list[tuple[int, int]], int]:
    """A real merge of two key-sorted lists (with duplicate key groups); returns pairs and comparisons."""
    i = j = cmps = 0
    out = []
    while i < len(lrows) and j < len(rrows):
        a, b = lk[lrows[i]], rk[rrows[j]]
        cmps += 1
        if a < b:
            i += 1
        elif a > b:
            j += 1
        else:
            i2, j2 = i, j
            while i2 < len(lrows) and lk[lrows[i2]] == a:
                i2 += 1
            while j2 < len(rrows) and rk[rrows[j2]] == a:
                j2 += 1
            out += [(x, y) for x in lrows[i:i2] for y in rrows[j:j2]]
            i, j = i2, j2
    return out, cmps


def _outer(pairs, lrows, rrows, keep_left, keep_right):
    out = list(pairs)
    if keep_left:
        hit = {l for l, _ in pairs}
        out += [(l, None) for l in lrows if l not in hit]
    if keep_right:
        hit = {r for _, r in pairs}
        out += [(None, r) for r in rrows if r not in hit]
    return out


def _chunks(n: int, k: int) -> list[list[int]]:
    k = max(1, min(k, n)) if n else 1
    edges = np.linspace(0, n, k + 1).round().astype(int)
    return [list(range(edges[i], edges[i + 1])) for i in range(k)]


def execute_join(strategy: str, build: str | None, join: str, cond: str, lk: np.ndarray, rk: np.ndarray,
                 partitions: int, executors: int, capture: bool = False) -> dict:
    """Run one strategy on actual rows (indices into the left and right tables) partition by partition.
    Returns the output pairs (left index or None, right index or None) and the steps it went through."""
    keep_l, keep_r = join in ("left", "full"), join == "full"
    nl, nr = len(lk), len(rk)
    lin, rin = _chunks(nl, executors), _chunks(nr, executors)
    steps, pairs, stats = [], [], {"comparisons": 0, "sort_comparisons": 0, "probes": 0}
    if capture:
        steps.append({"kind": "read", "title": "Read the input partitions",
                      "note": f"Each table arrives split across {executors} executor(s), as read from its files: no key order.",
                      "parts": [{"label": f"executor {i}", "left": lin[i] if i < len(lin) else [], "right": rin[i] if i < len(rin) else []}
                                for i in range(max(len(lin), len(rin)))]})
    if strategy in ("bhj", "bnlj"):
        stream = "left" if build == "right" else "right"
        bk, sk = (rk, lk) if build == "right" else (lk, rk)
        bidx = list(range(len(bk)))
        sparts = lin if stream == "left" else rin
        keep_stream = keep_l if stream == "left" else keep_r
        keep_build = keep_r if build == "right" else keep_l
        if capture:
            steps.append({"kind": "broadcast", "title": f"Collect the {build} side and broadcast it to every executor",
                          "note": (f"The driver builds the hash table H{build[0].upper()} on cust_id and sends one copy to each of {executors} executor(s)."
                                   if strategy == "bhj" else f"The whole {build} side is sent once to each of {executors} executor(s); nothing is hashed."),
                          "build": build, "build_rows": bidx, "copies": executors})
        matched_build: set[int] = set()
        probe_parts = []
        for p, rows in enumerate(sparts):
            ks = sk[rows] if rows else np.array([], dtype=np.int64)
            m = _matches(ks, bk, cond)
            part_pairs = [(rows[i], j) for i, j in m]
            matched_build.update(j for _, j in part_pairs)
            if strategy == "bnlj":
                stats["comparisons"] += len(rows) * len(bk)
            else:
                stats["probes"] += len(rows)
            out = list(part_pairs)
            if keep_stream:
                hit = {s for s, _ in part_pairs}
                out += [(s, None) for s in rows if s not in hit]
            if stream == "right":
                out = [(b, s) for s, b in out]
            pairs += out
            probe_parts.append({"label": f"{stream} partition {p} (executor {p % executors})", "stream": stream, "rows": rows,
                                "out": out, "comparisons": len(rows) * len(bk) if strategy == "bnlj" else len(rows)})
        final = []
        if keep_build:
            final = [(b, None) if build == "left" else (None, b) for b in bidx if b not in matched_build]
            pairs += final
        if capture:
            steps.append({"kind": "probe", "title": "Probe the hash table with each streamed partition" if strategy == "bhj"
                          else "Compare every streamed row with every broadcast row",
                          "note": ("Each streamed row looks up its cust_id in the local copy of the hash table: one probe per row."
                                   if strategy == "bhj" else f"{len(bk)} broadcast rows × each streamed row = {stats['comparisons']:,} comparisons in all."),
                          "parts": probe_parts})
            if keep_build:
                steps.append({"kind": "final", "title": f"Emit the {build} rows that matched nowhere",
                              "note": "The executors report which broadcast rows they matched; the rest are null-extended once.",
                              "out": final})
        return {"pairs": pairs, "steps": steps, "stats": stats}

    if strategy == "cartesian":
        tasks = []
        for i, lrows in enumerate(lin):
            for j, rrows in enumerate(rin):
                m = _matches(lk[lrows], rk[rrows], cond) if lrows and rrows else []
                out = [(lrows[a], rrows[b]) for a, b in m]
                stats["comparisons"] += len(lrows) * len(rrows)
                pairs += out
                tasks.append({"label": f"task ({i}, {j})", "left": lrows, "right": rrows, "out": out,
                              "comparisons": len(lrows) * len(rrows)})
        if capture:
            steps.append({"kind": "replicate", "title": "Pair every left partition with every right partition",
                          "note": f"{len(lin)} × {len(rin)} = {len(tasks)} tasks; each fetches one partition of each side and compares all pairs.",
                          "parts": tasks})
        return {"pairs": pairs, "steps": steps, "stats": stats}

    # shuffle-based: shj, smj, copart
    lp, rp = partition_of(lk, partitions), partition_of(rk, partitions)
    lparts = [np.nonzero(lp == p)[0].tolist() for p in range(partitions)]
    rparts = [np.nonzero(rp == p)[0].tolist() for p in range(partitions)]
    if capture:
        steps.append({"kind": "exchange", "title": ("Read the bucket files: both tables are already split by pmod(hash(cust_id), n)"
                                                    if strategy == "copart" else f"Exchange: shuffle both sides by pmod(murmur3(cust_id), {partitions})"),
                      "note": ("No data moves: bucket i of customers and bucket i of orders are read by the same task."
                               if strategy == "copart" else "Every row is written to the shuffle file of its partition and fetched by that partition's task."),
                      "parts": [{"label": f"partition {p} (executor {p % executors})", "left": lparts[p], "right": rparts[p]} for p in range(partitions)]})
    join_parts = []
    for p in range(partitions):
        lrows, rrows = lparts[p], rparts[p]
        if strategy == "shj":
            if build == "right":
                m = _matches(lk[lrows] if lrows else np.array([], np.int64), rk[rrows] if rrows else np.array([], np.int64), cond)
                part_pairs = [(lrows[i], rrows[j]) for i, j in m]
            else:
                m = _matches(rk[rrows] if rrows else np.array([], np.int64), lk[lrows] if lrows else np.array([], np.int64), cond)
                part_pairs = [(lrows[j], rrows[i]) for i, j in m]
            stats["probes"] += len(rrows) if build == "left" else len(lrows)
            sorted_l, sorted_r = lrows, rrows
        else:
            sorted_l = sorted(lrows, key=lambda i: (lk[i], i))
            sorted_r = sorted(rrows, key=lambda i: (rk[i], i))
            stats["sort_comparisons"] += int(sum(n * math.log2(n) for n in (len(lrows), len(rrows)) if n > 1))
            part_pairs, c = _merge_join(sorted_l, sorted_r, lk, rk)
            stats["comparisons"] += c
        out = _outer(part_pairs, lrows, rrows, keep_l, keep_r)
        pairs += out
        join_parts.append({"label": f"partition {p} (executor {p % executors})", "left": sorted_l, "right": sorted_r, "out": out})
    if capture:
        if strategy == "shj":
            steps.append({"kind": "join", "title": f"Build a hash table of the {build} side in each partition, probe it with the other",
                          "note": "Only the rows of one partition need to fit in memory at a time.", "parts": join_parts, "build": build})
        else:
            steps.append({"kind": "join", "title": "Sort each partition by cust_id, then merge the two sorted runs",
                          "note": "The merge walks both sorted lists once, pairing equal keys; sorting can spill to disk.",
                          "parts": join_parts})
    return {"pairs": pairs, "steps": steps, "stats": stats}


def reference_join(lk: np.ndarray, rk: np.ndarray, join: str, cond: str) -> list[tuple]:
    """The join computed directly (all pairs), to check every strategy against."""
    m = _matches(lk, rk, cond)
    return _outer(m, list(range(len(lk))), list(range(len(rk))), join in ("left", "full"), join == "full")


def _canon(pairs) -> list[tuple[int, int]]:
    return sorted((-1 if a is None else a, -1 if b is None else b) for a, b in pairs)


# --------------------------------------------------------------------------- scaled costs
def _partition_loads(source: str, R: int, L: int, skew: float, partitions: int, lk=None, rk=None) -> tuple[np.ndarray, np.ndarray]:
    """Rows of each side in each shuffle partition: exact for the course tables, expected for the
    synthetic scale-up (Zipf weights of the most frequent keys hashed exactly, the tail spread evenly)."""
    if source != "synthetic":
        return (np.bincount(partition_of(lk, partitions), minlength=partitions).astype(float),
                np.bincount(partition_of(rk, partitions), minlength=partitions).astype(float))
    top = min(R, _TOP_KEYS)
    kp = partition_of(np.arange(1, top + 1), partitions)
    if R <= _TOP_KEYS:
        lp = np.bincount(kp, minlength=partitions).astype(float)
    else:
        lp = np.full(partitions, R / partitions)
    w, tail = _zipf_weights(R, skew, _TOP_KEYS)
    rp = L * 0.99 * (np.bincount(kp, weights=w, minlength=partitions) + tail / partitions) + L * 0.01 / partitions
    return lp, rp


def strategy_cost(strategy: str, build: str | None, *, sizes: dict, R: float, L: float, lw: int, rw: int,
                  lp: np.ndarray, rp: np.ndarray, partitions: int, executors: int, aqe: bool, exchange: bool = True) -> dict:
    """Data moved, work and memory of one strategy at the full (logical) size, and the time the simple
    cost model predicts. Per-partition arrays describe the tasks of the join stage."""
    E = executors
    Sl, Sr = sizes["left_bytes"], sizes["right_bytes"]
    out = {"broadcast_bytes": 0.0, "shuffle_bytes": 0.0, "sort_comparisons": 0.0, "nl_comparisons": 0.0,
           "extra_read_bytes": 0.0, "skew_splits": 0}
    if strategy in ("bhj", "bnlj"):
        sb = Sl if build == "left" else Sr
        nb = R if build == "left" else L
        ns, ss = (L, Sr) if build == "left" else (R, Sl)
        tasks = int(min(4000, max(E, math.ceil(ss / MAX_PARTITION_BYTES))))
        per = np.full(tasks, ns / tasks)
        out["broadcast_bytes"] = sb * E
        out["peak_mem_bytes"] = sb
        if strategy == "bhj":
            work = per / HASH_RATE + nb / HASH_RATE / tasks    # probe each streamed row; build cost shared
            out["task_rows"] = per
        else:
            out["nl_comparisons"] = R * L
            work = per * nb / NL_RATE
            out["task_rows"] = per
        net = sb * E / (E * NET_BPS)
        out["tasks"] = tasks
    elif strategy == "cartesian":
        pl = int(max(E, math.ceil(Sl / MAX_PARTITION_BYTES)))
        pr = int(max(E, math.ceil(Sr / MAX_PARTITION_BYTES)))
        tasks = pl * pr
        out["nl_comparisons"] = R * L
        out["shuffle_bytes"] = Sl * pr + Sr * pl
        out["peak_mem_bytes"] = Sl / pl + Sr / pr
        per_cmp = (R / pl) * (L / pr)
        work = np.full(min(tasks, 4000), per_cmp / NL_RATE)
        out["task_rows"] = np.full(min(tasks, 4000), R / pl + L / pr)
        net = out["shuffle_bytes"] / (E * NET_BPS)
        out["tasks"] = tasks
    else:
        rows = lp + rp
        bytes_p = lp * lw + rp * rw
        if strategy != "copart" and exchange:
            out["shuffle_bytes"] = Sl + Sr
        if strategy in ("smj", "copart"):
            sc = np.where(lp > 1, lp * np.log2(np.maximum(lp, 1)), 0) + np.where(rp > 1, rp * np.log2(np.maximum(rp, 1)), 0)
            out["sort_comparisons"] = float(sc.sum())
            work = rows / HASH_RATE + sc / SORT_RATE
            out["peak_mem_bytes"] = float(bytes_p.max())
        else:
            bp = (lp * lw) if build == "left" else (rp * rw)
            work = rows / HASH_RATE
            out["peak_mem_bytes"] = float(bp.max())
        out["task_rows"] = rows
        if aqe and strategy != "copart" and exchange:
            med = float(np.median(bytes_p))
            target = max(ADVISORY_BYTES, med)
            new_work, new_rows, new_bytes, splits, extra = [], [], [], 0, 0.0
            for p in range(partitions):
                big = bytes_p[p] > SKEW_FACTOR * med and bytes_p[p] > SKEW_MIN_BYTES
                if big:
                    k = int(math.ceil(bytes_p[p] / target))
                    splits += k - 1
                    # the skewed (larger) side is split k ways, the other side's partition is read k times
                    other = min(lp[p] * lw, rp[p] * rw)
                    extra += other * (k - 1)
                    new_work += [work[p] / k] * k
                    new_rows += [rows[p] / k] * k
                    new_bytes += [(bytes_p[p] - other) / k + other] * k
                else:
                    new_work.append(work[p])
                    new_rows.append(rows[p])
                    new_bytes.append(bytes_p[p])
            work, out["task_rows"] = np.array(new_work), np.array(new_rows)
            if splits:
                out["peak_mem_bytes"] = min(out["peak_mem_bytes"], float(max(new_bytes)))
            out["skew_splits"], out["extra_read_bytes"] = splits, extra
        net = (out["shuffle_bytes"] + out["extra_read_bytes"]) / (E * NET_BPS)
        out["tasks"] = len(work)
    stage = max(float(np.sum(work)) / E, float(np.max(work)) if len(work) else 0.0)
    tr = out["task_rows"]
    med = float(np.median(tr)) if len(tr) else 0.0
    out["straggler_ratio"] = float(np.max(tr) / med) if med > 0 else 1.0
    out["straggler_task"] = int(np.argmax(tr)) if len(tr) else 0
    out["est_seconds"] = net + stage
    out["moved_bytes"] = out["broadcast_bytes"] + out["shuffle_bytes"] + out["extra_read_bytes"]
    return out


# --------------------------------------------------------------------------- the whole run
def run_join(*, source: str = "course", join: str = "inner", cond: str = "eq", hint: str = "none", layout: str = "plain",
             threshold_mb: float = AUTO_BROADCAST_DEFAULT_MB, partitions: int = 8, executors: int = 4,
             prefer_smj: bool = True, aqe: bool = False, lrows_exp: float = 5, rrows_exp: float = 8,
             lbytes: int = 100, rbytes: int = 50, skew: float = 0.0) -> dict:
    """Plan and execute ``customers c <join> orders o ON <cond>`` and report every stage."""
    if source in COURSE_JOIN_SOURCES:
        src = COURSE_JOIN_SOURCES[source]
        left, right = src["left"], src["right"]
        lrows_t, rrows_t = list(left["rows"]), list(right["rows"])
        lw, rw = _row_width(left["types"]), _row_width(right["types"])
        R, L = len(lrows_t), len(rrows_t)
        lcols, rcols = left["cols"], right["cols"]
        label = src["label"]
        scale_note = "the course tables, joined in full"
    else:
        R, L = int(round(10 ** lrows_exp)), int(round(10 ** rrows_exp))
        lw, rw = int(lbytes), int(rbytes)
        lrows_t, rrows_t = _synthetic_tables(R, L, skew)
        lcols, rcols = ["cust_id", "state"], ["order_id", "cust_id", "amount"]
        label = f"Synthetic scale-up: {fmt_rows(R)} customers, {fmt_rows(L)} orders"
        scale_note = (f"a sample of {len(lrows_t)} customers and {len(rrows_t)} orders is joined for real; "
                      "sizes, partitions and costs are computed for the full tables")
    lk = np.array([r[0] for r in lrows_t], dtype=np.int64)
    rk = np.array([r[1] for r in rrows_t], dtype=np.int64)
    sizes = _sizes(R, L, lw, rw)
    plan = plan_join(join=join, cond=cond, hint=hint, threshold_mb=threshold_mb, partitions=partitions,
                     prefer_smj=prefer_smj, bucketed=(layout == "bucketed"), sizes=sizes)
    strat, build = plan["strategy"], plan["build"]

    # execute the chosen strategy with every intermediate state, and every feasible one to check
    run = execute_join(strat, build, join, cond, lk, rk, partitions, executors, capture=True)
    ref = _canon(reference_join(lk, rk, join, cond))
    validation = []
    for s in STRATEGIES:
        ok, why = _feasible(s, join, cond)
        if not ok:
            validation.append({"strategy": s, "label": STRATEGY[s]["short"], "feasible": False, "reason": why})
            continue
        b = _pick_build(s, join, sizes) if s in _BUILD else None
        r = run if s == strat and b == build else execute_join(s, b, join, cond, lk, rk, partitions, executors)
        validation.append({"strategy": s, "label": STRATEGY[s]["short"], "feasible": True, "rows": len(r["pairs"]),
                           "same_as_reference": _canon(r["pairs"]) == ref})

    # costs at full size for every strategy
    lp, rp = _partition_loads(source, R, L, skew, partitions, lk, rk)
    compare = []
    for s in STRATEGIES:
        ok, why = _feasible(s, join, cond)
        b = build if s == strat else (_pick_build(s, join, sizes) if s in _BUILD else None)
        if ok and s in ("bhj", "bnlj") and (sizes["left_bytes"] if b == "left" else sizes["right_bytes"]) > BROADCAST_HARD_LIMIT:
            ok, why = False, f"the {b} side is over Spark's 8 GB broadcast limit"
        row = {"strategy": s, "label": STRATEGY[s]["short"], "feasible": ok, "reason": why, "build": b, "chosen": s == strat}
        if ok or s == strat:
            c = strategy_cost(s, b, sizes=sizes, R=R, L=L, lw=lw, rw=rw, lp=lp, rp=rp, partitions=partitions,
                              executors=executors, aqe=aqe, exchange=not (s == strat and not plan["exchange"]))
            row.update({k: v for k, v in c.items() if k != "task_rows"})
        compare.append(row)
    chosen = strategy_cost(strat, build, sizes=sizes, R=R, L=L, lw=lw, rw=rw, lp=lp, rp=rp, partitions=partitions,
                           executors=executors, aqe=aqe, exchange=plan["exchange"] or strat not in ("shj", "smj"))
    shuffle_based = strat in ("shj", "smj", "copart")

    # output rows
    def out_row(a, b):
        return list(lrows_t[a] if a is not None else [None] * len(lcols)) + list(rrows_t[b] if b is not None else [None] * len(rcols))
    ordered = sorted(run["pairs"], key=lambda p: (lk[p[0]] if p[0] is not None else rk[p[1]], -1 if p[0] is None else p[0],
                                                  -1 if p[1] is None else p[1]))
    result = {"cols": [f"c.{c}" for c in lcols] + [f"o.{c}" for c in rcols], "rows": [out_row(a, b) for a, b in ordered[:40]],
              "total": len(run["pairs"]), "unmatched_left": sum(1 for a, b in run["pairs"] if b is None),
              "unmatched_left_ids": sorted(int(lk[a]) for a, b in run["pairs"] if b is None)[:12],
              "unmatched_right": sum(1 for a, b in run["pairs"] if a is None)}

    def rows_of(side, idx, limit=14):
        t = lrows_t if side == "left" else rrows_t
        return {"n": len(idx), "rows": [list(t[i]) for i in idx[:limit]]}

    steps = []
    for st in run["steps"]:
        s2 = {k: v for k, v in st.items() if k not in ("parts", "build_rows", "out")}
        if "build_rows" in st:
            s2["build_table"] = rows_of(st["build"], st["build_rows"], 20)
        if "out" in st:
            s2["out"] = {"n": len(st["out"]), "rows": [out_row(a, b) for a, b in st["out"][:12]]}
        if "parts" in st:
            parts = []
            for p in st["parts"][:16]:
                q = {"label": p["label"]}
                for side in ("left", "right"):
                    if side in p:
                        q[side] = rows_of(side, p[side])
                if "rows" in p:
                    q["stream"] = rows_of(p["stream"], p["rows"])
                    q["stream_side"] = p["stream"]
                if "out" in p:
                    q["out"] = {"n": len(p["out"]), "rows": [out_row(a, b) for a, b in p["out"][:8]]}
                if "comparisons" in p:
                    q["comparisons"] = p["comparisons"]
                parts.append(q)
            s2["parts"], s2["parts_total"] = parts, len(st["parts"])
        steps.append(s2)

    # key → hash → partition map for the course tables (the shuffle's routing table)
    key_map = []
    if source != "synthetic":
        for k in sorted(set(lk.tolist()) | set(rk.tolist())):
            h = murmur3_int(int(k))
            key_map.append({"key": int(k), "hash": h, "partition": h % partitions})

    show = min(partitions, 200)
    tr = chosen["task_rows"]
    part_view = {"n": partitions, "shown": show,
                 "left": lp[:show].tolist(), "right": rp[:show].tolist(),
                 "left_bytes": (lp[:show] * lw).tolist(), "right_bytes": (rp[:show] * rw).tolist()}
    straggler = int(np.argmax(lp + rp)) if shuffle_based else int(chosen["straggler_task"])
    sql = (f"SELECT * FROM customers c {JOIN_SQL[join]} orders o\n  ON "
           + ("c.cust_id = o.cust_id" if cond == "eq" else "o.cust_id BETWEEN c.cust_id - 1 AND c.cust_id + 1"))
    if hint != "none":
        sql = sql.replace("SELECT *", f"SELECT /*+ {HINTS[hint]}({'o' if (plan['build'] == 'right') else 'c'}) */ *")
    return {
        "source": source, "label": label, "scale_note": scale_note, "sql": sql,
        "tables": {"left": {"name": "customers", "cols": lcols, "rows": R, "sample_rows": len(lrows_t), "row_bytes": lw,
                            "bytes": sizes["left_bytes"], "preview": [list(r) for r in lrows_t[:8]]},
                   "right": {"name": "orders", "cols": rcols, "rows": L, "sample_rows": len(rrows_t), "row_bytes": rw,
                             "bytes": sizes["right_bytes"], "preview": [list(r) for r in rrows_t[:8]]}},
        "threshold_bytes": -1 if threshold_mb < 0 else threshold_mb * MB,
        "plan": {k: v for k, v in plan.items()} | {"tree": physical_join_tree(plan, join, cond, partitions, lcols, rcols)},
        "steps": steps, "key_map": key_map,
        "partitions": part_view | {"straggler": straggler, "shuffle_based": shuffle_based,
                                   "task_rows": tr[:400].tolist(), "tasks": int(chosen["tasks"])},
        "metrics": {k: v for k, v in chosen.items() if k != "task_rows"} | {"strategy": strat},
        "compare": compare, "validation": validation, "result": result,
        "settings": {"partitions": partitions, "executors": executors, "threshold_mb": threshold_mb, "join": join,
                     "cond": cond, "hint": hint, "layout": layout, "prefer_smj": prefer_smj, "aqe": aqe},
    }


def physical_join_tree(plan: dict, join: str, cond: str, partitions: int, lcols, rcols) -> dict:
    """The physical plan as Spark's EXPLAIN would print it (simplified)."""
    s, build, jt = plan["strategy"], plan["build"], JOIN_TYPES[join]
    keys = "[cust_id], [cust_id]"
    scan_l = {"label": "FileScan customers", "detail": f"[{', '.join(lcols)}]"}
    scan_r = {"label": "FileScan orders", "detail": f"[{', '.join(rcols)}]"}
    cond_s = "(o.cust_id >= c.cust_id - 1) AND (o.cust_id <= c.cust_id + 1)"
    if s in ("bhj", "bnlj"):
        side = "BuildRight" if build == "right" else "BuildLeft"
        bc = {"label": "BroadcastExchange", "detail": "HashedRelationBroadcastMode(cust_id)" if s == "bhj" else "IdentityBroadcastMode", "tag": "move"}
        l = scan_l if build == "right" else dict(bc, children=[scan_l])
        r = dict(bc, children=[scan_r]) if build == "right" else scan_r
        detail = f"{keys}, {jt}, {side}" if s == "bhj" else f"{side}, {jt}, {cond_s}"
        return {"label": STRATEGY[s]["exec"], "detail": detail, "tag": "join", "children": [l, r]}
    if s == "cartesian":
        return {"label": "CartesianProduct", "detail": cond_s, "tag": "join", "children": [scan_l, scan_r]}

    def side_tree(scan):
        node = scan
        if plan["exchange"]:
            node = {"label": "Exchange", "detail": f"hashpartitioning(cust_id, {partitions})", "tag": "move", "children": [node]}
        else:
            node = dict(scan, detail=scan["detail"] + f", bucketed into {partitions} buckets by cust_id")
        if s in ("smj", "copart"):
            node = {"label": "Sort", "detail": "[cust_id ASC NULLS FIRST], false, 0", "tag": "sort", "children": [node]}
        return node
    detail = f"{keys}, {jt}" + (f", Build{'Left' if build == 'left' else 'Right'}" if s == "shj" else "")
    return {"label": STRATEGY[s]["exec"], "detail": detail, "tag": "join", "children": [side_tree(scan_l), side_tree(scan_r)]}


# =========================================================================== Spark SQL: Catalyst
# The employee table of the DataFrame deck's "company database of the DB book" (deck p14 schema:
# eno, name, dob, gender, salary, sup_eno, dno). eno, name, dno and salary are the course's EMP8
# (RDD deck p28); dob, gender and the supervisor come from the same COMPANY database
# (Elmasri & Navathe), as does the department table. Genders use the deck's M/F coding (employee_m.csv).
_EMP_EXTRA = {101: ("1965-01-09", "M", 102), 102: ("1955-12-08", "M", 105), 103: ("1972-07-31", "F", 102),
              104: ("1962-09-15", "M", 102), 105: ("1937-11-10", "M", None), 106: ("1941-06-20", "F", 105),
              107: ("1969-03-29", "M", 106), 108: ("1968-01-19", "F", 106)}
EMPLOYEE_COLS = ["eno", "name", "dob", "gender", "salary", "sup_eno", "dno"]
EMPLOYEE_TYPES = ["int", "string", "date", "string", "int", "int", "int"]
EMPLOYEE = [(eno, name, _EMP_EXTRA[eno][0], _EMP_EXTRA[eno][1], salary, _EMP_EXTRA[eno][2], dno)
            for eno, name, dno, salary in EMP8]
DEPARTMENT_COLS = ["dno", "name", "mgr_eno", "mgr_start"]
DEPARTMENT_TYPES = ["int", "string", "int", "date"]
DEPARTMENT = [(5, "Research", 102, "1988-05-22"), (4, "Administration", 106, "1995-01-01"), (1, "Headquarters", 105, "1981-06-19")]
# DDL deck p39: the marks.csv student ids, in the order of MARKS19.
MARKS_IDS = ["2015xx035", "2015xx070", "2015xx025", "2015xx032"] + [f"2016xx{i:03d}" for i in range(1, 16)]
MARKS_COLS, MARKS_TYPES = ["student_id", "marks"], ["string", "int"]
MARKS = list(zip(MARKS_IDS, MARKS19))
# DataFrame deck p11/p13: the visible rows of employee_m.csv; p17: employees.json.
EMPLOYEE_M_HEADER = ["empno", "name", "dno", "salary", "state", "gender"]
EMPLOYEE_M_ROWS = [(10026, "Wilson K Adinolfi", 5, 62506, "MA", "M"), (10084, "Karthikeyan Ait Sidi", 3, 104437, "MA", "M"),
                   (10196, "Sarah Akinkuolie", 5, 64955, "MA", "F")]
EMPLOYEES_JSON = [{"name": "Michael", "salary": 3000}, {"name": "Andy", "salary": 4500},
                  {"name": "Justin", "salary": 3500}, {"name": "Berta", "salary": 4000}]
SQL_TABLES = {"employee": (EMPLOYEE_COLS, EMPLOYEE_TYPES, EMPLOYEE), "department": (DEPARTMENT_COLS, DEPARTMENT_TYPES, DEPARTMENT),
              "marks": (MARKS_COLS, MARKS_TYPES, MARKS)}

# DDL deck p40, implemented exactly as written: the (40, 45] and (35, 40] branches both return "DD".
LETTER_GRADE_RULES = [(">=", 90, "AA"), (">=", 75, "AB"), (">=", 65, "BB"), (">=", 55, "BC"), (">=", 50, "CC"),
                      (">", 45, "CD"), (">", 40, "DD"), (">", 35, "DD"), (">", 30, "DE")]
SLIDE_GRADES_FIRST10 = ["DD", "F", "DD", "DE", "BB", "DD", "BC", "BB", "BB", "DD"]     # DDL deck p41


def letter_grade(marks: float) -> str:
    for opname, bound, grade in LETTER_GRADE_RULES:
        if (marks >= bound) if opname == ">=" else (marks > bound):
            return grade
    return "F"


def letter_grade_report() -> dict:
    graded = [{"student_id": s, "marks": m, "grade": letter_grade(m)} for s, m in MARKS]
    first10 = [g["grade"] for g in graded[:10]]
    dup = [r for r in LETTER_GRADE_RULES if r[2] == "DD"]
    counts: dict[str, int] = {}
    for g in graded:
        counts[g["grade"]] = counts.get(g["grade"], 0) + 1
    return {"rules": [{"test": f"marks {o} {b}", "grade": g, "duplicate": g == "DD"} for o, b, g in LETTER_GRADE_RULES] + [{"test": "otherwise", "grade": "F", "duplicate": False}],
            "graded": graded, "first10": first10, "slide_first10": SLIDE_GRADES_FIRST10, "matches_slide": first10 == SLIDE_GRADES_FIRST10,
            "duplicate": {"grade": "DD", "branches": [f"marks {o} {b}" for o, b, _ in dup],
                          "range": "35 < marks ≤ 45", "students": sum(1 for g in graded if 35 < g["marks"] <= 45)},
            "counts": counts}


# ---------------------------------------------------------------- expressions
def col(q, n): return ("col", q, n)
def lit(v): return ("lit", v)
def op(s, a, b): return ("op", s, a, b)
def agg(f, a): return ("agg", f, a)
def udf(n, a): return ("udf", n, a)
def alias(e, n): return ("as", e, n)
def order(e, desc=False): return ("ord", e, desc)


_CMP = {">": "GreaterThan", ">=": "GreaterThanOrEqual", "<": "LessThan", "<=": "LessThanOrEqual", "=": "EqualTo"}
_OPS = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b, "/": lambda a, b: a / b,
        ">": lambda a, b: a > b, ">=": lambda a, b: a >= b, "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
        "=": lambda a, b: a == b, "AND": lambda a, b: bool(a) and bool(b)}


def show(e, qualify: bool = True) -> str:
    t = e[0]
    if t == "col":
        return f"{e[1]}.{e[2]}" if qualify and e[1] else e[2]
    if t == "lit":
        v = e[1]
        return f"'{v}'" if isinstance(v, str) else (f"{v:g}" if isinstance(v, float) else str(v))
    if t == "op":
        return f"({show(e[2], qualify)} {e[1]} {show(e[3], qualify)})"
    if t == "agg":
        return f"{e[1]}({show(e[2], qualify)})"
    if t == "udf":
        return f"{e[1]}({show(e[2], qualify)})"
    if t == "as":
        return f"{show(e[1], qualify)} AS {e[2]}"
    if t == "ord":
        return f"{show(e[1], qualify)} {'DESC' if e[2] else 'ASC'}"
    raise ValueError(e)


def refs(e) -> set:
    t = e[0]
    if t == "col":
        return {(e[1], e[2])}
    if t == "lit":
        return set()
    if t == "op":
        return refs(e[2]) | refs(e[3])
    return refs(e[2] if t in ("agg", "udf") else e[1])


def _ops_in(e) -> int:
    """Operator and function evaluations an expression costs per row."""
    t = e[0]
    if t in ("col", "lit"):
        return 0
    if t == "op":
        return 1 + _ops_in(e[2]) + _ops_in(e[3])
    if t == "udf":
        return 1 + _ops_in(e[2])
    return _ops_in(e[2] if t == "agg" else e[1])


def _fold(e, log: list):
    t = e[0]
    if t == "op":
        a, b = _fold(e[2], log), _fold(e[3], log)
        if a[0] == "lit" and b[0] == "lit":
            v = _OPS[e[1]](a[1], b[1])
            log.append(f"{show(op(e[1], a, b))} → {show(lit(v))}")
            return lit(v)
        return op(e[1], a, b)
    if t in ("agg", "udf"):
        return (t, e[1], _fold(e[2], log))
    if t == "as":
        return alias(_fold(e[1], log), e[2])
    if t == "ord":
        return order(_fold(e[1], log), e[2])
    return e


def _ev(e, row):
    t = e[0]
    if t == "col":
        return row[f"{e[1]}.{e[2]}"]
    if t == "lit":
        return e[1]
    if t == "op":
        return _OPS[e[1]](_ev(e[2], row), _ev(e[3], row))
    if t == "udf":
        return letter_grade(_ev(e[2], row))
    if t == "as":
        return _ev(e[1], row)
    raise ValueError(f"cannot evaluate {e[0]} per row")


def _name(e) -> str:
    if e[0] == "as":
        return e[2]
    if e[0] == "col":
        return f"{e[1]}.{e[2]}"
    return show(e)


def _conjuncts(e) -> list:
    return _conjuncts(e[2]) + _conjuncts(e[3]) if e[0] == "op" and e[1] == "AND" else [e]


def _and(cs: list):
    out = cs[0]
    for c in cs[1:]:
        out = op("AND", out, c)
    return out


# ---------------------------------------------------------------- logical plans
def _rel(table: str, a: str) -> dict:
    cols = SQL_TABLES[table][0]
    return {"op": "Relation", "table": table, "alias": a, "cols": list(cols), "read": list(cols), "pushed": []}


def _star(a: str, table: str) -> list:
    return [col(a, c) for c in SQL_TABLES[table][0]]


SQL_QUERIES = {
    "filter": {"label": "Filter", "slide": "DataFrame deck p19", "tables": ["employee"]},
    "filter_and": {"label": "Filter with AND", "slide": "DataFrame deck p19", "tables": ["employee"]},
    "project": {"label": "Project", "slide": "DataFrame deck p21", "tables": ["employee"]},
    "order": {"label": "Order by", "slide": "DataFrame deck p20", "tables": ["employee"]},
    "agg": {"label": "Group by", "slide": "DataFrame deck p22", "tables": ["employee"]},
    "join_agg": {"label": "Join and sum", "slide": "DataFrame deck p24", "tables": ["employee", "department"]},
    "slide16": {"label": "Slide-16 query", "slide": "SQL intro deck p16", "tables": ["employee", "department"]},
    "udf": {"label": "letterGrade UDF", "slide": "DDL deck p39–41", "tables": ["marks"]},
}


def build_query(q: str, k: int = 30, dno: int = 5, gender: str = "F") -> dict:
    """The analysed (unoptimised) logical plan of a preset query, with its SQL and DataFrame text.
    Thresholds are written as k * 1000 so that ConstantFolding has something to fold."""
    thr = op("*", lit(k), lit(1000))
    if q == "filter":
        plan = {"op": "Project", "exprs": _star("e", "employee"),
                "child": {"op": "Filter", "cond": op(">", col("e", "salary"), thr), "child": _rel("employee", "e")}}
        sql = f"SELECT * FROM employee e WHERE e.salary > {k} * 1000"
        df = f'emp.filter("salary > {k * 1000}").show();'
    elif q == "filter_and":
        cond = op("AND", op(">=", col("e", "salary"), thr), op("=", col("e", "dno"), lit(dno)))
        plan = {"op": "Project", "exprs": _star("e", "employee"), "child": {"op": "Filter", "cond": cond, "child": _rel("employee", "e")}}
        sql = f"SELECT * FROM employee e WHERE e.salary >= {k} * 1000 AND e.dno = {dno}"
        df = f'emp.filter(emp.col("salary").geq({k * 1000}).and(emp.col("dno").equalTo({dno}))).show();'
    elif q == "project":
        plan = {"op": "Project", "exprs": [col("e", "eno"), col("e", "name"), alias(op("*", col("e", "salary"), lit(1.1)), "raised")],
                "child": {"op": "Filter", "cond": op("=", col("e", "dno"), lit(dno)), "child": _rel("employee", "e")}}
        sql = f"SELECT e.eno, e.name, e.salary * 1.1 AS raised FROM employee e WHERE e.dno = {dno}"
        df = f'emp.select(emp.col("eno"), emp.col("name"), emp.col("salary").multiply(1.1)).where("dno == {dno}").show();'
    elif q == "order":
        plan = {"op": "Sort", "orders": [order(col("e", "dno")), order(col("e", "salary"), True)],
                "child": {"op": "Project", "exprs": _star("e", "employee"), "child": _rel("employee", "e")}}
        sql = "SELECT * FROM employee e ORDER BY e.dno, e.salary DESC"
        df = 'emp.sort(emp.col("dno"), emp.col("salary").desc()).show();'
    elif q == "agg":
        plan = {"op": "Aggregate", "keys": [col("e", "dno")], "aggs": [agg("avg", col("e", "salary")), agg("max", col("e", "salary"))],
                "child": _rel("employee", "e")}
        sql = "SELECT e.dno, avg(e.salary), max(e.salary) FROM employee e GROUP BY e.dno"
        df = 'emp.groupBy(emp.col("dno")).agg(avg(emp.col("salary")), max(emp.col("salary"))).show();'
    elif q == "join_agg":
        plan = {"op": "Aggregate", "keys": [col("d", "name")], "aggs": [agg("sum", col("e", "salary"))],
                "child": {"op": "Join", "how": "Inner", "cond": op("=", col("e", "dno"), col("d", "dno")),
                          "left": _rel("employee", "e"), "right": _rel("department", "d")}}
        sql = "SELECT d.name, sum(e.salary) FROM employee e JOIN department d ON e.dno = d.dno GROUP BY d.name"
        df = 'emp.join(dep, emp.col("dno").equalTo(dep.col("dno"))).groupBy(dep.col("name")).agg(sum(emp.col("salary"))).show();'
    elif q == "slide16":
        plan = {"op": "Aggregate", "keys": [col("d", "dno"), col("d", "name")], "aggs": [agg("count", col("e", "name"))],
                "child": {"op": "Filter", "cond": op("=", col("e", "gender"), lit(gender)),
                          "child": {"op": "Join", "how": "Inner", "cond": op("=", col("e", "dno"), col("d", "dno")),
                                    "left": _rel("employee", "e"), "right": _rel("department", "d")}}}
        sql = (f"SELECT d.dno, d.name, count(e.name) FROM employee e JOIN department d ON e.dno = d.dno\n"
               f"WHERE e.gender = '{gender}' GROUP BY d.dno, d.name")
        df = ('employees.join(dept, employees("dno") === dept("dno"))\n  .where(employees("gender") === "' + gender
              + '")\n  .groupBy(dept("dno"), dept("name")).agg(count(employees("name")))')
    elif q == "udf":
        plan = {"op": "Project", "exprs": [col("m", "student_id"), col("m", "marks"), alias(udf("letterGrade", col("m", "marks")), "grade")],
                "child": _rel("marks", "m")}
        sql = "SELECT m.student_id, m.marks, letterGrade(m.marks) AS grade FROM marks m"
        df = 'spark.udf.register("letterGrade", letterGrade)\nspark.sql("SELECT student_id, marks, letterGrade(marks) AS grade FROM marks")'
    else:
        raise ValueError(q)
    return {"id": q, "sql": sql, "dataframe": df, "plan": plan, **SQL_QUERIES[q]}


def _copy(n):
    if isinstance(n, dict):
        return {k: _copy(v) for k, v in n.items()}
    if isinstance(n, list):
        return [_copy(v) for v in n]
    return n


def _children(n) -> list:
    return [n[k] for k in ("child", "left", "right") if k in n]


def _exprs(n) -> list:
    out = []
    for k in ("cond",):
        if k in n:
            out.append(n[k])
    for k in ("exprs", "keys", "aggs", "orders"):
        out += n.get(k, [])
    return out


def _aliases(n) -> set:
    if n["op"] == "Relation":
        return {n["alias"]}
    return set().union(*(_aliases(c) for c in _children(n)))


def rule_constant_folding(plan: dict) -> tuple[dict, list]:
    plan, log = _copy(plan), []

    def walk(n):
        for k in ("cond",):
            if k in n:
                before = len(log)
                n[k] = _fold(n[k], log)
                if len(log) > before:
                    n["_mark"] = True
        for k in ("exprs", "keys", "aggs", "orders"):
            if k in n:
                before = len(log)
                n[k] = [_fold(e, log) for e in n[k]]
                if len(log) > before:
                    n["_mark"] = True
        for c in _children(n):
            walk(c)
    walk(plan)
    return plan, [f"ConstantFolding: {x}" for x in log]


def _pushable(c) -> bool:
    """A predicate a file source can evaluate: column <op> literal, with no UDF or arithmetic."""
    return c[0] == "op" and c[1] in _CMP and ((c[2][0] == "col" and c[3][0] == "lit") or (c[2][0] == "lit" and c[3][0] == "col"))


def _source_filter(c) -> str:
    colname = c[2][2] if c[2][0] == "col" else c[3][2]
    value = c[3] if c[2][0] == "col" else c[2]
    return f"{_CMP[c[1]]}({colname},{show(value).strip(chr(39))})"


def rule_predicate_pushdown(plan: dict) -> tuple[dict, list]:
    """PushPredicateThroughJoin plus file-source filter pushdown (PushedFilters on the scan)."""
    log: list[str] = []

    def walk(n):
        n = dict(n)
        if n["op"] == "Filter" and n["child"]["op"] == "Join" and n["child"]["how"] == "Inner":
            j = dict(n["child"])
            la, ra = _aliases(j["left"]), _aliases(j["right"])
            lc, rc, keep = [], [], []
            for c in _conjuncts(n["cond"]):
                qs = {q for q, _ in refs(c)}
                (lc if qs <= la else rc if qs <= ra else keep).append(c)
            if lc:
                j["left"] = {"op": "Filter", "cond": _and(lc), "child": j["left"], "_mark": True}
                log.append(f"PushPredicateThroughJoin: Filter {show(_and(lc))} moved below the Join, onto the {j['left']['child'].get('table', 'left')} side")
            if rc:
                j["right"] = {"op": "Filter", "cond": _and(rc), "child": j["right"], "_mark": True}
                log.append(f"PushPredicateThroughJoin: Filter {show(_and(rc))} moved below the Join, onto the {j['right']['child'].get('table', 'right')} side")
            n = {"op": "Filter", "cond": _and(keep), "child": j} if keep else j
        for k in ("child", "left", "right"):
            if k in n:
                n[k] = walk(n[k])
        if n["op"] == "Filter" and n["child"]["op"] == "Relation":
            rel = dict(n["child"])
            pushed = [c for c in _conjuncts(n["cond"]) if _pushable(c)]
            unpushed = [c for c in _conjuncts(n["cond"]) if not _pushable(c)]
            if pushed:
                rel["pushed"] = pushed
                rel["_mark"] = True
                log.append(f"Data-source pushdown: {', '.join(show(c) for c in pushed)} handed to the {rel['table']} scan (PushedFilters)")
            if unpushed:
                log.append(f"Not pushed to the scan: {', '.join(show(c) for c in unpushed)} (not a column compared with a literal)")
            n["child"] = rel
        return n
    return walk(_copy(plan)), log


def rule_column_pruning(plan: dict) -> tuple[dict, list]:
    plan, log = _copy(plan), []
    needed: set = set()

    def collect(n):
        for e in _exprs(n):
            needed.update(refs(e))
        for c in _children(n):
            collect(c)

    def prune(n):
        if n["op"] == "Relation":
            keep = [c for c in n["cols"] if (n["alias"], c) in needed]
            if len(keep) < len(n["read"]):
                log.append(f"ColumnPruning: read {len(keep)} of {len(n['cols'])} columns of {n['table']} ({', '.join(keep)})")
                n["read"] = keep
                n["_mark"] = True
        for c in _children(n):
            prune(c)
    collect(plan)
    prune(plan)
    if not log:
        log.append("ColumnPruning: every column is used (SELECT *), nothing to prune")
    return plan, log


SQL_RULES = [("folding", "ConstantFolding", rule_constant_folding), ("pushdown", "PushDownPredicates", rule_predicate_pushdown),
             ("pruning", "ColumnPruning", rule_column_pruning)]


def optimise(plan: dict, rules: dict) -> tuple[dict, list]:
    stages = []
    for key, name, fn in SQL_RULES:
        if rules.get(key):
            plan, log = fn(_unmark(plan))
            stages.append({"rule": key, "name": name, "log": log, "tree": logical_tree(plan),
                           "fired": not (key == "pruning" and log and "nothing to prune" in log[0]) and bool(log)})
    return _unmark(plan), stages


def _unmark(n):
    if isinstance(n, dict):
        return {k: _unmark(v) for k, v in n.items() if k != "_mark"}
    if isinstance(n, list):
        return [_unmark(v) for v in n]
    return n


def logical_tree(n: dict) -> dict:
    o = n["op"]
    if o == "Relation":
        label, detail = "Relation", f"{n['table']}[{', '.join(n['read'])}] parquet"
    elif o == "Filter":
        label, detail = "Filter", show(n["cond"])
    elif o == "Project":
        label, detail = "Project", "[" + ", ".join(show(e) for e in n["exprs"]) + "]"
    elif o == "Join":
        label, detail = "Join", f"{n['how']}, {show(n['cond'])}"
    elif o == "Aggregate":
        label, detail = "Aggregate", "[" + ", ".join(show(e) for e in n["keys"]) + "], [" + ", ".join(show(e) for e in n["keys"] + n["aggs"]) + "]"
    elif o == "Sort":
        label, detail = "Sort", "[" + ", ".join(show(e) for e in n["orders"]) + "], true"
    else:
        label, detail = o, ""
    out = {"label": label, "detail": detail}
    if n.get("_mark"):
        out["tag"] = "changed"
    kids = _children(n)
    if kids:
        out["children"] = [logical_tree(c) for c in kids]
    return out


_TYPE = {"int": "int", "string": "string", "date": "date"}


def physical_tree(n: dict, broadcast: bool = True) -> dict:
    o = n["op"]
    if o == "Relation":
        cols, types, _ = SQL_TABLES[n["table"]]
        schema = ",".join(f"{c}:{_TYPE[types[cols.index(c)]]}" for c in n["read"])
        pf = []
        for c in n["pushed"]:
            pf += [f"IsNotNull({c[2][2] if c[2][0] == 'col' else c[3][2]})", _source_filter(c)]
        detail = f"[{', '.join(n['read'])}] PushedFilters: [{', '.join(dict.fromkeys(pf))}], ReadSchema: struct<{schema}>"
        return {"label": f"FileScan parquet {n['table']}", "detail": detail, "tag": "scan"}
    kids = [physical_tree(c, broadcast) for c in _children(n)]
    if o == "Filter":
        return {"label": "Filter", "detail": show(n["cond"]), "children": kids}
    if o == "Project":
        u = [e for e in n["exprs"] if e[0] == "as" and e[1][0] == "udf"]
        if u:
            ev = {"label": "BatchEvalPython", "detail": f"[{show(u[0][1])}], [pythonUDF0]", "tag": "udf", "children": kids}
            names = [("pythonUDF0 AS " + e[2]) if e in u else show(e) for e in n["exprs"]]
            return {"label": "Project", "detail": "[" + ", ".join(names) + "]", "children": [ev]}
        return {"label": "Project", "detail": "[" + ", ".join(show(e) for e in n["exprs"]) + "]", "children": kids}
    if o == "Join":
        lk, rk = show(n["cond"][2]), show(n["cond"][3])
        if broadcast:
            right = {"label": "BroadcastExchange", "detail": "HashedRelationBroadcastMode", "tag": "move", "children": [kids[1]]}
            return {"label": "BroadcastHashJoin", "detail": f"[{lk}], [{rk}], Inner, BuildRight", "tag": "join", "children": [kids[0], right]}
        sides = [{"label": "Sort", "detail": f"[{k} ASC NULLS FIRST], false, 0", "tag": "sort",
                  "children": [{"label": "Exchange", "detail": f"hashpartitioning({k}, 200)", "tag": "move", "children": [c]}]}
                 for k, c in ((lk, kids[0]), (rk, kids[1]))]
        return {"label": "SortMergeJoin", "detail": f"[{lk}], [{rk}], Inner", "tag": "join", "children": sides}
    if o == "Aggregate":
        keys = ", ".join(show(e) for e in n["keys"])
        fns = ", ".join(show(e) for e in n["aggs"])
        partial = ", ".join("partial_" + show(e) for e in n["aggs"])
        part = {"label": "HashAggregate", "detail": f"(keys=[{keys}], functions=[{partial}])", "children": kids}
        ex = {"label": "Exchange", "detail": f"hashpartitioning({keys}, 200)" if keys else "SinglePartition", "tag": "move", "children": [part]}
        return {"label": "HashAggregate", "detail": f"(keys=[{keys}], functions=[{fns}])", "children": [ex]}
    if o == "Sort":
        ords = ", ".join(show(e) for e in n["orders"])
        ex = {"label": "Exchange", "detail": f"rangepartitioning({ords}, 200)", "tag": "move", "children": kids}
        return {"label": "Sort", "detail": f"[{ords}], true, 0", "tag": "sort", "children": [ex]}
    return {"label": o, "detail": "", "children": kids}


def _row_group_stats(table: str) -> dict:
    cols, types, rows = SQL_TABLES[table]
    out = {}
    for i, c in enumerate(cols):
        vals = [r[i] for r in rows if r[i] is not None]
        if vals and types[i] == "int":
            out[c] = (min(vals), max(vals))
    return out


def _can_skip(table: str, pushed: list) -> bool:
    """Parquet min/max: True if no row of the (single) row group can satisfy every pushed filter."""
    stats = _row_group_stats(table)
    for c in pushed:
        cname = c[2][2] if c[2][0] == "col" else c[3][2]
        if cname not in stats or c[2][0] != "col":
            continue
        lo, hi = stats[cname]
        v, s = c[3][1], c[1]
        if not isinstance(v, (int, float)):
            continue
        if (s == ">" and hi <= v) or (s == ">=" and hi < v) or (s == "<" and lo >= v) or (s == "<=" and lo > v) or (s == "=" and not lo <= v <= hi):
            return True
    return False


def execute_plan(plan: dict) -> tuple[list[str], list[dict], dict, list]:
    """Run a logical plan on the course tables, counting what every operator reads and produces."""
    m = {"cells_read": 0, "rows_read": 0, "cols_read": 0, "rows_into_join": 0, "rows_processed": 0, "evals": 0, "row_groups_skipped": 0}
    trace: list[dict] = []

    def run(n) -> tuple[list[str], list[dict]]:
        o = n["op"]
        if o == "Relation":
            cols, types, rows = SQL_TABLES[n["table"]]
            if n["pushed"] and _can_skip(n["table"], n["pushed"]):
                data = []
                m["row_groups_skipped"] += 1
            else:
                data = [{f"{n['alias']}.{c}": r[cols.index(c)] for c in n["read"]} for r in rows]
            m["rows_read"] += len(data)
            m["cols_read"] += len(n["read"])
            m["cells_read"] += len(data) * len(n["read"])
            names = [f"{n['alias']}.{c}" for c in n["read"]]
            trace.append({"op": "Scan", "detail": f"{n['table']} ({len(n['read'])} of {len(cols)} columns)"
                          + (", row group skipped by min/max" if n["pushed"] and not data else ""),
                          "rows_in": len(rows), "rows_out": len(data), "cols": names, "rows": [[r[c] for c in names] for r in data[:8]]})
            return names, data
        if o == "Join":
            ln, lrows = run(n["left"])
            rn, rrows = run(n["right"])
            m["rows_into_join"] += len(lrows) + len(rrows)
            m["rows_processed"] += len(lrows) + len(rrows)
            a, b = _name(n["cond"][2]), _name(n["cond"][3])
            index = defaultdict(list)
            for r in rrows:
                index[r[b]].append(r)
            data = [{**l, **r} for l in lrows for r in index.get(l[a], ())]
            names = ln + rn
            label, rows_in = "Join", len(lrows) + len(rrows)
        else:
            names_in, rows = run(n["child"])
            rows_in = len(rows)
            m["rows_processed"] += rows_in
            if o == "Filter":
                m["evals"] += rows_in * _ops_in(n["cond"])
                data, names, label = [r for r in rows if _ev(n["cond"], r)], names_in, "Filter"
            elif o == "Project":
                m["evals"] += rows_in * sum(_ops_in(e) for e in n["exprs"])
                names = [_name(e) for e in n["exprs"]]
                data = [{nm: _ev(e, r) for nm, e in zip(names, n["exprs"])} for r in rows]
                label = "Project"
            elif o == "Aggregate":
                groups: dict = {}
                for r in rows:
                    groups.setdefault(tuple(_ev(k, r) for k in n["keys"]), []).append(r)
                names = [_name(k) for k in n["keys"]] + [_name(a) for a in n["aggs"]]
                data = []
                for key, grp in groups.items():
                    row = dict(zip(names, key))
                    for a in n["aggs"]:
                        vals = [_ev(a[2], r) for r in grp]
                        row[_name(a)] = {"avg": lambda v: sum(v) / len(v), "max": max, "min": min, "sum": sum, "count": len}[a[1]](vals)
                    data.append(row)
                label = "Aggregate"
            elif o == "Sort":
                data = list(rows)
                for e in reversed(n["orders"]):
                    data.sort(key=lambda r: _ev(e[1], r), reverse=e[2])
                names, label = names_in, "Sort"
            else:
                raise ValueError(o)
        trace.append({"op": label, "detail": logical_tree({k: v for k, v in n.items() if k not in ("child", "left", "right")} | {"op": o})["detail"]
                      if o != "Join" else f"{show(n['cond'])}", "rows_in": rows_in, "rows_out": len(data), "cols": names,
                      "rows": [[r.get(c) for c in names] for r in data[:8]]})
        return names, data

    names, rows = run(plan)
    return names, rows, m, trace


def _display(names: list[str]) -> list[str]:
    short = [n.split(".", 1)[1] if "." in n and "(" not in n else n for n in names]
    out = []
    for n, s in zip(names, short):
        out.append(s if short.count(s) == 1 else n)
    return [o.replace("e.", "").replace("d.", "").replace("m.", "") if "(" in o and short.count(o) == 1 else o for o in out]


def _canon_rows(names, rows, ordered: bool):
    vals = [tuple(round(r[c], 6) if isinstance(r[c], float) else r[c] for c in names) for r in rows]
    return vals if ordered else sorted(vals, key=lambda t: tuple((x is None, str(x)) for x in t))


def run_sql(*, query: str = "slide16", k: int = 30, dno: int = 5, gender: str = "F", folding: bool = True,
            pushdown: bool = True, pruning: bool = True, broadcast: bool = True) -> dict:
    """Parse (preset), analyse, optimise rule by rule, plan physically and execute a query; then
    execute it again with each rule on its own to count what each rule saves."""
    q = build_query(query, k=k, dno=dno, gender=gender)
    rules = {"folding": folding, "pushdown": pushdown, "pruning": pruning}
    analysed = q["plan"]
    opt, stages = optimise(analysed, rules)
    names, rows, metrics, trace = execute_plan(opt)
    ordered = query == "order"
    reference = _canon_rows(*execute_plan(analysed)[:2], ordered)

    variants = []
    for label, on in [("No optimisation", {}), ("ConstantFolding only", {"folding": True}),
                      ("PushDownPredicates only", {"pushdown": True}), ("ColumnPruning only", {"pruning": True}),
                      ("All three rules", {"folding": True, "pushdown": True, "pruning": True})]:
        p, _ = optimise(analysed, on)
        vn, vr, vm, _ = execute_plan(p)
        variants.append({"label": label, "rules": [k2 for k2, v in on.items() if v], **vm,
                         "same_result": _canon_rows(vn, vr, ordered) == reference})
    selected = {"label": "Your selection", "rules": [k2 for k2, v in rules.items() if v], **metrics,
                "same_result": _canon_rows(names, rows, ordered) == reference}
    none = variants[0]
    disp = _display(names)
    out_rows = [[round(r[c], 2) if isinstance(r[c], float) else r[c] for c in names] for r in rows]
    stages_out = [{"rule": "parsed", "name": "Analysed logical plan", "log": [f"Parsed and resolved against the catalog: {', '.join(q['tables'])}"],
                   "tree": logical_tree(analysed), "fired": True}] + stages
    stages_out.append({"rule": "optimised", "name": "Optimised logical plan", "log": [], "tree": logical_tree(opt), "fired": True})
    stages_out.append({"rule": "physical", "name": "Physical plan", "tree": physical_tree(opt, broadcast), "fired": True,
                       "log": ["department is 3 rows, far under the 10 MB autoBroadcastJoinThreshold: BroadcastHashJoin, BuildRight"
                               if broadcast else "autoBroadcastJoinThreshold = −1: both sides shuffled by dno and sorted (SortMergeJoin)"]
                       if any(n["op"] == "Join" for n in _walk(opt)) else []})
    tables = {t: {"cols": SQL_TABLES[t][0], "types": SQL_TABLES[t][1], "rows": [list(r) for r in SQL_TABLES[t][2]]} for t in q["tables"]}
    return {
        "query": {k2: v for k2, v in q.items() if k2 != "plan"}, "rules": rules, "stages": stages_out, "trace": trace,
        "result": {"cols": disp, "rows": out_rows, "same_as_unoptimised": selected["same_result"]},
        "variants": variants + [selected], "metrics": metrics,
        "saving": {"cells": none["cells_read"] - metrics["cells_read"], "cells_pct": 1 - metrics["cells_read"] / none["cells_read"] if none["cells_read"] else 0,
                   "join_rows": none["rows_into_join"] - metrics["rows_into_join"], "evals": none["evals"] - metrics["evals"],
                   "none_cells": none["cells_read"], "none_join_rows": none["rows_into_join"], "none_evals": none["evals"]},
        "tables": tables, "udf": letter_grade_report() if query == "udf" else None,
        "sources": read_sources(),
    }


def _walk(n):
    yield n
    for c in _children(n):
        yield from _walk(c)


def _infer(values: list[str]) -> str:
    def is_int(v):
        try:
            int(v)
            return True
        except ValueError:
            return False
    if all(is_int(v) for v in values):
        return "integer" if all(-2 ** 31 <= int(v) < 2 ** 31 for v in values) else "long"
    try:
        [float(v) for v in values]
        return "double"
    except ValueError:
        return "string"


def read_sources() -> dict:
    """What spark.read produces from the deck's employee_m.csv (p11–13) and employees.json (p17)."""
    lines = [",".join(EMPLOYEE_M_HEADER)] + [",".join(str(v) for v in r) for r in EMPLOYEE_M_ROWS]
    cells = [ln.split(",") for ln in lines]
    no_header = {"cols": [f"_c{i}" for i in range(len(cells[0]))], "types": ["string"] * len(cells[0]), "rows": cells}
    header = {"cols": cells[0], "types": ["string"] * len(cells[0]), "rows": cells[1:]}
    inferred = {"cols": cells[0], "types": [_infer([r[i] for r in cells[1:]]) for i in range(len(cells[0]))], "rows": cells[1:]}
    js = {"cols": ["name", "salary"], "types": ["string", "long"], "rows": [[e["name"], e["salary"]] for e in EMPLOYEES_JSON]}
    return {"csv_lines": lines, "no_header": no_header, "header": header, "infer": inferred, "json": js}


# =========================================================================== Tables and file layout
TBL_STATES = ["CA", "NY", "MA", "TX", "WA"]
_TBL_STATE_P = [0.3, 0.25, 0.2, 0.15, 0.1]
_FIRST = ["Aisha", "Ben", "Carla", "Dev", "Elena", "Farid", "Grace", "Hiro", "Ines", "Jamal", "Kavya", "Liam", "Mei",
          "Nikhil", "Olga", "Pedro", "Quinn", "Ravi", "Sofia", "Tomas"]
_LAST = ["Adams", "Bose", "Chen", "Diaz", "Evans", "Fischer", "Gupta", "Hughes", "Iyer", "Jones", "Khan", "Lopez",
         "Mehta", "Nair", "Okafor", "Patel", "Quinn", "Rossi", "Shah", "Tanaka"]
LOOKUP_EMPNO = 10084                 # employee_m.csv row 2 (DataFrame deck p11)
_JOB_ID = "5e1a"                     # Spark writes a job UUID into each file name; shortened here
_COL_BYTES = {"empno": 4, "dno": 5, "salary": 4}       # INT32 = 4 B; STRING '5' = 4-byte length + 1
_CHUNK_META, _FOOTER, _MAGIC = 64, 400, 8              # approximate Parquet metadata sizes (bytes)
DATA_COLS = ["empno", "name", "dno", "salary"]         # state and gender live in the directory names


def synthetic_employees(n: int, seed: int = 494) -> dict:
    """An employee table shaped like employee_m.csv (empno, name, dno, salary, state, gender). The first
    three rows are the deck's; the rest are generated with a fixed seed, drawing (dno, salary) pairs from
    the course's 30 employees (EMP30) with a little noise."""
    rng = np.random.default_rng(seed)
    pool = [p for node in EMP30_NODES for p in node]
    m = max(0, n - len(EMPLOYEE_M_ROWS))
    pick = rng.integers(0, len(pool), m)
    sal = np.array([pool[i][1] for i in pick], dtype=float) + rng.normal(0, 4000, m)
    sal = (np.clip(sal, 30000, 125000) / 10).round().astype(int) * 10
    gaps = rng.integers(1, 40, m)
    empno = list(r[0] for r in EMPLOYEE_M_ROWS) + (EMPLOYEE_M_ROWS[-1][0] + np.cumsum(gaps)).tolist()
    names = [r[1] for r in EMPLOYEE_M_ROWS] + [f"{_FIRST[a]} {_LAST[b]}" for a, b in zip(rng.integers(0, 20, m), rng.integers(0, 20, m))]
    states = [r[4] for r in EMPLOYEE_M_ROWS] + [TBL_STATES[i] for i in rng.choice(len(TBL_STATES), m, p=_TBL_STATE_P)]
    genders = [r[5] for r in EMPLOYEE_M_ROWS] + ["M" if x < 0.55 else "F" for x in rng.random(m)]
    dno = [r[2] for r in EMPLOYEE_M_ROWS] + [pool[i][0] for i in pick]
    salary = [r[3] for r in EMPLOYEE_M_ROWS] + sal.tolist()
    k = min(n, len(empno))
    return {"empno": np.array(empno[:k], dtype=np.int64), "name": names[:k], "dno": dno[:k], "salary": np.array(salary[:k], dtype=np.int64),
            "state": states[:k], "gender": genders[:k], "n": k}


def _col_bytes(emp, idx, c) -> int:
    if c == "name":
        return int(sum(4 + len(emp["name"][i].encode()) for i in idx))
    return _COL_BYTES[c] * len(idx)


def write_table(emp: dict, *, buckets: int, tasks: int, distribute: bool, sort_salary: bool, rg_rows: int) -> list[dict]:
    """CREATE TABLE employees (...) USING parquet PARTITIONED BY (state, gender) CLUSTERED BY (empno)
    [SORTED BY (salary DESC)] INTO n BUCKETS, then INSERT: every write task writes one file per
    (partition directory, bucket) it holds rows for."""
    n = emp["n"]
    if distribute:
        task = np.array([murmur3_str(emp["gender"][i], murmur3_str(emp["state"][i])) % tasks for i in range(n)])
    else:
        task = np.zeros(n, dtype=int)
        for t, ch in enumerate(_chunks(n, tasks)):
            task[ch] = t
    bucket = partition_of(emp["empno"], buckets)
    groups: dict = defaultdict(list)
    for i in range(n):
        groups[(int(task[i]), emp["state"][i], emp["gender"][i], int(bucket[i]))].append(i)
    files = []
    for (t, st, g, b), idx in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][3], kv[0][0])):
        if sort_salary:
            idx = sorted(idx, key=lambda i: -emp["salary"][i])
        rgs = []
        for s in range(0, len(idx), rg_rows):
            part = idx[s:s + rg_rows]
            sal, eno = emp["salary"][part], emp["empno"][part]
            cb = {c: _col_bytes(emp, part, c) for c in DATA_COLS}
            rgs.append({"rows": part, "n": len(part), "min_salary": int(sal.min()), "max_salary": int(sal.max()),
                        "min_empno": int(eno.min()), "max_empno": int(eno.max()), "col_bytes": cb,
                        "bytes": sum(cb.values()) + _CHUNK_META * len(DATA_COLS)})
        meta = _FOOTER + 2 * _MAGIC
        files.append({"dir": (st, g), "state": st, "gender": g, "bucket": b, "task": t,
                      "name": f"part-{t:05d}-{_JOB_ID}_{b:05d}.c000.snappy.parquet", "rows": idx, "row_groups": rgs,
                      "meta_bytes": meta, "bytes": meta + sum(r["bytes"] for r in rgs)})
    return files


def _clause_sql(clause: str) -> str:
    return {"none": "SELECT * FROM employees",
            "distribute": "SELECT * FROM employees DISTRIBUTE BY state",
            "distribute_sort": "SELECT * FROM employees DISTRIBUTE BY state SORT BY salary DESC",
            "sort": "SELECT * FROM employees SORT BY salary DESC",
            "cluster": "SELECT * FROM employees CLUSTER BY state",
            "order": "SELECT * FROM employees ORDER BY salary DESC"}[clause]


def place_rows(emp: dict, clause: str, rows: int, parts: int) -> dict:
    """Where DISTRIBUTE BY / SORT BY / CLUSTER BY / ORDER BY put the first `rows` rows, with `parts`
    partitions before and after. DISTRIBUTE BY hashes with Murmur3 (pmod(hash(state), n)); ORDER BY
    range-partitions on salary (Spark samples the boundaries; here they are exact quantiles)."""
    idx = list(range(min(rows, emp["n"])))
    sal = {i: int(emp["salary"][i]) for i in idx}
    st = {i: emp["state"][i] for i in idx}
    before = _chunks(len(idx), parts)
    before = [[idx[j] for j in ch] for ch in before] + [[] for _ in range(parts - len(before))]
    if clause in ("distribute", "distribute_sort", "cluster"):
        out = [[] for _ in range(parts)]
        for ch in before:
            for i in ch:
                out[murmur3_str(st[i]) % parts].append(i)
        if clause == "distribute_sort":
            out = [sorted(p, key=lambda i: -sal[i]) for p in out]
        if clause == "cluster":
            out = [sorted(p, key=lambda i: st[i]) for p in out]
    elif clause == "sort":
        out = [sorted(p, key=lambda i: -sal[i]) for p in before]
    elif clause == "order":
        ranked = sorted(idx, key=lambda i: -sal[i])
        out = [[ranked[j] for j in ch] for ch in _chunks(len(ranked), parts)]
        out += [[] for _ in range(parts - len(out))]
    else:
        out = before
    flat = [i for p in out for i in p]
    states_split = sum(1 for s in set(st.values()) if sum(1 for p in out if any(st[i] == s for i in p)) > 1)
    within = all(all(sal[p[j]] >= sal[p[j + 1]] for j in range(len(p) - 1)) for p in out)
    glob = all(sal[flat[j]] >= sal[flat[j + 1]] for j in range(len(flat) - 1))
    row = lambda i: [int(emp["empno"][i]), emp["name"][i], st[i], sal[i]]
    hashes = {s: {"hash": murmur3_str(s), "partition": murmur3_str(s) % parts} for s in sorted(set(st.values()))}
    return {"clause": clause, "sql": _clause_sql(clause), "cols": ["empno", "name", "state", "salary"],
            "before": [[row(i) for i in p] for p in before], "after": [[row(i) for i in p] for p in out],
            "checks": {"states_split": states_split, "sorted_within": within, "globally_sorted": glob,
                       "sizes": [len(p) for p in out], "shuffle": clause in ("distribute", "distribute_sort", "cluster", "order")},
            "state_hash": hashes}


def _tree_summary(files: list[dict], root: str) -> dict:
    dirs: dict = defaultdict(list)
    for f in files:
        dirs[f["dir"]].append(f)
    states = defaultdict(list)
    for (s, g), fs in sorted(dirs.items()):
        states[s].append({"name": f"gender={g}", "kind": "dir", "files": len(fs), "bytes": sum(f["bytes"] for f in fs)})
    return {"name": root, "kind": "dir", "files": len(files), "bytes": sum(f["bytes"] for f in files),
            "children": [{"name": f"state={s}", "kind": "dir", "files": sum(c["files"] for c in cs), "bytes": sum(c["bytes"] for c in cs),
                          "children": cs} for s, cs in states.items()]}


def table_layout(*, rows: int = 4000, seed: int = 494, buckets: int = 4, tasks: int = 2, distribute: bool = False,
                 sort_salary: bool = False, rg_rows: int = 16, f_state: str = "CA", f_gender: str = "all",
                 f_salary: int = 90000, lookup: bool = False, select: str = "two", kind: str = "managed",
                 clause: str = "distribute", demo_rows: int = 12, out_parts: int = 3) -> dict:
    emp = synthetic_employees(rows, seed)
    files = write_table(emp, buckets=buckets, tasks=tasks, distribute=distribute, sort_salary=sort_salary, rg_rows=rg_rows)
    n = emp["n"]
    lookup_bucket = murmur3_int(LOOKUP_EMPNO) % buckets
    sel_cols = DATA_COLS if select == "all" else ["name", "salary"]
    need = list(dict.fromkeys(sel_cols + ["salary"] + (["empno"] if lookup else [])))

    def row_ok(i):
        return ((f_state == "all" or emp["state"][i] == f_state) and (f_gender == "all" or emp["gender"][i] == f_gender)
                and emp["salary"][i] > f_salary and (not lookup or emp["empno"][i] == LOOKUP_EMPNO))

    stage = {"total": 0, "partitions": 0, "buckets": 0, "row_groups": 0, "columns": 0}
    rowsc = {"total": n, "partitions": 0, "buckets": 0, "row_groups": 0, "result": 0}
    rg_total = rg_read = files_read = rg_in_read = 0
    result: list[int] = []
    rg_view = []
    for f in files:
        f_dir_ok = (f_state == "all" or f["state"] == f_state) and (f_gender == "all" or f["gender"] == f_gender)
        f_bucket_ok = (not lookup) or f["bucket"] == lookup_bucket
        stage["total"] += f["bytes"]
        rg_total += len(f["row_groups"])
        f["status"] = "pruned-partition" if not f_dir_ok else "pruned-bucket" if not f_bucket_ok else "read"
        if f_dir_ok:
            stage["partitions"] += f["bytes"]
            rowsc["partitions"] += len(f["rows"])
        if not (f_dir_ok and f_bucket_ok):
            continue
        files_read += 1
        rg_in_read += len(f["row_groups"])
        stage["buckets"] += f["bytes"]
        rowsc["buckets"] += len(f["rows"])
        stage["row_groups"] += f["meta_bytes"]
        stage["columns"] += f["meta_bytes"]
        rgv = []
        for g in f["row_groups"]:
            skip = g["max_salary"] <= f_salary or (lookup and not g["min_empno"] <= LOOKUP_EMPNO <= g["max_empno"])
            g["skipped"] = skip
            rgv.append({"min": g["min_salary"], "max": g["max_salary"], "rows": g["n"], "skipped": skip,
                        "min_empno": g["min_empno"], "max_empno": g["max_empno"]})
            if skip:
                continue
            rg_read += 1
            stage["row_groups"] += g["bytes"]
            stage["columns"] += sum(g["col_bytes"][c] for c in need) + _CHUNK_META * len(need)
            rowsc["row_groups"] += g["n"]
            result += [i for i in g["rows"] if row_ok(i)]
        if len(rg_view) < 36:
            rg_view.append({"file": f"state={f['state']}/gender={f['gender']}/{f['name']}", "bucket": f["bucket"], "row_groups": rgv})
    brute = sorted(i for i in range(n) if row_ok(i))
    rowsc["result"] = len(result)

    # the directory tree with every file (capped per directory)
    dirs: dict = defaultdict(list)
    for f in files:
        dirs[f["dir"]].append(f)
    tree = []
    for st in sorted({d[0] for d in dirs}):
        kids = []
        for g in sorted({d[1] for d in dirs if d[0] == st}):
            fs = dirs[(st, g)]
            kids.append({"name": f"gender={g}", "status": fs[0]["status"] if fs[0]["status"] == "pruned-partition" else "read",
                         "rows": sum(len(f["rows"]) for f in fs), "bytes": sum(f["bytes"] for f in fs), "files_total": len(fs),
                         "files": [{"name": f["name"], "bucket": f["bucket"], "task": f["task"], "rows": len(f["rows"]), "bytes": f["bytes"],
                                    "row_groups": len(f["row_groups"]), "status": f["status"]} for f in fs[:48]]})
        tree.append({"name": f"state={st}", "rows": sum(k["rows"] for k in kids), "children": kids,
                     "status": "read" if any(k["status"] == "read" for k in kids) else "pruned-partition"})
    bucket_rows = np.bincount(partition_of(emp["empno"], buckets), minlength=buckets).tolist()
    sample = [[int(emp["empno"][i]), emp["name"][i], emp["dno"][i], int(emp["salary"][i]), emp["state"][i], emp["gender"][i],
               int(murmur3_int(int(emp["empno"][i]))), int(murmur3_int(int(emp["empno"][i])) % buckets)] for i in range(min(8, n))]
    root = "spark-warehouse/employees" if kind == "managed" else "/data/external/employees"
    summary = _tree_summary(files, root)
    preds = []
    if f_state != "all":
        preds.append(f"state = '{f_state}'")
    if f_gender != "all":
        preds.append(f"gender = '{f_gender}'")
    preds.append(f"salary > {f_salary}")
    if lookup:
        preds.append(f"empno = {LOOKUP_EMPNO}")
    ddl = ("CREATE TABLE employees (empno INT, name STRING, dno STRING, salary INT, state STRING, gender STRING)\n"
           "USING parquet\nPARTITIONED BY (state, gender)\n"
           f"CLUSTERED BY (empno){' SORTED BY (salary DESC)' if sort_salary else ''} INTO {buckets} BUCKETS"
           + ("" if kind == "managed" else f"\nLOCATION '{root}'"))
    insert = ("INSERT INTO employees SELECT * FROM staging" + (" DISTRIBUTE BY state, gender" if distribute else "")
              + f"   -- {tasks} write task{'s' if tasks > 1 else ''}")
    n_dirs = len(dirs)
    return {
        "rows": n, "ddl": ddl, "insert": insert, "sample": sample,
        "sample_cols": ["empno", "name", "dno", "salary", "state", "gender", "hash(empno)", "bucket"],
        "write": {"files": len(files), "dirs": n_dirs, "buckets": buckets, "tasks": tasks, "ideal_files": len({(f["state"], f["gender"], f["bucket"]) for f in files}),
                  "max_files": n_dirs * buckets * tasks, "bytes": stage["total"], "avg_file_bytes": stage["total"] / max(1, len(files)),
                  "row_groups": rg_total, "bucket_rows": bucket_rows, "lookup_bucket": lookup_bucket},
        "tree": tree, "root": root,
        "query": {"sql": f"SELECT {', '.join(sel_cols) if select != 'all' else '*'} FROM employees WHERE " + " AND ".join(preds),
                  "need": need, "lookup_empno": LOOKUP_EMPNO, "lookup_bucket": lookup_bucket},
        "read": {"bytes": stage, "rows": rowsc, "files_read": files_read, "files_total": len(files), "dirs_read": sum(1 for d in dirs if (f_state == "all" or d[0] == f_state) and (f_gender == "all" or d[1] == f_gender)),
                 "dirs_total": n_dirs, "row_groups_read": rg_read, "row_groups_total": rg_total, "row_groups_in_files_read": rg_in_read,
                 "same_as_full_scan": sorted(result) == brute, "result_rows": len(result)},
        "result": {"cols": sel_cols,
                   "rows": [[int(emp[c][i]) if c in ("salary", "empno") else emp[c][i] for c in sel_cols]
                            for i in sorted(result, key=lambda i: -emp["salary"][i])[:12]]},
        "row_groups": rg_view,
        "drop": drop_table(summary, kind, root),
        "placement": place_rows(emp, clause, demo_rows, out_parts),
    }


def drop_table(summary: dict, kind: str, root: str) -> dict:
    meta = {"table": "default.employees", "type": "MANAGED" if kind == "managed" else "EXTERNAL", "location": root,
            "provider": "parquet", "partition_columns": ["state", "gender"], "bucket_columns": ["empno"]}
    after_fs = None if kind == "managed" else summary
    return {"kind": kind, "sql": "DROP TABLE employees", "before": {"metastore": meta, "fs": summary},
            "after": {"metastore": None, "fs": after_fs},
            "note": ("Managed: Spark owns the files, so DROP TABLE deletes the metadata and the directory."
                     if kind == "managed" else "External (LOCATION given): DROP TABLE removes only the metastore entry; the files stay.")}
