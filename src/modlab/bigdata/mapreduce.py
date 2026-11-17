"""A small but real MapReduce engine, the jobs from the IE494 MapReduce decks, and MR joins.

The engine runs every phase for real and keeps the data at each one, so the lab can show it:

    split -> map -> combine -> partition -> shuffle/sort (group) -> reduce

* **map** runs once per input split (one map task per split / DataNode).
* **combine** (optional) runs on each map task's own output, grouped by key.
* **partition** sends each (key, value) to reducer ``partitioner(key, R)``; the matrix of
  records and bytes from each mapper to each reducer is the shuffle traffic.
* **shuffle/sort**: each reducer merges what it received, sorts by the *sort key* and cuts the
  sorted run into reduce calls wherever the *grouping key* changes (Hadoop's sort comparator
  and grouping comparator).
* **reduce** runs once per group.

Bytes are counted as Hadoop's text format writes a pair, ``key<TAB>value\\n``.

Jobs: group-by average (L3 slide 17 / MR deck 1 p4-6, with the naive AVG combiner bug and the
``<sum, count>`` fix of p33), WordCount (MR deck 1 p18/p30), an inverted index (p42), Top-N
(MR deck 2 p24-29) and secondary sort (MR deck 1 p46-47). Joins (MR deck 2 p8-22): the
standard and improved repartition joins and the broadcast join.
"""
from __future__ import annotations

import bisect
import heapq
import math
import re
from collections import Counter, defaultdict
from typing import Any, Callable

from .datasets import EMP30_NODES, MR_CUSTOMERS, MR_ORDERS, TOPN_SPLITS, WORDCOUNT_RANGES, WORDCOUNT_SPLITS

# ============================================================================ formatting and sizes

def fmt(x: Any) -> str:
    """How a key or value is written in the text output format."""
    if x is None:
        return "NULL"
    if isinstance(x, bool):
        return str(x).lower()
    if isinstance(x, float):
        return f"{x:.2f}".rstrip("0").rstrip(".") if not x.is_integer() else str(int(x))
    if isinstance(x, tuple):
        return "<" + ",".join(fmt(v) for v in x) + ">"
    if isinstance(x, list):
        return "[" + ",".join(fmt(v) for v in x) + "]"
    return str(x)


def nbytes(k: Any, v: Any) -> int:
    return len(f"{fmt(k)}\t{fmt(v)}\n".encode())


def java_hash(x: Any) -> int:
    """Java hashCode for the key types used here (Integer, String, null, and a pair key)."""
    if x is None:
        return 0
    if isinstance(x, bool):
        return 1231 if x else 1237
    if isinstance(x, int):
        return ((x + 2 ** 31) % 2 ** 32) - 2 ** 31
    if isinstance(x, str):
        h = 0
        for ch in x:
            h = (31 * h + ord(ch)) & 0xFFFFFFFF
        return h - 2 ** 32 if h >= 2 ** 31 else h
    if isinstance(x, tuple):                       # like Objects.hash(a, b)
        h = 1
        for v in x:
            h = (31 * h + java_hash(v)) & 0xFFFFFFFF
        return h - 2 ** 32 if h >= 2 ** 31 else h
    return java_hash(str(x))


def hash_partition(key: Any, R: int) -> int:
    """Hadoop's HashPartitioner: (key.hashCode() & Integer.MAX_VALUE) % R."""
    return (java_hash(key) & 0x7FFFFFFF) % R


def letter_ranges(R: int) -> list[tuple[str, str]]:
    """Split A-Z into R contiguous ranges; R = 3 gives A-I, J-Q, R-Z as in the IBM figure (MR deck 1 p18)."""
    cuts = [round(i * 26 / R) for i in range(R + 1)]
    return [(chr(65 + cuts[i]), chr(65 + cuts[i + 1] - 1)) for i in range(R) if cuts[i + 1] > cuts[i]]


class RangePartitioner:
    """Range partitioning. String keys go by first letter (A-I / J-Q / R-Z for 3 reducers).
    Numeric keys use split points sampled from the map-output keys, so each reducer gets about the
    same number of *records* (Hadoop's TotalOrderPartitioner); a single hot key cannot be split."""

    def __init__(self, R: int, sample: list):
        self.R = R
        self.kind = "letters" if sample and isinstance(sample[0], str) else "sampled"
        if self.kind == "letters":
            self.ranges = letter_ranges(R)
            self.cuts = [ord(a) - 65 for a, _ in self.ranges[1:]]
        else:
            s = sorted(x for x in sample if x is not None)
            self.cuts = [s[min(len(s) - 1, round(len(s) * i / R))] for i in range(1, R)] if s else []
            self.ranges = None

    def __call__(self, key: Any, R: int) -> int:
        if key is None:
            return 0
        if self.kind == "letters":
            c = ord(str(key)[:1].upper() or "A") - 65
            return min(R - 1, bisect.bisect_right(self.cuts, max(0, min(25, c))))
        return min(R - 1, bisect.bisect_right(self.cuts, key))

    def describe(self) -> list[str]:
        if self.kind == "letters":
            return [f"{a}–{b}" for a, b in self.ranges]
        lo = ["min"] + [fmt(c) for c in self.cuts]
        hi = [fmt(c) for c in self.cuts] + ["max"]
        return [f"[{a}, {b})" if b != "max" else f"[{a}, max]" for a, b in zip(lo, hi)]


# ============================================================================ the engine

def run(splits: list[list], map_task: Callable[[list], list], R: int, partitioner: Callable[[Any, int], int],
        combiner: Callable[[Any, list], list] | None = None, reducer: Callable[[Any, list], list] | None = None,
        sort_key: Callable[[Any], Any] = lambda k: k, group_key: Callable[[Any], Any] = lambda k: k,
        combine_key: Callable[[Any], Any] | None = None) -> dict:
    """Run one MapReduce job and keep every intermediate stage."""
    M = len(splits)
    map_out = [map_task(s) for s in splits]

    def group_sorted(pairs, gk):
        pairs = sorted(pairs, key=lambda kv: _sortable(sort_key(kv[0])))
        groups = []
        for k, v in pairs:
            if groups and gk(groups[-1][0]) == gk(k):
                groups[-1][1].append(v)
            else:
                groups.append((k, [v]))
        return groups

    if combiner:
        comb_out = []
        for pairs in map_out:
            out = []
            for k, vs in group_sorted(pairs, combine_key or (lambda k: k)):
                out.extend(combiner(k, vs))
            comb_out.append(out)
    else:
        comb_out = None
    sent = comb_out if comb_out is not None else map_out

    parts = [[[] for _ in range(R)] for _ in range(M)]
    for m, pairs in enumerate(sent):
        for k, v in pairs:
            parts[m][partitioner(k, R)].append((k, v))
    traffic = [[{"records": len(parts[m][r]), "bytes": sum(nbytes(k, v) for k, v in parts[m][r])} for r in range(R)]
               for m in range(M)]

    received = [[kv for m in range(M) for kv in parts[m][r]] for r in range(R)]
    shuffle = [group_sorted(rec, group_key) for rec in received]
    reduce_out = []
    for groups in shuffle:
        out = []
        for k, vs in groups:
            out.extend(reducer(k, vs) if reducer else [(k, v) for v in vs])
        reduce_out.append(out)

    def pairs_json(pairs):
        return [[fmt(k), fmt(v)] for k, v in pairs]

    per_red_records = [len(x) for x in received]
    per_red_bytes = [sum(nbytes(k, v) for k, v in x) for x in received]
    return {
        "M": M, "R": R,
        "splits": [[fmt(x) for x in s] for s in splits],
        "map": [pairs_json(p) for p in map_out],
        "combine": [pairs_json(p) for p in comb_out] if comb_out is not None else None,
        "partition": [[pairs_json(parts[m][r]) for r in range(R)] for m in range(M)],
        "traffic": traffic,
        "shuffle": [[{"key": fmt(k), "values": [fmt(v) for v in vs]} for k, vs in g] for g in shuffle],
        "reduce": [pairs_json(p) for p in reduce_out],
        "raw_output": [kv for p in reduce_out for kv in p],
        "stats": {
            "input_records": sum(len(s) for s in splits),
            "map_records": sum(len(p) for p in map_out),
            "map_bytes": sum(nbytes(k, v) for p in map_out for k, v in p),
            "shuffled_records": sum(per_red_records), "shuffled_bytes": sum(per_red_bytes),
            "reducer_records": per_red_records, "reducer_bytes": per_red_bytes,
            "reduce_calls": [len(g) for g in shuffle],
            "output_records": sum(len(p) for p in reduce_out),
        },
    }


def _sortable(k):
    """Sort NULL first and keep mixed tuples comparable."""
    if k is None:
        return (0,)
    if isinstance(k, tuple):
        return (1,) + tuple(_sortable(x) for x in k)
    return (1, k)


def skew(records: list[int]) -> float:
    """Busiest reducer over the mean load (1 = perfectly even)."""
    busy = [r for r in records]
    mean = sum(busy) / len(busy) if busy else 0
    return max(busy) / mean if mean else 0.0


# ============================================================================ datasets unique to these views
# Toy documents for the inverted index (MR deck 1 p42 gives the algorithm but no example documents).
DOCS = {
    "d1": "Hadoop stores big files as HDFS blocks",
    "d2": "Spark caches big datasets in memory",
    "d3": "MapReduce reads HDFS blocks and writes HDFS files",
}
# Two rows added to the join example when "unmatched rows" is on, so inner and outer joins differ.
JOIN_EXTRA_CUSTOMER = (113, "India", "Maharashtra")       # a customer with no orders
JOIN_EXTRA_ORDER = (20, 115, 610)                         # an order whose customer is not in the table

JOBS = {
    "avg": {"label": "Average salary per department", "slides": "L3 slide 17 · MR deck 1 p4–6, p32–34",
            "task": "SELECT dno, AVG(salary) FROM emp GROUP BY dno, over 30 employees stored on 3 DataNodes."},
    "wordcount": {"label": "Word count", "slides": "MR deck 1 p18 (no combiner), p30 (with combiner)",
                  "task": "Count every word in three input splits; reducers own the ranges A–I, J–Q and R–Z."},
    "index": {"label": "Inverted index", "slides": "MR deck 1 p42–43",
              "task": "For every word, list the documents that contain it."},
    "topn": {"label": "Top-N customers", "slides": "MR deck 2 p23–30",
             "task": "Find the N customers with the largest order totals, with a min-heap of size N per mapper and one reducer."},
    "secsort": {"label": "Secondary sort", "slides": "MR deck 1 p46–47",
                "task": "List each department's salaries in descending order, with one reduce call per department."},
}


def _emp_splits():
    return [list(node) for node in EMP30_NODES]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+", text)


# ============================================================================ expected outputs (ground truth, no MapReduce)

def expected_avg() -> dict[int, float]:
    tot, cnt = defaultdict(float), Counter()
    for node in EMP30_NODES:
        for d, s in node:
            tot[d] += s
            cnt[d] += 1
    return {d: tot[d] / cnt[d] for d in sorted(tot)}


def expected_wordcount() -> dict[str, int]:
    return dict(sorted(Counter(w for s in WORDCOUNT_SPLITS for w in _tokens(s)).items()))


def expected_index() -> dict[str, list[str]]:
    out = defaultdict(set)
    for d, t in DOCS.items():
        for w in _tokens(t):
            out[w.lower()].add(d)
    return {w: sorted(ds) for w, ds in sorted(out.items())}


def expected_topn(n: int) -> list[tuple[str, int]]:
    rows = [r for s in TOPN_SPLITS for r in s]
    return sorted(rows, key=lambda r: -r[1])[:n]


def expected_secsort() -> dict[int, list[int]]:
    out = defaultdict(list)
    for node in EMP30_NODES:
        for d, s in node:
            out[d].append(s)
    return {d: sorted(v, reverse=True) for d, v in sorted(out.items())}


# ============================================================================ jobs

COMBINERS = ("none", "reducer", "proper")
PARTITIONERS = ("hash", "range")


def _partitioner(kind: str, R: int, sample_keys: list, natural: Callable = lambda k: k):
    if kind == "range":
        rp = RangePartitioner(R, [natural(k) for k in sample_keys])
        return (lambda k, R_: rp(natural(k), R_)), rp.describe()
    return (lambda k, R_: hash_partition(natural(k), R_)), [f"hash mod {R} = {r}" for r in range(R)]


def run_job(job: str = "avg", combiner: str = "none", partitioner: str = "range", reducers: int = 2,
            group_cmp: bool = True, natural_partition: bool = True, topn: int = 3) -> dict:
    """Run one of the lab's jobs. ``combiner``: none | reducer (reuse the reducer as combiner) | proper."""
    R = reducers
    natural = lambda k: k
    sort_key = lambda k: k
    group_key = lambda k: k
    notes: list[str] = []
    comb = None

    if job == "avg":
        splits = _emp_splits()
        if combiner == "proper":
            map_task = lambda recs: [(d, (s, 1)) for d, s in recs]
            comb = lambda k, vs: [(k, (sum(v[0] for v in vs), sum(v[1] for v in vs)))]
            red = lambda k, vs: [(k, sum(v[0] for v in vs) / sum(v[1] for v in vs))]
        else:
            map_task = lambda recs: [(d, s) for d, s in recs]
            red = lambda k, vs: [(k, sum(vs) / len(vs))]
            if combiner == "reducer":
                comb = red                         # the bug: AVG is not associative
        exp = expected_avg()
        expected = [[fmt(k), fmt(v)] for k, v in exp.items()]

    elif job == "wordcount":
        splits = [_tokens(s) for s in WORDCOUNT_SPLITS]
        map_task = lambda recs: [(w, 1) for w in recs]
        red = lambda k, vs: [(k, sum(vs))]
        comb = red if combiner != "none" else None
        exp = expected_wordcount()
        expected = [[k, str(v)] for k, v in exp.items()]

    elif job == "index":
        splits = [[(d, w.lower()) for w in _tokens(t)] for d, t in DOCS.items()]
        map_task = lambda recs: [(w, d) for d, w in recs]
        red = lambda k, vs: [(k, ",".join(sorted(set(",".join(vs).split(",")))))]
        comb = red if combiner != "none" else None
        exp = expected_index()
        expected = [[k, ",".join(v)] for k, v in exp.items()]

    elif job == "topn":
        splits = [list(s) for s in TOPN_SPLITS]
        N = topn

        def heap_top(rows):
            h: list = []
            for c, a in rows:
                heapq.heappush(h, (a, c))          # min-heap on amount
                if len(h) > N:
                    heapq.heappop(h)               # drop the smallest: keeps the N largest
            return sorted(h, reverse=True)

        if combiner == "none":
            map_task = lambda recs: [(None, (c, a)) for c, a in recs]
        else:                                      # in-mapper min-heap, emitted in cleanup()
            map_task = lambda recs: [(None, (c, a)) for a, c in heap_top(recs)]
        red = lambda k, vs: [(None, (c, a)) for a, c in heap_top(vs)]
        exp = expected_topn(N)
        expected = [["NULL", fmt(r)] for r in exp]
        if combiner == "reducer":
            notes.append("For Top-N the reducer's heap logic is exactly the in-mapper heap, so both settings send N records per mapper.")
        if R > 1:
            notes.append(f"Every map output key is NULL, so all records land on one reducer; the other {R - 1} sit idle.")

    elif job == "secsort":
        splits = _emp_splits()
        map_task = lambda recs: [((d, s), s) for d, s in recs]
        sort_key = lambda k: (k[0], -k[1])         # sort comparator: dno ASC, salary DESC
        group_key = (lambda k: k[0]) if group_cmp else (lambda k: k)
        natural = (lambda k: k[0]) if natural_partition else (lambda k: k)
        red = lambda k, vs: [(k[0], list(vs))]
        if combiner != "none":
            notes.append("Secondary sort has nothing to combine: every salary must reach the reducer. The combiner setting is ignored.")
        exp = expected_secsort()
        expected = [[str(k), fmt(v)] for k, v in exp.items()]
    else:
        raise ValueError(f"unknown job {job!r}")

    # partition on the natural key; sample keys come from the plain map output
    sample = [k for p in (map_task(s) for s in splits) for k, _ in p]
    part, part_desc = _partitioner(partitioner, R, sample, natural)
    res = run(splits, map_task, R, part, comb, red, sort_key, group_key)
    # the same job with no combiner (and no in-mapper heap), for the "saved by combining" figures
    if combiner != "none" and job != "secsort":
        base_map = {"avg": lambda recs: [(d, s) for d, s in recs],
                    "topn": lambda recs: [(None, (c, a)) for c, a in recs]}.get(job, map_task)
        base_red = (lambda k, vs: [(k, sum(vs) / len(vs))]) if job == "avg" else red
        base = run(splits, base_map, R, part, None, base_red, sort_key, group_key)
    else:
        base = res

    output = sorted(res.pop("raw_output"), key=lambda kv: _sortable(kv[0]) if job != "topn" else -kv[1][1])
    base.pop("raw_output", None)
    out_rows = [[fmt(k), fmt(v)] for k, v in output]
    check = _check(job, output, topn)
    st, bs = res["stats"], base["stats"]
    keys_by_count = Counter(k if job != "secsort" else k[0] for p in (map_task(s) for s in splits) for k, _ in p)
    hot_key, hot_n = (keys_by_count.most_common(1)[0] if keys_by_count else (None, 0))
    return {
        "job": job, "info": JOBS[job], "combiner": combiner, "partitioner": partitioner, "R": R,
        "group_cmp": group_cmp, "natural_partition": natural_partition, "topn": topn,
        "partitions": part_desc, "stages": res, "notes": notes,
        "output": out_rows, "expected": expected, "check": check,
        "shuffle": {"records": st["shuffled_records"], "bytes": st["shuffled_bytes"],
                    "records_no_combiner": bs["shuffled_records"], "bytes_no_combiner": bs["shuffled_bytes"],
                    "saved_records": bs["shuffled_records"] - st["shuffled_records"],
                    "saved_bytes": bs["shuffled_bytes"] - st["shuffled_bytes"],
                    "reducer_records_no_combiner": bs["reducer_records"]},
        "load": {"reducer_records": st["reducer_records"], "reducer_bytes": st["reducer_bytes"],
                 "reduce_calls": st["reduce_calls"], "skew": skew(st["reducer_records"]),
                 "hot_key": fmt(hot_key), "hot_key_records": hot_n, "map_records": st["map_records"],
                 "idle_reducers": sum(1 for r in st["reducer_records"] if r == 0)},
    }


def _check(job: str, output: list, topn: int) -> dict:
    """Compare the job's output with the ground truth computed without MapReduce."""
    rows = []
    if job == "avg":
        exp = expected_avg()
        got = {k: v for k, v in output}
        for k, v in exp.items():
            g = got.get(k)
            rows.append({"key": fmt(k), "expected": fmt(v), "got": fmt(g), "ok": g is not None and abs(g - v) < 1e-6,
                         "error": None if g is None else g - v})
    elif job == "wordcount":
        got = dict(output)
        rows = [{"key": k, "expected": str(v), "got": fmt(got.get(k)), "ok": got.get(k) == v} for k, v in expected_wordcount().items()]
    elif job == "index":
        got = dict(output)
        rows = [{"key": k, "expected": ",".join(v), "got": fmt(got.get(k)), "ok": got.get(k) == ",".join(v)}
                for k, v in expected_index().items()]
    elif job == "topn":
        exp = expected_topn(topn)
        got = [v for _, v in output]
        for i, e in enumerate(exp):
            g = got[i] if i < len(got) else None
            rows.append({"key": f"#{i + 1}", "expected": fmt(e), "got": fmt(g), "ok": g == e})
    elif job == "secsort":
        exp = expected_secsort()
        by = defaultdict(list)
        for k, v in output:
            by[k].append(v)
        for k, v in exp.items():
            g = by.get(k, [])
            rows.append({"key": str(k), "expected": fmt(v), "got": " + ".join(fmt(x) for x in g) or "—",
                         "ok": len(g) == 1 and g[0] == v, "calls": len(g)})
    ok = all(r["ok"] for r in rows) and (job != "secsort" or len(output) == len(rows))
    return {"rows": rows, "all_ok": ok, "n_ok": sum(r["ok"] for r in rows), "n": len(rows)}


def avg_of_averages() -> dict:
    """What the naive AVG combiner returns per department, against the true average (MR deck 1 p32)."""
    exp = expected_avg()
    per_node = []
    for node in EMP30_NODES:
        g = defaultdict(list)
        for d, s in node:
            g[d].append(s)
        per_node.append({d: sum(v) / len(v) for d, v in g.items()})
    naive = {}
    for d in exp:
        avs = [p[d] for p in per_node if d in p]
        naive[d] = sum(avs) / len(avs)
    return {"dno": list(exp), "true": [exp[d] for d in exp], "naive": [naive[d] for d in exp],
            "per_node": [{str(d): v for d, v in p.items()} for p in per_node],
            "counts": {d: sum(1 for n in EMP30_NODES for x in n if x[0] == d) for d in exp}}


# ============================================================================ joins (MR deck 2 p8-22)

JOIN_ALGOS = ("standard", "improved", "broadcast")
JOIN_TYPES = ("inner", "left", "right", "full")
# L = orders (the log / foreign-key side, many rows per key), R = customers (reference / primary-key side).


def join_tables(extra: bool = False):
    cust = list(MR_CUSTOMERS) + ([JOIN_EXTRA_CUSTOMER] if extra else [])
    orders = list(MR_ORDERS) + ([JOIN_EXTRA_ORDER] if extra else [])
    return cust, orders


def _row(c, o):
    """Output row (cno, country, state, ordrno, amount) as on MR deck 2 p12; NULL for a missing side."""
    cno = c[0] if c else o[1]
    return (cno, c[1] if c else None, c[2] if c else None, o[0] if o else None, o[2] if o else None)


def expected_join(join: str, extra: bool = False) -> list[tuple]:
    cust, orders = join_tables(extra)
    cmap = {c[0]: c for c in cust}
    rows = []
    for o in orders:
        c = cmap.get(o[1])
        if c or join in ("left", "full"):
            rows.append(_row(c, o))
    if join in ("right", "full"):
        have = {o[1] for o in orders}
        rows += [_row(c, None) for c in cust if c[0] not in have]
    return sorted(rows, key=_join_sort)


def _join_sort(r):
    return tuple((0, x) if x is None else (1, x) for x in (r[0], r[3]))


def _order_splits(orders, n_splits=3):
    k = math.ceil(len(orders) / n_splits)
    return [orders[i:i + k] for i in range(0, len(orders), k)]


def run_join(algo: str = "standard", join: str = "inner", reducers: int = 3, extra: bool = False,
             group_cmp: bool = True, natural_partition: bool = True, partitioner: str = "hash") -> dict:
    cust, orders = join_tables(extra)
    # Map tasks finish in any order, so the reducer cannot rely on where the customer record lands.
    # Putting the customers task third reproduces MR deck 2 p12, where the C record sits mid-list.
    osplits = [[("O", o) for o in s] for s in _order_splits(orders)]
    splits = osplits[:2] + [[("C", c) for c in cust]] + osplits[2:]
    split_names = [f"orders.csv split {i + 1}" for i in range(2)] + ["customers.csv"] + \
                  [f"orders.csv split {i + 3}" for i in range(len(osplits) - 2)]
    R = reducers
    keep_l = join in ("left", "full")       # keep orders without a customer
    keep_r = join in ("right", "full")      # keep customers without orders
    calls = []                              # per reduce call: buffer sizes, for the memory comparison
    notes = []

    if algo == "broadcast":
        HR = {c[0]: c for c in cust}        # map_init(): build a hash table from R (customers)
        rbytes = sum(len(",".join(map(str, c)) + "\n") for c in cust)
        mapped, out_parts = [], []
        order_splits = [sp for sp, nm in zip(splits, split_names) if nm != "customers.csv"]
        for s in order_splits:
            out = []
            for _, o in s:
                c = HR.get(o[1])
                if c is not None or keep_l:
                    out.append(_row(c, o))
            mapped.append(out)
            out_parts.append(out)
        output = sorted([r for p in out_parts for r in p], key=_join_sort)
        if keep_r:
            notes.append("A map-only broadcast join cannot emit customers that have no orders: each mapper sees one split "
                         "of orders, so no mapper knows that nobody matched a customer. Right and full outer joins need "
                         "a repartition join (or a second pass).")
        exp = expected_join(join, extra)
        stages = {
            "M": len(order_splits), "R": 0, "splits": [[fmt(x[1]) for x in s] for s in order_splits],
            "split_names": [nm for nm in split_names if nm != "customers.csv"], "hash_table": [fmt(c) for c in cust],
            "map": [[["NULL", fmt(r)] for r in p] for p in mapped],
            "traffic": [], "shuffle": [], "reduce": [],
        }
        net = {"shuffled_records": 0, "shuffled_bytes": 0, "broadcast_bytes": rbytes * len(order_splits),
               "broadcast_records": len(cust) * len(order_splits)}
        mem = {"peak_records": len(cust), "peak_bytes": rbytes, "where": "each mapper holds all of customers"}
        return _join_result(algo, join, R, extra, group_cmp, natural_partition, partitioner, stages, output, exp, net, mem, notes,
                            reduce_calls=0, cust=cust, orders=orders)

    if algo == "standard":
        def map_task(recs):
            out = []
            for tag, r in recs:
                if tag == "C":
                    out.append((r[0], ("C", r[1], r[2])))
                else:
                    out.append((r[1], ("O", r[0], r[2])))
            return out
        sort_key = lambda k: k
        group_key = lambda k: k
        natural = lambda k: k

        def red(k, vs):
            c, buf = None, []                 # the reducer buffers every order (L) record
            for v in vs:
                if v[0] == "C":
                    c = (k, v[1], v[2])
                else:
                    buf.append((v[1], k, v[2]))
            calls.append({"key": fmt(k), "buffered": len(buf) + (1 if c else 0)})
            if not buf:
                return [(None, _row(c, None))] if (c and keep_r) else []
            return [(None, _row(c, o)) for o in buf if c or keep_l]
    else:                                   # improved: composite key (cid, tag), tag C < O
        def map_task(recs):
            out = []
            for tag, r in recs:
                if tag == "C":
                    out.append(((r[0], "C"), ("C", r[1], r[2])))
                else:
                    out.append(((r[1], "O"), ("O", r[0], r[2])))
            return out
        sort_key = lambda k: k
        group_key = (lambda k: k[0]) if group_cmp else (lambda k: k)
        natural = (lambda k: k[0]) if natural_partition else (lambda k: k)

        def red(k, vs):
            # BR <- LIST_V'[0] (p20), with the tag checked so a key with no customer is handled
            c = (k[0], vs[0][1], vs[0][2]) if vs[0][0] == "C" else None
            rest = vs[1:] if c else vs
            calls.append({"key": fmt(k), "buffered": 1 if c else 0})
            out = []
            for v in rest:
                if v[0] != "O":
                    continue
                if c or keep_l:
                    out.append((None, _row(c, (v[1], k[0], v[2]))))
            if not rest and c and keep_r:
                out.append((None, _row(c, None)))
            return out

    sample = [k for s in splits for k, _ in map_task(s)]
    part, part_desc = _partitioner(partitioner, R, sample, natural)
    res = run(splits, map_task, R, part, None, red, sort_key, group_key)
    output = sorted([v for _, v in res.pop("raw_output")], key=_join_sort)
    exp = expected_join(join, extra)
    res["split_names"] = split_names
    res["partitions"] = part_desc
    st = res["stats"]
    net = {"shuffled_records": st["shuffled_records"], "shuffled_bytes": st["shuffled_bytes"], "broadcast_bytes": 0,
           "broadcast_records": 0}
    peak = max((c["buffered"] for c in calls), default=0)
    rec_bytes = st["shuffled_bytes"] / max(1, st["shuffled_records"])
    mem = {"peak_records": peak, "peak_bytes": round(peak * rec_bytes),
           "where": "orders of one customer, buffered" if algo == "standard" else "one customer record, then orders stream past"}
    if algo == "improved" and not group_cmp:
        notes.append("Without a grouping comparator each (cid, tag) is its own reduce call: the customer and its orders "
                     "arrive in different calls and never meet.")
    if algo == "improved" and not natural_partition:
        notes.append("Partitioning on the whole composite key can send a customer and its orders to different reducers.")
    return _join_result(algo, join, R, extra, group_cmp, natural_partition, partitioner, res, output, exp, net, mem, notes,
                        reduce_calls=sum(st["reduce_calls"]), cust=cust, orders=orders, calls=calls)


def _join_result(algo, join, R, extra, group_cmp, natural_partition, partitioner, stages, output, exp, net, mem, notes,
                 reduce_calls, cust, orders, calls=None):
    got, want = Counter(output), Counter(exp)
    missing = list((want - got).elements())
    spurious = list((got - want).elements())
    return {
        "algo": algo, "join": join, "R": R, "extra": extra, "group_cmp": group_cmp, "natural_partition": natural_partition,
        "partitioner": partitioner, "stages": stages, "notes": notes,
        "columns": ["cno", "country", "state", "ordrno", "amount"],
        "output": [[fmt(x) for x in r] for r in output], "expected": [[fmt(x) for x in r] for r in exp],
        "check": {"ok": not missing and not spurious, "rows": len(output), "expected_rows": len(exp),
                  "missing": [[fmt(x) for x in r] for r in missing], "spurious": [[fmt(x) for x in r] for r in spurious]},
        "network": net, "memory": mem, "reduce_calls": reduce_calls, "calls": calls or [],
        "tables": {"customers": [list(c) for c in cust], "orders": [list(o) for o in orders]},
    }


def join_scale(log_cust: float = 0.0, log_orders: float = 0.0, nodes: int = 10, heap_mb: float = 1024,
               split_mb: float = 128) -> dict:
    """Network bytes and peak memory of the three join strategies at a synthetic scale.

    Record sizes are measured on the real tagged records of the slide example; the tables are
    scaled to 3·10^x customers and 9·10^y orders with the same key distribution (orders spread
    evenly over customers). Repartition joins shuffle both tables; the broadcast join copies the
    customers table once to every node that runs a map task over the orders."""
    cust, orders = join_tables(False)
    c_b = sum(len(f"{c[0]}\t" + fmt(("C", c[1], c[2])) + "\n") for c in cust) / len(cust)
    o_b = sum(len(f"{o[1]}\t" + fmt(("O", o[0], o[2])) + "\n") for o in orders) / len(orders)
    raw_c = sum(len(",".join(map(str, c)) + "\n") for c in cust) / len(cust)
    raw_o = sum(len(",".join(map(str, o)) + "\n") for o in orders) / len(orders)

    def at(nc, no):
        L_bytes = no * raw_o
        maps = max(1, math.ceil(L_bytes / (split_mb * 2 ** 20)))
        copies = min(nodes, maps)
        rep = nc * c_b + no * o_b
        bc = nc * raw_c * copies
        per_key = no / nc
        return {"maps": maps, "copies": copies, "repartition_bytes": rep, "broadcast_bytes": bc,
                "standard_mem": max(1.0, per_key) * o_b, "improved_mem": c_b, "broadcast_mem": nc * raw_c}

    nc, no = 3 * 10 ** log_cust, 9 * 10 ** log_orders
    now = at(nc, no)
    xs = [3 * 10 ** (k / 10) for k in range(0, 81)]
    curve = [at(x, no) for x in xs]
    fits = now["broadcast_mem"] <= heap_mb * 2 ** 20
    winner = "broadcast" if fits and now["broadcast_bytes"] < now["repartition_bytes"] else "repartition"
    # crossover: nc * raw_c * copies = nc * c_b + no * o_b  ->  nc = no * o_b / (raw_c * copies - c_b)
    denom = raw_c * now["copies"] - c_b
    cross = no * o_b / denom if denom > 0 else None
    return {
        "customers": nc, "orders": no, "nodes": nodes, "heap_mb": heap_mb, "split_mb": split_mb,
        "bytes_per_record": {"customer_tagged": c_b, "order_tagged": o_b, "customer_raw": raw_c, "order_raw": raw_o},
        "now": now, "winner": winner, "broadcast_fits": fits, "crossover_customers": cross,
        "max_broadcast_customers": heap_mb * 2 ** 20 / raw_c,
        "curve": {"customers": xs, "repartition_gb": [c["repartition_bytes"] / 2 ** 30 for c in curve],
                  "broadcast_gb": [c["broadcast_bytes"] / 2 ** 30 for c in curve]},
    }


WORDCOUNT_FIGURE_RANGES = WORDCOUNT_RANGES
