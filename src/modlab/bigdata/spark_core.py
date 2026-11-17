"""A mini RDD engine: a plain-Python simulation of Spark's behaviour (no PySpark).

It keeps the parts of Spark that the IE494 lessons are about, and records everything so it
can be shown:

* **Partitions and placement.** ``parallelize`` / ``textFile`` slice the data the way Spark's
  ``ParallelCollectionRDD`` does; partition *i* lives on executor ``i % executors``.
* **Lazy lineage.** Transformations only build RDD objects with their dependencies; nothing is
  computed until an action calls :meth:`Context.run_job`.
* **Dependencies.** Narrow (one-to-one, range for ``union``, groups for ``coalesce``) or shuffle.
  ``join`` is a cogroup: an input already hash-partitioned like the output is a narrow
  dependency (co-partitioned), any other input is shuffled.
* **Shuffles.** The map side runs once per map partition (with map-side combining for
  ``reduceByKey`` / ``aggregateByKey``, none for ``groupByKey``), buckets records with Spark's
  ``HashPartitioner`` (Java ``hashCode`` mod n) and keeps the bucket files, as Spark keeps shuffle
  files, so later jobs skip finished map stages. Every record written is counted with its size.
* **Stages.** :meth:`Context.stages` cuts the lineage at shuffle dependencies the way the
  DAGScheduler does, one task per partition of a stage's last RDD.
* **Persistence and recovery.** ``persist()`` keeps computed partitions; losing an executor drops
  its cached partitions (and, optionally, its shuffle files) and the next job recomputes exactly
  the missing pieces from lineage.

Record sizes are the length of the record's text form in bytes (a stand-in for serialized size).
"""
from __future__ import annotations

import copy
from collections import Counter
from typing import Any, Callable


# --------------------------------------------------------------------------- hashing / display
def java_hash(k: Any) -> int:
    """Java hashCode, as Spark's HashPartitioner sees keys from the Java API."""
    if isinstance(k, bool):
        return 1231 if k else 1237
    if isinstance(k, int):
        return ((k + 2 ** 31) % 2 ** 32) - 2 ** 31
    if isinstance(k, str):
        h = 0
        for ch in k:
            h = (31 * h + ord(ch)) & 0xFFFFFFFF
        return h - 2 ** 32 if h >= 2 ** 31 else h
    if isinstance(k, tuple):
        h = 1
        for x in k:
            h = (31 * h + java_hash(x)) & 0xFFFFFFFF
        return h - 2 ** 32 if h >= 2 ** 31 else h
    raise TypeError(f"no Java hash for {type(k).__name__}")


def show(rec: Any) -> str:
    """Text form of a record (Python repr, floats as Spark prints them)."""
    if isinstance(rec, float):
        return repr(round(rec, 4))
    if isinstance(rec, tuple) and not hasattr(rec, "_fields"):
        inner = ", ".join(show(x) for x in rec)
        return f"({inner}{',' if len(rec) == 1 else ''})"
    if isinstance(rec, list):
        return "[" + ", ".join(show(x) for x in rec) + "]"
    return repr(rec)


def size(rec: Any) -> int:
    return len(show(rec).encode())


class HashPartitioner:
    def __init__(self, n: int):
        self.n = n

    def partition(self, key) -> int:
        return java_hash(key) % self.n          # Python's % is already non-negative (nonNegativeMod)

    def __eq__(self, other):
        return isinstance(other, HashPartitioner) and other.n == self.n

    def __hash__(self):
        return hash(("hash", self.n))

    def __repr__(self):
        return f"HashPartitioner({self.n})"


def slices(n_items: int, n: int) -> list[tuple[int, int]]:
    """ParallelCollectionRDD.slice: partition i holds positions [i·len/n, (i+1)·len/n)."""
    return [(i * n_items // n, (i + 1) * n_items // n) for i in range(n)]


# --------------------------------------------------------------------------- dependencies
class Narrow:
    """Narrow dependency: child partition i reads parent partitions parents(i)."""
    kind = "narrow"

    def __init__(self, parent: "RDD", parents: Callable[[int], list[int]], name: str = "OneToOne"):
        self.parent, self.parents, self.name = parent, parents, name


class Shuffle:
    """Shuffle (wide) dependency with an optional map-side combine."""
    kind = "shuffle"

    def __init__(self, ctx: "Context", parent: "RDD", partitioner, combine=None, round_robin=False):
        self.parent, self.partitioner, self.combine, self.round_robin = parent, partitioner, combine, round_robin
        self.id = len(ctx.shuffles)
        self.name = "Shuffle"
        ctx.shuffles.append(self)


# --------------------------------------------------------------------------- RDD
class RDD:
    def __init__(self, ctx: "Context", n: int, deps: list, op: str, compute: Callable[[int], list],
                 partitioner=None, name: str | None = None):
        self.ctx, self.id = ctx, len(ctx.rdds)
        ctx.rdds.append(self)
        self.n, self.deps, self.op, self._compute, self.partitioner = n, deps, op, compute, partitioner
        self.name = name or f"{op}[{self.id}]"
        self.persisted = False
        self.stored: dict[int, list] = {}

    # ---- evaluation
    def iterator(self, i: int) -> list:
        ctx = self.ctx
        if i in self.stored:
            ctx.hits[(self.id, i)] += 1
            return self.stored[i]
        out = self._compute(i)
        ctx.computed[(self.id, i)] += 1
        ctx.job_computed[(self.id, i)] += 1
        ctx.last[(self.id, i)] = out
        if self.persisted:
            self.stored[i] = out
        return out

    def persist(self) -> "RDD":
        self.persisted = True
        return self

    cache = persist

    # ---- narrow transformations
    def _narrow(self, op, fn, keep=False, name=None) -> "RDD":
        return RDD(self.ctx, self.n, [Narrow(self, lambda i: [i])], op,
                   lambda i: fn(i, self.iterator(i)), self.partitioner if keep else None, name)

    def map(self, f, name=None):
        return self._narrow("map", lambda i, it: [f(x) for x in it], name=name)

    def filter(self, f, name=None):
        return self._narrow("filter", lambda i, it: [x for x in it if f(x)], keep=True, name=name)

    def flatMap(self, f, name=None):
        return self._narrow("flatMap", lambda i, it: [y for x in it for y in f(x)], name=name)

    def mapValues(self, f, name=None):
        return self._narrow("mapValues", lambda i, it: [(k, f(v)) for k, v in it], keep=True, name=name)

    def mapPartitions(self, f, name=None):
        return self._narrow("mapPartitions", lambda i, it: list(f(iter(it))), name=name)

    def union(self, other: "RDD", name=None) -> "RDD":
        a, b = self, other
        deps = [Narrow(a, lambda i: [i] if i < a.n else [], "Range"),
                Narrow(b, lambda i: [i - a.n] if i >= a.n else [], "Range")]
        return RDD(self.ctx, a.n + b.n, deps, "union",
                   lambda i: a.iterator(i) if i < a.n else b.iterator(i - a.n), None, name)

    def coalesce(self, m: int, name=None) -> "RDD":
        """Merge adjacent partitions without a shuffle (cannot increase the count)."""
        m = max(1, min(m, self.n))
        groups = slices(self.n, m)
        dep = Narrow(self, lambda i: list(range(*groups[i])), "Coalesced")
        return RDD(self.ctx, m, [dep], "coalesce",
                   lambda i: [x for p in range(*groups[i]) for x in self.iterator(p)], None, name)

    # ---- shuffles
    def _shuffled(self, part, op, combine, reduce_fn, name=None, round_robin=False) -> "RDD":
        dep = Shuffle(self.ctx, self, part, combine, round_robin)
        ctx = self.ctx

        def compute(i):
            ctx.ensure_map_outputs(dep)
            got = [r for m in range(self.n) for r in ctx.shuffle_store[dep.id][m][i]]
            ctx.fetch(dep, i)
            return reduce_fn(got)
        return RDD(ctx, part.n, [dep], op, compute, None if round_robin else part, name)

    def _default(self, n):
        return HashPartitioner(n or self.n)

    def combineByKey(self, create, merge_value, merge_comb, n=None, op="combineByKey", name=None, map_side=True):
        part = self._default(n)

        def reduce_side(recs):
            acc: dict = {}
            for k, c in recs:
                acc[k] = merge_comb(acc[k], c) if k in acc else c
            return list(acc.items())
        return self._shuffled(part, op, (create, merge_value) if map_side else None, reduce_side, name)

    def reduceByKey(self, f, n=None, name=None):
        return self.combineByKey(lambda v: v, f, f, n, "reduceByKey", name)

    def aggregateByKey(self, zero, seq, comb, n=None, name=None):
        return self.combineByKey(lambda v: seq(copy.deepcopy(zero), v), seq, comb, n, "aggregateByKey", name)

    def groupByKey(self, n=None, name=None):
        part = self._default(n)

        def reduce_side(recs):
            acc: dict = {}
            for k, v in recs:
                acc.setdefault(k, []).append(v)
            return list(acc.items())
        return self._shuffled(part, "groupByKey", None, reduce_side, name)

    def partitionBy(self, part: HashPartitioner, name=None):
        if self.partitioner == part:
            return self
        return self._shuffled(part, "partitionBy", None, lambda recs: list(recs), name)

    def repartition(self, m: int, name=None):
        """Always a full shuffle; records are dealt round-robin to the new partitions."""
        return self._shuffled(HashPartitioner(m), "repartition", None, lambda recs: list(recs), name, round_robin=True)

    def join(self, other: "RDD", n=None, name=None) -> "RDD":
        """Inner join as a cogroup; co-partitioned inputs are read through narrow dependencies."""
        ctx = self.ctx
        part = next((r.partitioner for r in (self, other) if r.partitioner is not None), None) or HashPartitioner(n or max(self.n, other.n))
        sides = []
        for r in (self, other):
            if r.partitioner == part:
                sides.append(Narrow(r, lambda i: [i], "OneToOne"))
            else:
                sides.append(Shuffle(ctx, r, part))

        def read(dep, i):
            if dep.kind == "narrow":
                return dep.parent.iterator(i)
            ctx.ensure_map_outputs(dep)
            ctx.fetch(dep, i)
            return [x for m in range(dep.parent.n) for x in ctx.shuffle_store[dep.id][m][i]]

        def compute(i):
            left, right = read(sides[0], i), read(sides[1], i)
            groups: dict = {}
            for k, v in left:
                groups.setdefault(k, ([], []))[0].append(v)
            for k, w in right:
                groups.setdefault(k, ([], []))[1].append(w)
            return [(k, (v, w)) for k, (vs, ws) in groups.items() for v in vs for w in ws]
        return RDD(ctx, part.n, sides, "join", compute, part, name)


# --------------------------------------------------------------------------- context
class Context:
    """SparkContext stand-in: owns the RDDs, the shuffle files and every counter."""

    def __init__(self, executors: int = 2):
        self.executors = executors
        self.rdds: list[RDD] = []
        self.shuffles: list[Shuffle] = []
        self.shuffle_store: dict[int, dict[int, dict[int, list]]] = {}   # shuffle -> map task -> bucket -> records
        self.computed: Counter = Counter()        # (rdd, partition) -> times computed, all jobs
        self.job_computed: Counter = Counter()    # same, current job only
        self.hits: Counter = Counter()            # cache reads
        self.last: dict = {}                      # (rdd, partition) -> last computed records
        self.shuffle_log: list[dict] = []         # one entry per map task written
        self.fetch_log: list[dict] = []
        self.jobs: list[dict] = []
        self.accumulators: dict[str, int] = {}

    def exec_of(self, i: int) -> int:
        return i % self.executors

    # ---- sources
    def parallelize(self, data: list, n: int, name=None, op="parallelize") -> RDD:
        data = list(data)
        cuts = slices(len(data), n)
        return RDD(self, n, [], op, lambda i: list(data[cuts[i][0]:cuts[i][1]]), None, name)

    def textFile(self, lines: list[str], n: int, name=None) -> RDD:
        return self.parallelize(lines, n, name, op="textFile")

    # ---- shuffle machinery
    def ensure_map_outputs(self, dep: Shuffle) -> None:
        store = self.shuffle_store.setdefault(dep.id, {})
        for m in range(dep.parent.n):
            if m in store:
                continue
            recs = dep.parent.iterator(m)
            written = recs
            if dep.combine:
                create, merge_value = dep.combine
                acc: dict = {}
                for k, v in recs:
                    acc[k] = merge_value(acc[k], v) if k in acc else create(v)
                written = list(acc.items())
            buckets: dict[int, list] = {r: [] for r in range(dep.partitioner.n)}
            for j, rec in enumerate(written):
                r = (m + j) % dep.partitioner.n if dep.round_robin else dep.partitioner.partition(rec[0])
                buckets[r].append(rec)
            store[m] = buckets
            self.shuffle_log.append({
                "shuffle": dep.id, "map": m, "job": len(self.jobs), "records_in": len(recs), "written": written,
                "records": len(written), "bytes": sum(size(x) for x in written),
                "to": [{"reduce": r, "records": len(b), "bytes": sum(size(x) for x in b),
                        "cross": self.exec_of(m) != self.exec_of(r)} for r, b in buckets.items()]})

    def fetch(self, dep, i: int) -> None:
        self.fetch_log.append({"shuffle": dep.id, "reduce": i, "job": len(self.jobs)})

    # ---- jobs
    def run_job(self, rdd: RDD, action: str = "collect", partitions: list[int] | None = None) -> dict:
        plan = self.stages(rdd)
        missing = self.missing_stages(rdd, plan)
        self.job_computed = Counter()
        n_log = len(self.shuffle_log)
        parts = list(range(rdd.n)) if partitions is None else partitions
        data = [rdd.iterator(i) for i in parts]
        flat = [x for p in data for x in p]
        result = {"collect": flat, "count": len(flat)}.get(action, flat)
        new_maps = self.shuffle_log[n_log:]
        tasks = {}
        for st in plan:
            if st["kind"] == "result":
                tasks[st["id"]] = len(parts) if st["id"] in missing else 0
            else:
                tasks[st["id"]] = sum(1 for e in new_maps if e["shuffle"] == st["shuffle"])
        job = {"id": len(self.jobs), "action": action, "rdd": rdd.id, "result": result,
               "computed": dict(self.job_computed), "partitions": parts,
               "stages_run": [s["id"] for s in plan if tasks[s["id"]] > 0 or (s["kind"] == "result" and s["id"] in missing)],
               "stages_skipped": [s["id"] for s in plan if not (tasks[s["id"]] > 0 or (s["kind"] == "result" and s["id"] in missing))],
               "tasks": tasks, "shuffle_written": sum(e["records"] for e in new_maps),
               "shuffle_bytes": sum(e["bytes"] for e in new_maps)}
        self.jobs.append(job)
        return job

    # ---- DAGScheduler
    def stages(self, final: RDD) -> list[dict]:
        """Cut the lineage of `final` into stages at shuffle dependencies (parents first)."""
        by_shuffle: dict[int, dict] = {}
        order: list[dict] = []

        def build(rdd: RDD, kind: str, dep: Shuffle | None) -> dict:
            st = {"kind": kind, "shuffle": dep.id if dep else None, "rdds": [], "parents": [], "out": rdd.id,
                  "tasks": rdd.n}
            seen: set[int] = set()

            def visit(r: RDD):              # depth-first in dependency order, so parents get lower ids
                if r.id in seen:
                    return
                seen.add(r.id)
                st["rdds"].append(r.id)
                for d in r.deps:
                    if d.kind == "narrow":
                        visit(d.parent)
                    else:
                        if d.id not in by_shuffle:
                            by_shuffle[d.id] = build(d.parent, "shuffle_map", d)
                        p = by_shuffle[d.id]
                        if p not in st["parents"]:
                            st["parents"].append(p)
            visit(rdd)
            st["rdds"].sort()
            order.append(st)
            return st

        build(final, "result", None)
        for i, st in enumerate(order):
            st["id"] = i + 1
        for st in order:
            st["parents"] = [p["id"] for p in st["parents"]]
        return order

    def missing_stages(self, final: RDD, plan: list[dict]) -> set[int]:
        """Stages that must run: walk back from the final RDD, stopping at fully cached RDDs and at
        shuffles whose map outputs all exist (Spark's getMissingParentStages)."""
        by_shuffle = {st["shuffle"]: st for st in plan if st["shuffle"] is not None}
        result = next(st for st in plan if st["kind"] == "result")
        need: set[int] = set()

        def walk(st):
            if st["id"] in need:
                return
            need.add(st["id"])
            seen, stack = set(), [self.rdds[st["out"]]]
            while stack:
                r = stack.pop()
                if r.id in seen:
                    continue
                seen.add(r.id)
                if r.persisted and len(r.stored) == r.n:
                    continue
                for d in r.deps:
                    if d.kind == "narrow":
                        stack.append(d.parent)
                    else:
                        have = self.shuffle_store.get(d.id, {})
                        if len(have) < d.parent.n:
                            walk(by_shuffle[d.id])
        walk(result)
        return need

    # ---- failures
    def lose_executor(self, e: int, shuffle_files: bool = True) -> dict:
        """Executor e dies: its cached partitions go, and its shuffle files (all shuffle files when
        `shuffle_files` is False, the worst case the slides draw)."""
        lost_cache, lost_maps = [], []
        for r in self.rdds:
            for i in list(r.stored):
                if self.exec_of(i) == e:
                    del r.stored[i]
                    lost_cache.append((r.id, i))
        for sid, store in self.shuffle_store.items():
            for m in list(store):
                if not shuffle_files or self.exec_of(m) == e:
                    del store[m]
                    lost_maps.append((sid, m))
        return {"executor": e, "cache": lost_cache, "map_outputs": lost_maps}


# =========================================================================== the two labs
# Everything below builds the course's pipelines on the engine above and reports every stage.
from collections import namedtuple  # noqa: E402

from .datasets import EMP8, REGIONS12, SPARK_CUSTOMERS, SPARK_ORDERS  # noqa: E402

INT_MAX, INT_MIN = 2 ** 31 - 1, -(2 ** 31)


class Stats(namedtuple("Stats", "sum count min max")):
    """rdds-2 p12's Stats(sum, count, min, max) accumulator."""
    __slots__ = ()

    def average(self) -> float:
        return self.sum / self.count


EMP_LINES = [f"{e},{n},{d},{s}" for e, n, d, s in EMP8]                 # eno,name,dno,salary
CUST_LINES = [f"{c},{st}" for c, st in SPARK_CUSTOMERS]                  # custId,state
ORDER_LINES = [f"{o},{c},{a}" for o, c, a in SPARK_ORDERS]               # orderNo,custId,amount
# A small tab-separated server log for the log-mining lineage (spark-intro p11, rdds-more p14):
# level, component, message, time (time is field #3, as on the slide).
LOG_LINES = [
    "INFO\tweb\tGET /index.html 200\t09:58:02", "ERROR\tdb\tMySQL connection refused\t09:58:40",
    "WARN\tweb\tslow response 2.3 s\t09:59:13", "ERROR\tapp\tPHP fatal: missing key 'qty'\t10:00:05",
    "INFO\tweb\tGET /cart 200\t10:00:31", "ERROR\tdb\tMySQL deadlock on orders\t10:01:22",
    "INFO\tauth\tlogin ok user=17\t10:01:50", "ERROR\tapp\tPHP warning: division by zero\t10:02:44",
    "ERROR\thdfs\tHDFS block blk_1073 missing\t10:03:09", "INFO\tweb\tGET /checkout 200\t10:03:57",
    "ERROR\tdb\tMySQL timeout after 30 s\t10:04:30", "WARN\tjvm\tGC pause 900 ms\t10:05:12",
]

RDD_PIPES = {
    "avg_reduce": {"label": "reduceByKey", "title": "Average salary with reduceByKey", "slides": "RDD 1 p27–31", "data": "EMP8"},
    "avg_group": {"label": "groupByKey", "title": "Average salary with groupByKey", "slides": "RDD 1 p32–37, RDD 2 p2–5", "data": "EMP8"},
    "avg_aggregate": {"label": "aggregateByKey", "title": "Average salary with aggregateByKey", "slides": "RDD 2 p7–12", "data": "EMP8"},
    "join": {"label": "join", "title": "Customers ⋈ orders: sales per state", "slides": "RDD 1 p40–41, RDD more p7", "data": "SPARK_CUSTOMERS ⋈ SPARK_ORDERS"},
    "regions": {"label": "regions", "title": "Customers per region (map, then reduceByKey)", "slides": "Spark impl p43–45", "data": "REGIONS12"},
    "mappartitions": {"label": "mapPartitions", "title": "mapPartitions with a per-partition counter", "slides": "Spark impl p46–47", "data": "[1..9]"},
    "broadcast": {"label": "broadcast", "title": "A broadcast variable", "slides": "RDD more p19", "data": "[1, 2, 3, 4]"},
    "accumulator": {"label": "accumulator", "title": "An accumulator updated inside map()", "slides": "RDD more p21–22", "data": "[1, 2, 3, 4]"},
}


def _mp_counter(it):
    """Spark impl p46: a per-partition counter; even positions ×100, odd ×10."""
    i, out = 0, []
    for v in it:
        i += 1
        out.append(v * 100 if i % 2 == 0 else v * 10)
    return out


def _expected(pipe: str, parts: int, stats: bool) -> list:
    """The right answer computed directly in plain Python (no engine), to check the run against."""
    if pipe in ("avg_reduce", "avg_group", "avg_aggregate"):
        acc: dict = {}
        for _, _, d, s in EMP8:
            acc.setdefault(str(d), []).append(s)
        return [(k, round(sum(v) / len(v), 2)) for k, v in acc.items()]
    if pipe == "join":
        state = {str(c): s for c, s in SPARK_CUSTOMERS}
        tot: dict = {}
        for _, c, a in SPARK_ORDERS:
            if str(c) in state:
                tot[state[str(c)]] = tot.get(state[str(c)], 0) + a
        return list(tot.items())
    if pipe == "regions":
        tot = {}
        for _, _, r in REGIONS12:
            tot[r] = tot.get(r, 0) + 1
        return list(tot.items())
    if pipe == "mappartitions":
        data = list(range(1, 10))
        return [y for a, b in slices(9, parts) for y in _mp_counter(data[a:b])]
    if pipe == "broadcast":
        return [x + [1, 2, 3][1] for x in [1, 2, 3, 4]]
    if pipe == "accumulator":
        return [x * x for x in [1, 2, 3, 4]]
    raise KeyError(pipe)


def _build_rdd_pipe(ctx: Context, pipe: str, P: int, stats: bool):
    """Returns (steps, final rdd, action calls). Each step is (variable, code, rdd)."""
    S = []
    add = lambda var, code, rdd, note=None: (S.append({"var": var, "code": code, "rdd": rdd, "note": note}), rdd)[1]
    if pipe in ("avg_reduce", "avg_group", "avg_aggregate"):
        lines = add("lines", f'lines = sc.textFile("employee.csv", {P})', ctx.textFile(EMP_LINES, P, "lines"),
                    "eno,name,dno,salary: one text line per employee")
        emps = add("emps", 'emps = lines.map(lambda line: line.split(","))', lines.map(lambda s: s.split(","), name="emps"))
        if pipe == "avg_reduce":
            pairs = add("dnosals1", "dnosals1 = emps.map(lambda rec: (rec[2], (int(rec[3]), 1)))",
                        emps.map(lambda r: (r[2], (int(r[3]), 1)), name="dnosals1"),
                        "key = dno (field 2), value = (salary, 1) so sums and counts add together")
            sums = add("sums1", "sums1 = dnosals1.reduceByKey(lambda s1, s2: (s1[0]+s2[0], s1[1]+s2[1]))",
                       pairs.reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]), name="sums1"),
                       "combines inside each partition first, then shuffles one partial per key per partition")
            final = add("avgs", "avgs = sums1.map(lambda s: (s[0], round(1.0*s[1][0]/s[1][1], 2)))",
                        sums.map(lambda s: (s[0], round(1.0 * s[1][0] / s[1][1], 2)), name="avgs"))
        elif pipe == "avg_group":
            pairs = add("dnosals1", "dnosals1 = emps.map(lambda rec: (rec[2], (int(rec[3]), 1)))",
                        emps.map(lambda r: (r[2], (int(r[3]), 1)), name="dnosals1"))
            groups = add("groups", "groups = dnosals1.groupByKey()", pairs.groupByKey(name="groups"),
                         "no map-side combine: every record crosses the shuffle")

            def map_fun(t):
                key, values = t
                s = c = 0
                for v in values:
                    s += v[0]
                    c += v[1]
                return key, s / c
            final = add("avgs", "avgs = groups.map(map_fun)   # sum and count the group, then divide",
                        groups.map(map_fun, name="avgs"))
        else:
            pairs = add("dnoSalPairs", "dnoSalPairs = emps.map(lambda rec: (rec[2], int(rec[3])))",
                        emps.map(lambda r: (r[2], int(r[3])), name="dnoSalPairs"))
            if stats:
                agg = add("aggregate", "aggregate = dnoSalPairs.aggregateByKey(Stats(0.0, 0, MAX, MIN),\n"
                          "    lambda s, x: Stats(s.sum+x, s.count+1, min(s.min, x), max(s.max, x)),   # seqFunc\n"
                          "    lambda a, b: Stats(a.sum+b.sum, a.count+b.count, min(a.min, b.min), max(a.max, b.max)))  # combFunc",
                          pairs.aggregateByKey(Stats(0.0, 0, INT_MAX, INT_MIN),
                                               lambda s, x: Stats(s.sum + x, s.count + 1, min(s.min, x), max(s.max, x)),
                                               lambda a, b: Stats(a.sum + b.sum, a.count + b.count, min(a.min, b.min), max(a.max, b.max)),
                                               name="aggregate"),
                          "seqFunc folds each salary into the partition's Stats; combFunc merges partitions' Stats")
                final = add("averages", "averages = aggregate.mapValues(lambda s: s.average())",
                            agg.mapValues(lambda s: s.average(), name="averages"))
            else:
                agg = add("aggregate", "aggregate = dnoSalPairs.aggregateByKey((0, 0),\n"
                          "    lambda a, x: (a[0] + x, a[1] + 1),          # seqFunc: (sum, count) + salary\n"
                          "    lambda a, b: (a[0] + b[0], a[1] + b[1]))    # combFunc: (sum, count) + (sum, count)",
                          pairs.aggregateByKey((0, 0), lambda a, x: (a[0] + x, a[1] + 1),
                                               lambda a, b: (a[0] + b[0], a[1] + b[1]), name="aggregate"),
                          "zero value (0, 0); seqFunc runs per partition, combFunc after the shuffle")
                final = add("averages", "averages = aggregate.mapValues(lambda x: x[0] / x[1])",
                            agg.mapValues(lambda x: x[0] / x[1], name="averages"))
        calls = [(final, "collect", f"{S[-1]['var']}.collect()")]
    elif pipe == "join":
        cu = add("customers", f'customers = sc.textFile("customers.csv", {P})', ctx.textFile(CUST_LINES, P, "customers"),
                 "custId,state")
        od = add("orders", f'orders = sc.textFile("orders.csv", {P})', ctx.textFile(ORDER_LINES, P, "orders"),
                 "orderNo,custId,amount")
        cr = add("custRecs", 'custRecs = customers.map(lambda x: (x.split(",")[0], x.split(",")[1]))',
                 cu.map(lambda x: (x.split(",")[0], x.split(",")[1]), name="custRecs"))
        orr = add("orderRecs", 'orderRecs = orders.map(lambda x: (x.split(",")[1], int(x.split(",")[2])))',
                  od.map(lambda x: (x.split(",")[1], int(x.split(",")[2])), name="orderRecs"))
        jn = add("joined", "joined = custRecs.join(orderRecs)", cr.join(orr, name="joined"),
                 "both sides are hash-partitioned on custId and shuffled; customers without orders drop out")
        so = add("stateOrders", "stateOrders = joined.map(lambda t: (t[1][0], t[1][1]))",
                 jn.map(lambda t: (t[1][0], t[1][1]), name="stateOrders"))
        final = add("summary", "summary = stateOrders.reduceByKey(lambda x, y: x + y)",
                    so.reduceByKey(lambda x, y: x + y, name="summary"))
        calls = [(final, "collect", "summary.collect()")]
    elif pipe == "regions":
        r1 = add("rdd1", f"rdd1 = sc.parallelize(customers12, {P})", ctx.parallelize(REGIONS12, P, "rdd1"),
                 "(id, name, region)")
        r2 = add("rdd2", "rdd2 = rdd1.map(lambda c: (c[0], c[2]))", r1.map(lambda c: (c[0], c[2]), name="rdd2"),
                 "the p43 exercise: same partitions, (id, region)")
        pr = add("pairs", "pairs = rdd2.map(lambda r: (r[1], 1))", r2.map(lambda r: (r[1], 1), name="pairs"))
        final = add("counts", "counts = pairs.reduceByKey(lambda a, b: a + b)",
                    pr.reduceByKey(lambda a, b: a + b, name="counts"))
        calls = [(final, "collect", "counts.collect()")]
    elif pipe == "mappartitions":
        r = add("rdd", f"rdd = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9], {P})",
                ctx.parallelize(list(range(1, 10)), P, "rdd"))
        final = add("new_rdd", "new_rdd = rdd.mapPartitions(f)   # i counts within the partition: even i ×100, odd i ×10",
                    r.mapPartitions(_mp_counter, name="new_rdd"), "f runs once per partition, so i restarts at 1 in each")
        calls = [(final, "collect", "new_rdd.collect()")]
    elif pipe == "broadcast":
        N = [1, 2, 3]
        r1 = add("rdd1", f"N = sc.broadcast([1, 2, 3])\nrdd1 = sc.parallelize([1, 2, 3, 4], {P})",
                 ctx.parallelize([1, 2, 3, 4], P, "rdd1"))
        final = add("rdd2", "rdd2 = rdd1.map(lambda x: x + N.value[1])", r1.map(lambda x: x + N[1], name="rdd2"),
                    "every task reads the executor's copy of N; N.value[1] = 2")
        calls = [(final, "collect", "rdd2.collect()")]
    elif pipe == "accumulator":
        ctx.accumulators["counter"] = 0

        def f1(x):
            ctx.accumulators["counter"] += 1
            return x * x
        r1 = add("rdd1", f"counter = sc.accumulator(0)\nrdd1 = sc.parallelize([1, 2, 3, 4], {P})",
                 ctx.parallelize([1, 2, 3, 4], P, "rdd1"))
        final = add("rdd2", "rdd2 = rdd1.map(lambda x: f1(x))   # f1: counter.add(1); return x*x",
                    r1.map(f1, name="rdd2"), "the counter is updated inside a transformation")
        calls = [(final, "collect", "rdd2.collect()"), (final, "count", "rdd2.count()")]
    else:
        raise KeyError(pipe)
    return S, final, calls


def _partitions_of(ctx: Context, rdd: RDD) -> list[dict] | None:
    if not any((rdd.id, i) in ctx.last or i in rdd.stored for i in range(rdd.n)):
        return None
    out = []
    for i in range(rdd.n):
        recs = rdd.stored.get(i, ctx.last.get((rdd.id, i)))
        out.append({"p": i, "executor": ctx.exec_of(i), "records": None if recs is None else [show(x) for x in recs],
                    "count": None if recs is None else len(recs), "bytes": None if recs is None else sum(size(x) for x in recs),
                    "computed": ctx.computed[(rdd.id, i)], "cache_hits": ctx.hits[(rdd.id, i)], "cached": i in rdd.stored})
    return out


def _shuffle_view(ctx: Context, dep: Shuffle, names: dict) -> dict:
    logs = sorted((e for e in ctx.shuffle_log if e["shuffle"] == dep.id), key=lambda e: e["map"])
    first = {}
    for e in logs:                       # the first write of each map task (rewrites only after failures)
        first.setdefault(e["map"], e)
    maps = [first[m] for m in sorted(first)]
    n_red = dep.partitioner.n
    matrix = [[0] * n_red for _ in maps]
    bmatrix = [[0] * n_red for _ in maps]
    for i, e in enumerate(maps):
        for t in e["to"]:
            matrix[i][t["reduce"]] = t["records"]
            bmatrix[i][t["reduce"]] = t["bytes"]
    return {
        "id": dep.id, "from": names.get(dep.parent.id, dep.parent.name), "partitioner": repr(dep.partitioner) if not dep.round_robin else f"round-robin({n_red})",
        "combine": dep.combine is not None,
        "map": [{"p": e["map"], "executor": ctx.exec_of(e["map"]), "records_in": e["records_in"],
                 "written": [show(x) for x in e["written"]], "records": e["records"], "bytes": e["bytes"]} for e in maps],
        "matrix": matrix, "bytes_matrix": bmatrix,
        "records": sum(e["records"] for e in maps), "bytes": sum(e["bytes"] for e in maps),
        "records_in": sum(e["records_in"] for e in maps),
        "cross_records": sum(t["records"] for e in maps for t in e["to"] if t["cross"]),
        "cross_bytes": sum(t["bytes"] for e in maps for t in e["to"] if t["cross"]),
    }


def _dep_view(d, names):
    return {"parent": names.get(d.parent.id, d.parent.name), "kind": d.kind, "name": d.name}


def rdd_lab(pipe: str = "avg_reduce", parts: int = 2, persist: bool = False, actions: int = 1, stats: bool = False) -> dict:
    """Run one of the course pipelines on the engine and report every step, partition by partition."""
    ctx = Context(executors=parts)
    steps, final, calls = _build_rdd_pipe(ctx, pipe, parts, stats)
    if persist:
        final.persist()
    names = {s["rdd"].id: s["var"] for s in steps}
    jobs = []
    for a in range(actions):
        rdd, act, label = calls[a % len(calls)]
        before = dict(ctx.computed)
        job = ctx.run_job(rdd, act)
        recomputed = [{"step": names.get(r, str(r)), "p": p} for (r, p), c in job["computed"].items() if before.get((r, p), 0) > 0]
        jobs.append({"i": a + 1, "call": label, "action": act,
                     "result": show(job["result"]) if act == "collect" else str(job["result"]),
                     "partitions_computed": sum(job["computed"].values()), "recomputed": recomputed,
                     "stages_run": len(job["stages_run"]), "stages_skipped": len(job["stages_skipped"]),
                     "tasks": sum(job["tasks"].values()), "shuffle_written": job["shuffle_written"],
                     "counter": ctx.accumulators.get("counter")})
    out_steps = []
    for k, s in enumerate(steps):
        r: RDD = s["rdd"]
        kind = "source" if not r.deps else ("shuffle" if any(d.kind == "shuffle" for d in r.deps) else "narrow")
        out_steps.append({
            "i": k + 1, "var": s["var"], "op": r.op, "code": s["code"], "note": s["note"], "kind": kind, "n": r.n,
            "partitioner": repr(r.partitioner) if r.partitioner else None, "persisted": r.persisted,
            "deps": [_dep_view(d, names) for d in r.deps],
            "partitions": _partitions_of(ctx, r),
            "shuffles": [_shuffle_view(ctx, d, names) for d in r.deps if d.kind == "shuffle"] if jobs else [],
        })
    expected = _expected(pipe, parts, stats)
    result = jobs[0]["result"] if jobs else None
    final_records = [x for i in range(final.n) for x in (final.stored.get(i) or ctx.last.get((final.id, i)) or [])] if jobs else []
    correct = bool(jobs) and sorted(map(show, final_records)) == sorted(map(show, expected))
    shuffles = [sh for st in out_steps for sh in st["shuffles"]]
    recomputations = sum(c - 1 for c in ctx.computed.values() if c > 1)
    plan = ctx.stages(final)
    bvar = None
    if pipe == "broadcast":
        bsize = size([1, 2, 3])
        tasks = parts * max(actions, 0)
        bvar = {"value": "[1, 2, 3]", "bytes": bsize, "executors": parts, "tasks": tasks,
                "broadcast_bytes": bsize * min(parts, ctx.executors) if actions else 0,
                "closure_bytes": bsize * tasks}
    return {
        "pipe": pipe, "label": RDD_PIPES[pipe]["title"], "slides": RDD_PIPES[pipe]["slides"], "data": RDD_PIPES[pipe]["data"],
        "parts": parts, "persist": persist, "actions": actions, "stats": stats, "lazy": actions == 0,
        "steps": out_steps, "jobs": jobs, "calls": [c[2] for c in calls],
        "result": result, "expected": [show(x) for x in expected], "correct": correct,
        "stages": [{"id": st["id"], "kind": st["kind"], "rdds": [names.get(r, str(r)) for r in st["rdds"]],
                    "tasks": st["tasks"], "parents": st["parents"]} for st in plan],
        "shuffled_records": sum(sh["records"] for sh in shuffles),
        "shuffled_bytes": sum(sh["bytes"] for sh in shuffles),
        "shuffle_input": sum(sh["records_in"] for sh in shuffles),
        "cross_records": sum(sh["cross_records"] for sh in shuffles),
        "cross_bytes": sum(sh["cross_bytes"] for sh in shuffles),
        "partitions_computed": sum(ctx.computed.values()), "recomputations": recomputations,
        "counter": ctx.accumulators.get("counter"), "counter_expected": 4 if pipe == "accumulator" else None,
        "broadcast": bvar,
    }


# --------------------------------------------------------------------------- DAG lab
DAG_PIPES = {
    "quiz": {"label": "Quiz", "title": "The stage quiz: textFile/hadoopFile, groupByKey, join", "slides": "Spark impl p21–22"},
    "paper": {"label": "p20", "title": "The paper's figure: groupBy, map, union, join", "slides": "Spark impl p20"},
    "logmine": {"label": "Logs", "title": "Log mining: one cached RDD, three actions", "slides": "Spark intro p11, RDD more p14"},
    "join": {"label": "Join", "title": "Customers ⋈ orders, shuffled or co-partitioned", "slides": "Spark impl p11–15, p35–39"},
    "resize": {"label": "Resize", "title": "repartition against coalesce", "slides": "Spark impl p49–50"},
    "hash": {"label": "Hash", "title": "HashPartitioner(n) placement", "slides": "Spark impl p51"},
}
DAG_EXECUTORS = 3


def _build_dag(ctx: Context, pipe: str, P: int, cache: bool, copart: bool, target: int, how: str):
    """Returns (rdd names in order, list of (rdd, action, label) calls, the cache point or None)."""
    N: dict[int, str] = {}

    def nm(var, rdd):
        N[rdd.id] = var
        rdd.name = var
        return rdd
    cache_at = None
    if pipe == "quiz":
        a = nm("textFile", ctx.textFile(ORDER_LINES, P))
        b = nm("map (pairs)", a.map(lambda x: (x.split(",")[1], int(x.split(",")[2]))))
        c = nm("filter", b.filter(lambda kv: kv[1] >= 400))
        d = nm("hadoopFile", ctx.parallelize([(str(k), s) for k, s in SPARK_CUSTOMERS], P, op="hadoopFile"))
        e = nm("groupByKey", d.groupByKey(P))
        f = nm("mapValues (first state)" if copart else "map (first state)", e.mapValues(lambda v: v[0]) if copart else e.map(lambda kv: (kv[0], kv[1][0])))
        g = nm("join", c.join(f))
        h = nm("map (state, amount)", g.map(lambda t: (t[1][1], t[1][0])))
        cache_at = f
        calls = [(h, "collect", "result.collect()")]
    elif pipe == "paper":
        A = nm("A", ctx.parallelize([(r, n) for _, n, r in REGIONS12], P))
        B = nm("B", A.groupByKey(P))
        C = nm("C", ctx.parallelize([(r, i) for i, _, r in REGIONS12[:6]], 2))
        D = nm("D", C.map(lambda kv: (kv[0], kv[1] * 10)))
        E = nm("E", ctx.parallelize([(r, i * 10) for i, _, r in REGIONS12[6:]], 2))
        F = nm("F", D.union(E))
        G = nm("G", B.join(F))
        cache_at = B
        calls = [(G, "collect", "G.collect()")]
    elif pipe == "logmine":
        lines = nm("lines", ctx.textFile(LOG_LINES, P))
        errors = nm("errors", lines.filter(lambda s: s.startswith("ERROR")))
        mysql = nm("mysql", errors.filter(lambda s: "MySQL" in s))
        php = nm("php", errors.filter(lambda s: "PHP" in s))
        times = nm("times", php.map(lambda s: s.split("\t")[3]))
        cache_at = errors
        calls = [(errors, "count", "errors.count()"), (mysql, "count", 'errors.filter(_.contains("MySQL")).count()'),
                 (times, "collect", 'errors.filter(_.contains("PHP")).map(_.split("\\t")(3)).collect()')]
    elif pipe == "join":
        cu = nm("customers", ctx.textFile(CUST_LINES, P))
        od = nm("orders", ctx.textFile(ORDER_LINES, P))
        cr = nm("custRecs", cu.map(lambda x: (x.split(",")[0], x.split(",")[1])))
        orr = nm("orderRecs", od.map(lambda x: (x.split(",")[1], int(x.split(",")[2]))))
        if copart:
            part = HashPartitioner(P)
            cr = nm("custParted", cr.partitionBy(part))
            orr = nm("orderParted", orr.partitionBy(part))
            cache_at = (cr, orr)
        else:
            cache_at = None
        jn = nm("joined", cr.join(orr))
        out = nm("stateOrders", jn.map(lambda t: (t[1][0], t[1][1])))
        calls = [(out, "collect", "stateOrders.collect()")]
    elif pipe == "resize":
        src = nm("rdd1", ctx.parallelize([(i, r) for i, _, r in REGIONS12], P))
        out = nm(how, src.repartition(target) if how == "repartition" else src.coalesce(target))
        cache_at = src
        calls = [(out, "collect", f"rdd1.{how}({target}).collect()")]
    elif pipe == "hash":
        src = nm("rdd1", ctx.parallelize([(r, i) for i, _, r in REGIONS12], P))
        out = nm("partitioned", src.partitionBy(HashPartitioner(target)))
        cache_at = src
        calls = [(out, "collect", f"rdd1.partitionBy(HashPartitioner({target})).collect()")]
    else:
        raise KeyError(pipe)
    if cache and cache_at is not None:
        for r in (cache_at if isinstance(cache_at, tuple) else (cache_at,)):
            r.persist()
    return N, calls, cache_at


def _depths(ctx: Context) -> dict[int, int]:
    d: dict[int, int] = {}
    for r in ctx.rdds:                     # rdds are created after their parents
        d[r.id] = 1 + max((d[x.parent.id] for x in r.deps), default=-1)
    return d


def iterative_cost(iterations: int = 10, data_gb: float = 100.0) -> dict:
    """Spark intro p13: logistic regression, 100 GB on 50 nodes. MapReduce re-reads the input from
    disk every iteration (110 s); Spark reads it once and then takes about 1 s per cached iteration.
    Both scale linearly with the data; data beyond the cluster's cache capacity is re-read from disk."""
    ref_gb, mr_s, sp_s = 100.0, 110.0, 1.0
    cap = ITER_CACHE_GB
    scale = data_gb / ref_gb
    fit = min(1.0, cap / data_gb)
    per_mr = mr_s * scale
    per_sp = (fit * sp_s + (1 - fit) * mr_s) * scale
    xs = list(range(1, max(30, iterations) + 1))
    mr = [n * per_mr for n in xs]
    sp = [per_mr + (n - 1) * per_sp for n in xs]
    i = iterations - 1
    at10 = 9
    return {"iterations": xs, "mapreduce": mr, "spark": sp, "n": iterations, "data_gb": data_gb,
            "per_iter_mr": per_mr, "per_iter_spark": per_sp, "first": per_mr, "cached_share": fit,
            "speedup": mr[i] / sp[i], "speedup_10": mr[at10] / sp[at10], "per_iter_ratio": per_mr / per_sp,
            "limit": per_mr / per_sp, "cache_gb": cap, "mr_total": mr[i], "spark_total": sp[i]}


ITER_CACHE_GB = 50 * 68.4 * 0.6            # 50 m2.4xlarge nodes, 68.4 GB each, 60 % for Spark's memory


def dag_lab(pipe: str = "quiz", parts: int = 3, cache: bool = False, actions: int = 1, reuse: bool = True,
            copart: bool = False, target: int = 2, how: str = "repartition", lose: int = -1) -> dict:
    ctx = Context(executors=DAG_EXECUTORS)
    names, calls, cache_at = _build_dag(ctx, pipe, parts, cache, copart, target, how)
    final = calls[-1][0]
    jobs = []
    for a in range(actions):
        rdd, act, label = calls[a % len(calls)]
        if not reuse and a:
            ctx.shuffle_store.clear()
        plan = ctx.stages(rdd)
        job = ctx.run_job(rdd, act)
        jobs.append({"i": a + 1, "call": label, "rdd": names[rdd.id],
                     "result": show(job["result"]) if act == "collect" else str(job["result"]),
                     "stages": [{"id": s["id"], "kind": s["kind"], "rdds": [names[r] for r in s["rdds"]],
                                 "tasks": s["tasks"], "tasks_run": job["tasks"][s["id"]],
                                 "skipped": s["id"] in job["stages_skipped"], "parents": s["parents"]} for s in plan],
                     "partitions_computed": sum(job["computed"].values()),
                     "computed": [{"rdd": names[r], "p": p} for (r, p) in job["computed"]],
                     "shuffle_written": job["shuffle_written"], "shuffle_bytes": job["shuffle_bytes"]})
    recovery = None
    if lose >= 0:
        lost = ctx.lose_executor(lose, shuffle_files=True)
        targets = [i for i in range(final.n) if ctx.exec_of(i) == lose]
        plan = ctx.stages(final)
        job = ctx.run_job(final, "collect", targets)
        recovery = {"executor": lose, "targets": targets,
                    "lost_cache": [{"rdd": names[r], "p": p} for r, p in lost["cache"]],
                    "lost_map_outputs": [{"shuffle": s, "map": m} for s, m in lost["map_outputs"]],
                    "recomputed": [{"rdd": names[r], "p": p} for (r, p) in job["computed"]],
                    "tasks": sum(job["tasks"].values()),
                    "stages": [{"id": s["id"], "tasks_run": job["tasks"][s["id"]], "skipped": s["id"] in job["stages_skipped"]} for s in plan],
                    "partitions_total": sum(r.n for r in ctx.rdds)}
    plan = ctx.stages(final)
    stage_of = {r: s["id"] for s in plan for r in s["rdds"]}
    depth = _depths(ctx)
    lanes: dict[int, int] = {}            # a derived RDD stays in its first parent's lane when that cell is free
    taken: set = set()
    for r in ctx.rdds:
        want = lanes[r.deps[0].parent.id] if r.deps else (max(lanes.values()) + 1 if lanes else 0)
        if (depth[r.id], want) in taken:
            want = max(lanes.values()) + 1
        lanes[r.id] = want
        taken.add((depth[r.id], want))
    rec_set = {(x["rdd"], x["p"]) for x in (recovery["recomputed"] if recovery else [])}
    rdds = []
    for r in ctx.rdds:
        parts_out = _partitions_of(ctx, r) or []
        for p in parts_out:
            p["recomputed"] = (names[r.id], p["p"]) in rec_set
        rdds.append({"id": r.id, "name": names[r.id], "op": r.op, "n": r.n, "depth": depth[r.id], "lane": lanes[r.id],
                     "stage": stage_of.get(r.id), "persisted": r.persisted,
                     "partitioner": repr(r.partitioner) if r.partitioner else None,
                     "deps": [{"parent": d.parent.id, "kind": d.kind, "name": d.name,
                               "map": [d.parents(i) for i in range(r.n)] if d.kind == "narrow" else None} for d in r.deps],
                     "partitions": parts_out})
    names_ = {r.id: names[r.id] for r in ctx.rdds}
    shuffles = [_shuffle_view(ctx, d, names_) for d in ctx.shuffles if d.id in ctx.shuffle_store or any(e["shuffle"] == d.id for e in ctx.shuffle_log)]
    out = {
        "pipe": pipe, "label": DAG_PIPES[pipe]["title"], "slides": DAG_PIPES[pipe]["slides"], "executors": DAG_EXECUTORS,
        "parts": parts, "cache": cache, "actions": actions, "reuse": reuse, "copart": copart, "target": target, "how": how,
        "lose": lose, "cache_point": None if cache_at is None else
        ", ".join(names[r.id] for r in (cache_at if isinstance(cache_at, tuple) else (cache_at,))),
        "rdds": rdds, "jobs": jobs, "recovery": recovery, "shuffles": shuffles,
        "stages": [{"id": s["id"], "kind": s["kind"], "rdds": [names[r] for r in s["rdds"]], "out": names[s["out"]],
                    "tasks": s["tasks"], "parents": s["parents"],
                    "narrow_ops": [names[r] for r in s["rdds"]]} for s in plan],
        "n_stages": len(plan), "tasks_total": sum(s["tasks"] for s in plan),
        "narrow_deps": sum(1 for r in ctx.rdds for d in r.deps if d.kind == "narrow"),
        "wide_deps": sum(1 for r in ctx.rdds for d in r.deps if d.kind == "shuffle"),
        "stages_run": sum(sum(1 for s in j["stages"] if not s["skipped"]) for j in jobs),
        "stages_skipped": sum(sum(1 for s in j["stages"] if s["skipped"]) for j in jobs),
        "tasks_run": sum(sum(s["tasks_run"] for s in j["stages"]) for j in jobs),
        "partitions_computed": sum(ctx.computed.values()),
        "recomputations": sum(c - 1 for c in ctx.computed.values() if c > 1),
        "result": jobs[-1]["result"] if jobs else None,
    }
    if pipe == "hash":
        keys = list(dict.fromkeys(r for _, _, r in REGIONS12))
        out["placement"] = [{"key": k, "hash": java_hash(k), "partition": java_hash(k) % target,
                             "records": sum(1 for _, _, r in REGIONS12 if r == k)} for k in keys]
    if pipe == "resize":
        src = ctx.rdds[0]
        dst = final
        moves = []
        if how == "coalesce":
            groups = slices(src.n, dst.n)
            for j, (a, b) in enumerate(groups):
                for p in range(a, b):
                    n = len(ctx.last.get((src.id, p), []))
                    moves.append({"from": p, "to": j, "records": n, "cross": ctx.exec_of(p) != ctx.exec_of(j),
                                  "bytes": sum(size(x) for x in ctx.last.get((src.id, p), []))})
        else:
            sh = shuffles[0] if shuffles else None
            if sh:
                for i, row in enumerate(sh["matrix"]):
                    for j, n in enumerate(row):
                        if n:
                            moves.append({"from": sh["map"][i]["p"], "to": j, "records": n, "bytes": sh["bytes_matrix"][i][j],
                                          "cross": ctx.exec_of(sh["map"][i]["p"]) != ctx.exec_of(j)})
        out["moves"] = moves
        out["moved_records"] = sum(m["records"] for m in moves if m["cross"])
        out["shuffle_records"] = sum(m["records"] for m in moves) if how == "repartition" else 0
        out["out_partitions"] = dst.n
        out["sizes"] = [len(ctx.last.get((dst.id, i), [])) for i in range(dst.n)]
    return out
