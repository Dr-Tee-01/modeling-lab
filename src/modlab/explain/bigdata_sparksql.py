"""Explainers for the Big Data Lab's Spark SQL views: join strategies, the optimiser, tables on disk.

Every figure is computed by modlab.bigdata.spark_sql, the same simulator the live views call.
"""
from __future__ import annotations

import functools
import math

import numpy as np

from ..bigdata import spark_sql as S
from .registry import explainer, figure

MB = S.MB
_STRAT_COLOR = {"bhj": "s1", "shj": "s2", "smj": "s3", "copart": "s4", "bnlj": "warn", "cartesian": "bad"}


def _b(v) -> str:
    return S.fmt_bytes(v)


# ===================================================================== joins: figures
@functools.lru_cache(maxsize=64)
def _course(join="inner", threshold=10.0, cond="eq", layout="plain", hint="none"):
    return S.run_join(source="course", join=join, threshold_mb=threshold, cond=cond, layout=layout, hint=hint)


_JOIN_OPTS = [["inner", "Inner"], ["left", "Left outer"], ["full", "Full outer"]]


@figure("sj-course-plan")
def _fig_course_plan(p):
    join = p.choice("join", "inner", ["inner", "left", "full"])
    thr = p.num("threshold", 10, -1, 10, integer=True)
    r = _course(join, float(thr))
    st, b, m = r["plan"]["strategy"], r["plan"]["build"], r["metrics"]
    L, R = r["tables"]["left"], r["tables"]["right"]
    moved_l = m["broadcast_bytes"] if b == "left" else (L["bytes"] if m["shuffle_bytes"] else 0)
    moved_r = m["broadcast_bytes"] if b == "right" else (R["bytes"] if m["shuffle_bytes"] else 0)
    w = lambda v: 1.5 + 6 * min(1.0, v / 1200) if v else 1
    return {
        "kind": "flow", "height": 230,
        "nodes": [
            {"id": "c", "x": 10, "y": 20, "w": 140, "h": 70, "title": "customers", "value": f"{L['rows']} rows", "sub": _b(L["bytes"])},
            {"id": "o", "x": 10, "y": 135, "w": 140, "h": 70, "title": "orders", "value": f"{R['rows']} rows", "sub": _b(R["bytes"])},
            {"id": "s", "x": 235, "y": 72, "w": 160, "h": 82, "title": S.STRATEGY[st]["short"], "sub": f"build {b}" if b else ("shuffle both" if m["shuffle_bytes"] else "no exchange"), "hot": True},
            {"id": "r", "x": 470, "y": 78, "w": 120, "h": 70, "title": "result", "value": f"{r['result']['total']} rows",
             "sub": f"{r['result']['unmatched_left']} null-extended" if r["result"]["unmatched_left"] else "all matched"},
        ],
        "edges": [
            {"from": "c", "to": "s", "label": _b(moved_l) if moved_l else "stays", "width": w(moved_l), "flow": bool(moved_l), "color": "s1"},
            {"from": "o", "to": "s", "label": _b(moved_r) if moved_r else "stays", "width": w(moved_r), "flow": bool(moved_r), "color": "s2"},
            {"from": "s", "to": "r", "width": 2.5},
        ],
        "texts": [{"x": 300, "y": 22, "text": f"{S.JOIN_SQL[join]} · threshold {'off (−1)' if thr < 0 else f'{thr} MB'}", "cls": "n-title"}],
        "notes": [f"Spark picks {S.STRATEGY[st]['label'].lower()}: {r['plan']['why']}.",
                  f"{_b(m['moved_bytes'])} crosses the network and the join returns {r['result']['total']} rows"
                  + (f", {r['result']['unmatched_left']} of them customers with no order (cust_id {' and '.join(map(str, r['result']['unmatched_left_ids']))})." if r["result"]["unmatched_left"] else ".")],
    }


_SCALE = dict(R_rows=lambda mb: int(mb * MB / 100), L=int(1e8), lw=100, rw=50, partitions=200)


def _scale_costs(mb: float, executors: int) -> dict:
    R, L = _SCALE["R_rows"](mb), _SCALE["L"]
    sizes = {"left_bytes": R * 100, "right_bytes": L * 50}
    lp, rp = S._partition_loads("synthetic", R, L, 0.0, 200)
    out = {}
    for s, b in (("bhj", "left"), ("smj", None), ("shj", "left")):
        out[s] = S.strategy_cost(s, b, sizes=sizes, R=R, L=L, lw=100, rw=50, lp=lp, rp=rp, partitions=200, executors=executors, aqe=False)
    return out


@figure("sj-threshold")
def _fig_threshold(p):
    E = p.num("executors", 4, 2, 32, integer=True)
    xs = np.logspace(0, 4, 33)                    # 1 MB .. 10 GB of customers
    costs = [_scale_costs(float(x), E) for x in xs]
    bhj = [c["bhj"]["est_seconds"] for c in costs]
    smj = [c["smj"]["est_seconds"] for c in costs]
    cross = next((float(x) for x, a, b in zip(xs, bhj, smj) if a > b), None)
    return {
        "kind": "lines", "x": xs.tolist(), "xlog": True, "ylog": True, "xlabel": "Size of customers (MB, log scale)", "ylabel": "Modelled seconds",
        "series": [{"name": "Broadcast hash (customers broadcast)", "color": "s1", "y": bhj},
                   {"name": "Sort-merge (both shuffled)", "color": "s3", "y": smj}],
        "markers": [{"x": 10, "label": "10 MB default"}, {"x": 8 * 1024, "label": "8 GB limit"}],
        "notes": [f"Orders stays at {_b(_SCALE['L'] * 50)} (100 million rows). Broadcasting costs size × {E} executors on the network; sort-merge always shuffles both tables.",
                  (f"With {E} executors the modelled crossover is near {cross:,.0f} MB, far above the 10 MB default: the threshold protects executor memory (every executor holds the whole table) more than it saves network time."
                   if cross else f"With {E} executors broadcasting stays cheaper over the whole range; the threshold exists to protect executor memory, since each executor holds the whole broadcast table.")],
    }


@functools.lru_cache(maxsize=16)
def _scale_run(executors: int):
    return S.run_join(source="synthetic", lrows_exp=5, rrows_exp=8, lbytes=100, rbytes=50, threshold_mb=10, partitions=200, executors=executors)


@figure("sj-moved")
def _fig_moved(p):
    E = p.num("executors", 4, 1, 32, integer=True)
    r = _scale_run(E)
    rows = [c for c in r["compare"] if c["feasible"]]
    labels = [c["label"] for c in rows]
    bc = [c["broadcast_bytes"] / MB for c in rows]
    sh = [(c["shuffle_bytes"] + c.get("extra_read_bytes", 0)) / MB for c in rows]
    big = max(rows, key=lambda c: c["moved_bytes"])
    return {
        "kind": "bars", "labels": labels, "ylabel": "MB moved", "ylog": False,
        "series": [{"name": "broadcast", "color": "s1", "values": bc}, {"name": "shuffled", "color": "s2", "values": sh}],
        "notes": [f"customers is {_b(r['tables']['left']['bytes'])} and orders {_b(r['tables']['right']['bytes'])}. "
                  f"Broadcasting customers to {E} executor{'s' if E > 1 else ''} moves {_b(r['compare'][0]['broadcast_bytes'])}; a shuffle join moves {_b(r['tables']['left']['bytes'] + r['tables']['right']['bytes'])}.",
                  f"The worst is the {big['label'].lower()} join at {_b(big['moved_bytes'])}. Each extra executor adds one more copy of the broadcast table and nothing to a shuffle."],
    }


@figure("sj-skew")
def _fig_skew(p):
    s = p.num("skew", 1.2, 0, 2)
    P, R, L = 16, int(1e6), int(1e9)
    lp, rp = S._partition_loads("synthetic", R, L, s, P)
    tot = lp + rp
    w, _ = S._zipf_weights(R, s, S._TOP_KEYS)
    med = float(np.median(tot))
    k = int(np.argmax(tot))
    ratio = tot[k] / med
    second = (f"Partition {k} holds {tot[k] / 1e6:,.0f} million rows, {ratio:.1f}× the median: the stage cannot finish before that one task does."
              if ratio >= 1.5 else f"Every partition is within {100 * (ratio - 1):.0f}% of the median ({med / 1e6:,.0f} million rows): no straggler.")
    return {
        "kind": "bars", "labels": [str(i) for i in range(P)], "xlabel": "Shuffle partition", "ylabel": "Million rows",
        "series": [{"name": "customers", "color": "s1", "values": (lp / 1e6).tolist()}, {"name": "orders", "color": "s2", "values": (rp / 1e6).tolist()}],
        "hlines": [{"y": med / 1e6, "label": "median", "color": "neutral"}],
        "notes": [f"With Zipf s = {s:.1f} the most frequent customer owns {100 * w[0]:.3g}% of the 1 billion orders"
                  + (f", and every one of them hashes to partition {int(S.partition_of([1], P)[0])}." if s > 0 else ": with s = 0 every customer has the same share."),
                  second],
    }


_GRID_MB = np.logspace(-1, 4, 14)
_CODE = {("bhj", "left"): 1, ("bhj", "right"): 2, ("smj", None): 3, ("shj", "left"): 4, ("shj", "right"): 4}


@figure("sj-planner-map")
def _fig_planner(p):
    join = p.choice("join", "inner", ["inner", "left", "full"])
    cells, counts = [], {1: 0, 2: 0, 3: 0, 4: 0}
    for lmb in _GRID_MB[::-1]:
        for rmb in _GRID_MB:
            pl = S.plan_join(join=join, cond="eq", hint="none", threshold_mb=10, partitions=200, prefer_smj=True, bucketed=False,
                             sizes={"left_bytes": lmb * MB, "right_bytes": rmb * MB})
            c = _CODE[(pl["strategy"], pl["build"])]
            counts[c] += 1
            cells.append(c)
    n = len(cells)
    tick = lambda v: f"{v:g}" if v < 1 else f"{v:,.0f}"
    return {
        "kind": "grid", "cols": len(_GRID_MB), "rows": len(_GRID_MB), "cells": cells, "palette": "cat",
        "colors": ["neutral", "s1", "s4", "s3", "s2"], "height": 330, "xlabel": "orders size (MB)", "ylabel": "customers (MB)",
        "xticks": [[j, tick(v)] for j, v in enumerate(_GRID_MB) if j % 3 == 0],
        "yticks": [[i, tick(v)] for i, v in enumerate(_GRID_MB[::-1]) if i % 3 == 0],
        "legend": [{"name": "broadcast customers", "color": "s1"}, {"name": "broadcast orders", "color": "s4"}, {"name": "sort-merge", "color": "s3"}],
        "notes": [f"For a {S.JOIN_SQL[join]} with the 10 MB default: {counts[1]} of {n} size pairs broadcast customers, {counts[2]} broadcast orders and {counts[3]} fall back to sort-merge.",
                  {"inner": "An inner join may broadcast whichever side is under 10 MB, the smaller one if both are.",
                   "left": "A LEFT OUTER join may only broadcast the right side (orders): a small customers table does not help.",
                   "full": "A FULL OUTER join can never use broadcast hash, so every size pair is a sort-merge join."}[join]],
    }


@figure("sj-nested")
def _fig_nested(p):
    n = np.logspace(1, 6, 26)
    r = _course("inner", 10.0, cond="band")
    return {
        "kind": "lines", "x": n.tolist(), "xlog": True, "ylog": True, "xlabel": "Rows on each side", "ylabel": "Comparisons or probes",
        "series": [{"name": "nested loop |L| × |R|", "color": "bad", "y": (n * n).tolist()},
                   {"name": "sort-merge 2·n·log₂n + merge", "color": "s3", "y": (2 * n * np.log2(n) + 2 * n).tolist()},
                   {"name": "hash join: build + probe", "color": "s1", "y": (2 * n).tolist()}],
        "notes": [f"On the course tables the BETWEEN join compares {len(S.SPARK_CUSTOMERS)} × {len(S.SPARK_ORDERS)} = {int(r['metrics']['nl_comparisons'])} pairs to return {r['result']['total']} rows.",
                  "At a million rows a side the nested loop needs 10¹² comparisons against 2 × 10⁶ hash probes: it is the fallback when there is no equality to hash on."],
    }


@figure("sj-agree")
def _fig_agree(p):
    join = p.choice("join", "inner", ["inner", "left", "full"])
    r = _course(join, -1.0)
    items = [{"label": v["label"], "value": f"{v['rows']} rows" if v["feasible"] else "n/a",
              "note": ("same rows as the all-pairs join" if v["same_as_reference"] else "differs") if v["feasible"] else v["reason"],
              "tone": ("good" if v["same_as_reference"] else "bad") if v["feasible"] else None} for v in r["validation"]]
    ok = [v for v in r["validation"] if v["feasible"]]
    return {"kind": "stats", "items": items,
            "notes": [f"{len(ok)} strategies can run a {S.JOIN_SQL[join]} with an equality; all {len(ok)} return the same {ok[0]['rows']} rows.",
                      f"{len(r['validation']) - len(ok)} cannot: " + "; ".join(f"{v['label']} ({v['reason']})" for v in r["validation"] if not v["feasible"]) + "."
                      if len(ok) < len(r["validation"]) else "Every strategy is possible here; the planner's job is only to pick the cheapest."]}


_J0 = _course("inner", 10.0)
_J1 = _course("inner", -1.0)
_JL = _course("left", -1.0)
_JB = _course("inner", 10.0, cond="band")

explainer("bigdata/spark-joins", {
    "title": "How Spark joins two tables",
    "kicker": "Big Data Lab · Spark join strategies",
    "summary": f"The course's {len(S.SPARK_CUSTOMERS)} customers and {len(S.SPARK_ORDERS)} orders can be joined six ways in Spark. "
               "Which one runs depends on table sizes, the join type and the join condition. This page follows Spark's planner through "
               "each check, runs every strategy partition by partition, and shows what each one moves and where it breaks.",
    "sections": [
        {"id": "task", "title": "The lab task", "blocks": [
            {"type": "p", "text": "Join <code>customers</code> (cust_id, state) with <code>orders</code> (order_id, cust_id, amount) on cust_id, "
                                  "the tables from the RDD deck (p40–41). In SQL: <code>SELECT * FROM customers c JOIN orders o ON c.cust_id = o.cust_id</code>. "
                                  "Spark turns that one line into one of several physical plans."},
            {"type": "cards", "items": [
                {"title": "How big is each side?", "text": f"customers is {_b(_J0['tables']['left']['bytes'])} and orders {_b(_J0['tables']['right']['bytes'])} by Spark's estimate. Anything under 10 MB can be copied to every executor."},
                {"title": "Which join type?", "text": "Inner, left outer or full outer: outer joins must remember unmatched rows, which rules out some strategies."},
                {"title": "Which condition?", "text": "An equality (=) lets Spark hash or sort on the key; a range (BETWEEN) leaves only nested loops."},
            ]},
            {"type": "figure", "figure": "sj-course-plan", "title": "What Spark does with the course tables",
             "caption": "Change the join type and the broadcast threshold; the planner's choice and the bytes moved are recomputed.",
             "controls": [{"k": "join", "label": "Join type", "options": _JOIN_OPTS, "value": "inner"},
                          {"k": "threshold", "label": "autoBroadcastJoinThreshold (MB)", "min": -1, "max": 10, "step": 1, "value": 10}]},
        ]},
        {"id": "hash", "title": "Hash joins: build and probe", "blocks": [
            {"type": "p", "text": "The deck's two hash-join variations. <b>HJ1</b>: scan the smaller table R, build a hash table H<sub>R</sub> on the join key "
                                  "(the hashing phase), then scan S and look each row's key up in H<sub>R</sub> (the probing phase). <b>HJ2</b>: hash both "
                                  "tables into the same buckets, then join bucket i of R with bucket i of S."},
            {"type": "steps", "title": "HJ1 on the course tables", "items": [
                f"Build: put the {len(S.SPARK_CUSTOMERS)} customers into a hash table keyed by cust_id (7 entries).",
                f"Probe: for each of the {len(S.SPARK_ORDERS)} orders, hash its cust_id and look it up: one probe per order.",
                f"Emit a joined row for every match: {_J0['result']['total']} rows, since every order has a customer.",
                f"Customers {' and '.join(map(str, _JL['result']['unmatched_left_ids']))} have no orders, so no probe ever finds them; an inner join drops them, a left outer join null-extends them.",
            ]},
            {"type": "eq", "text": "work(hash join) ≈ |R| + |S|\nwork(nested loop) = |R| × |S|",
             "where": [["|R|", "rows of the build side, hashed once"], ["|S|", "rows of the probe side, one lookup each"],
                       ["|R| × |S|", "every pair compared, as a nested loop must"]]},
            {"type": "callout", "tone": "info", "title": "Equality only",
             "text": "A hash table only answers 'which rows have exactly this key?'. A condition such as o.cust_id BETWEEN c.cust_id − 1 AND c.cust_id + 1 "
                     "cannot be looked up, which is why range joins fall back to nested loops."},
        ]},
        {"id": "broadcast", "title": "Broadcast hash join and the 10 MB threshold", "blocks": [
            {"type": "p", "text": "Spark's broadcast hash join is HJ1 spread over a cluster: the driver collects the small side, builds H<sub>R</sub>, and "
                                  "sends a copy to every executor. Each executor probes it with its own partitions of the large side, so the large "
                                  "side never moves. Spark picks it automatically when a side's estimated size is at most "
                                  "<code>spark.sql.autoBroadcastJoinThreshold</code>, 10 MB by default (beyond the slides)."},
            {"type": "eq", "text": "broadcast bytes = size(build side) × executors\nshuffle bytes = size(left) + size(right)",
             "where": [["size", "Spark's estimate: file size, or rows × row width for an in-memory table"],
                       ["executors", "every executor receives its own copy"]]},
            {"type": "figure", "figure": "sj-moved", "title": "Bytes moved by each strategy at scale",
             "caption": "100,000 customers (10 MB) joined with 100 million orders (5 GB). Change the number of executors.",
             "controls": [{"k": "executors", "label": "Executors", "min": 1, "max": 32, "step": 1, "value": 4}]},
            {"type": "figure", "figure": "sj-threshold", "title": "When does broadcasting stop paying?",
             "caption": "Modelled time as the customers table grows, orders fixed at 5 GB. Both axes are logarithmic.",
             "controls": [{"k": "executors", "label": "Executors", "min": 2, "max": 32, "step": 1, "value": 4}]},
            {"type": "quiz", "q": "A LEFT OUTER JOIN from a 2 MB customers table to a 50 GB orders table. Can Spark broadcast customers?",
             "options": ["Yes, it is under 10 MB", "No: a left outer join can only build (broadcast) the right side", "Only with 2 executors"], "answer": 1,
             "why": "To null-extend customers with no orders, the customers rows must be streamed, not built: each executor would see only some of the orders, "
                    "so no single executor could know that a broadcast customer matched nothing. Spark therefore shuffles both sides."},
            {"type": "try", "label": "Watch the threshold flip the plan", "params": {"source": "synthetic", "lrows": 5, "lbytes": 100, "rrows": 8, "threshold": 10, "partitions": 200}},
        ]},
        {"id": "shuffle", "title": "Shuffle joins: hash and sort-merge", "blocks": [
            {"type": "p", "text": "When neither side can be broadcast, both are <b>shuffled</b>: every row is sent to partition "
                                  "pmod(murmur3(cust_id), n), so equal keys from both tables meet in the same task. Inside each partition Spark "
                                  "either builds a hash table (shuffle hash join) or sorts both sides and merges them (sort-merge join)."},
            {"type": "steps", "title": "Sort-merge join on the course tables (threshold −1, 8 partitions)", "items": [
                f"Exchange: each of the {_J1['tables']['left']['rows'] + _J1['tables']['right']['rows']} rows is written to the shuffle file of its partition ({_b(_J1['metrics']['shuffle_bytes'])} in all).",
                "Sort: each partition sorts its customers and its orders by cust_id; sorting can spill to disk, so memory never runs out.",
                "Merge: two cursors walk the sorted lists; equal keys produce output, the smaller key advances.",
                f"The busiest partition holds {_J1['metrics']['straggler_ratio']:.1f}× the median rows: hashing 7 customer ids into 8 partitions leaves some empty.",
            ]},
            {"type": "compare", "items": [
                {"title": "Shuffle hash join", "points": ["Builds a hash table of one side per partition", "No sort: faster when a side is small per partition",
                                                          "That table must fit in memory", "Spark only chooses it with a hint or preferSortMergeJoin = false"]},
                {"title": "Sort-merge join", "points": ["Sorts both sides of every partition", "Sorting spills to disk: works for any size",
                                                       "Spark's default for large equi-joins", "Output is sorted by the key, which later steps can reuse"]},
            ]},
            {"type": "figure", "figure": "sj-skew", "title": "Skewed keys make a straggler",
             "caption": "1 million customers and 1 billion orders whose customer ids follow a Zipf law, hashed into 16 partitions. Raise the skew.",
             "controls": [{"k": "skew", "label": "Zipf s", "min": 0, "max": 2, "step": 0.1, "value": 1.2}]},
            {"type": "callout", "tone": "tip", "title": "Adaptive execution splits skew (beyond the slides)",
             "text": f"With AQE, Spark marks a shuffle partition as skewed if it is over {S.SKEW_FACTOR}× the median and over {_b(S.SKEW_MIN_BYTES)}, "
                     "splits it into several tasks and reads the other side's matching partition once per piece. The live view's AQE switch shows the extra tasks and the re-read bytes."},
            {"type": "try", "label": "Skewed keys without and with AQE", "params": {"source": "synthetic", "lrows": 6, "rrows": 9, "skew": 1.2, "threshold": -1, "partitions": 200, "aqe": True}},
        ]},
        {"id": "copart", "title": "The co-partitioned (\"map-side\") join", "blocks": [
            {"type": "p", "text": "If both tables were written <code>CLUSTERED BY (cust_id) INTO n BUCKETS</code> with the same n, bucket i of customers and "
                                  "bucket i of orders already hold the same keys, because bucketing uses the same pmod(murmur3(key), n) as the shuffle. "
                                  "Spark then plans the join without an Exchange: no data moves before the join."},
            {"type": "callout", "tone": "warn", "title": "Not a separate Spark SQL strategy",
             "text": "The deck lists \"Map Side Join\" as a strategy of its own. In Spark SQL it is a sort-merge (or shuffle hash) join whose Exchange the planner "
                     "removed because the inputs are already partitioned on the key; EXPLAIN shows SortMergeJoin with no Exchange under it. "
                     "Broadcast hash join is also sometimes called a map-side join, which adds to the confusion."},
            {"type": "try", "label": "Bucket both tables", "params": {"source": "course", "layout": "bucketed", "threshold": -1, "partitions": 8}},
        ]},
        {"id": "nested", "title": "Nested loops for conditions without a key", "blocks": [
            {"type": "p", "text": "For a range condition Spark compares rows pair by pair. If one side can be broadcast it uses a <b>broadcast nested-loop "
                                  "join</b>; otherwise, for an inner join, a <b>Cartesian product</b> that pairs every partition of one side with every "
                                  "partition of the other."},
            {"type": "figure", "figure": "sj-nested", "title": "Why nested loops are the last resort",
             "caption": "Work for each algorithm as both sides grow (log–log axes)."},
            {"type": "callout", "tone": "warn", "title": "The slide's pseudocode broadcasts the wrong thing",
             "text": "p12 writes 'for each partition p1 in relation_1: broadcast partition p1 to all partitions of relation_2', which sends every partition of "
                     "relation_1 everywhere: that is the Cartesian (shuffle-and-replicate) product, not a broadcast nested loop. The broadcast nested-loop join "
                     "broadcasts the whole smaller relation once; each partition of the other relation then loops over that one copy. "
                     f"On the course tables that is {int(_JB['metrics']['nl_comparisons'])} comparisons and {_b(_JB['metrics']['broadcast_bytes'])} broadcast."},
            {"type": "quiz", "q": "A FULL OUTER join with a BETWEEN condition and both tables at 50 GB. What does Spark do?",
             "options": ["Sort-merge join", "Cartesian product", "Broadcast nested-loop join, broadcasting a 50 GB side"], "answer": 2,
             "why": "Sort-merge needs an equality, and the Cartesian product only does inner joins. Spark's last resort is a broadcast nested loop on the smaller "
                    "side even when it is far over the threshold, which usually fails with an out-of-memory error."},
        ]},
        {"id": "planner", "title": "How the planner decides", "blocks": [
            {"type": "steps", "title": "Spark 3's JoinSelection, in order", "items": [
                "Hints first: BROADCAST, SHUFFLE_HASH, MERGE, SHUFFLE_REPLICATE_NL, if the join type allows them.",
                "Equi-join: broadcast hash join if a side the join type can build is at most autoBroadcastJoinThreshold.",
                "Shuffle hash join if preferSortMergeJoin is false, the side fits a hash map per partition and is 3× smaller.",
                "Otherwise sort-merge join, if the keys can be sorted.",
                "Non-equi join: broadcast nested loop if a side can be broadcast; Cartesian product for an inner join; else broadcast nested loop anyway.",
            ]},
            {"type": "figure", "figure": "sj-planner-map", "title": "The planner's choice for every pair of sizes",
             "caption": "Each cell is one (customers, orders) size pair with the default 10 MB threshold. Switch the join type.",
             "controls": [{"k": "join", "label": "Join type", "options": _JOIN_OPTS, "value": "inner"}]},
            {"type": "figure", "figure": "sj-agree", "title": "Same rows from every strategy",
             "caption": "Each possible strategy executed on the course tables and compared with an all-pairs join.",
             "controls": [{"k": "join", "label": "Join type", "options": _JOIN_OPTS, "value": "left"}]},
            {"type": "quiz", "q": "Which change can turn a broadcast hash join into a sort-merge join without touching the data?",
             "options": ["Adding executors", "SET spark.sql.autoBroadcastJoinThreshold = -1", "Changing the order of the tables in an inner join"], "answer": 1,
             "why": "−1 disables automatic broadcasting, so the size rule fails for both sides and the planner falls through to sort-merge. "
                    "Executors change the cost, not the decision; an inner join may build either side."},
            {"type": "try", "label": "Full outer join on the course tables", "params": {"source": "course", "join": "full", "threshold": 10}},
        ]},
        {"id": "slides", "title": "Where the slides need care", "blocks": [
            {"type": "callout", "tone": "warn", "title": "Corrections to the join-strategies deck",
             "text": "(1) p2 announces five strategies and lists six. (2) \"Map side join\" is not a separate Spark SQL operator: it is a sort-merge or shuffle "
                     "hash join whose Exchange is skipped because the tables are co-partitioned (bucketed). (3) The nested-loop pseudocode on p12 broadcasts "
                     "every partition, which describes the Cartesian product; a broadcast nested loop sends the smaller side once. (4) p5's \"all join types "
                     "except full outer\" hides a constraint: a left outer join can only broadcast the right side, a right outer join only the left. "
                     "(5) \"Spark allows broadcast DataFrames larger than 10 MB\": 10 MB is only the automatic threshold; hints or a higher setting broadcast "
                     "more, up to a hard 8 GB limit."},
            {"type": "list", "items": [
                f"The live view runs every possible strategy on the same rows and checks them against an all-pairs join: {sum(v['feasible'] for v in _JL['validation'])} strategies agree on {_JL['result']['total']} rows for the left join.",
                "Murmur3 with seed 42 is Spark's hash, reproduced exactly, so the partition numbers in the live view are the ones Spark would use.",
                "Times come from a deliberately simple cost model and are good for comparing strategies, not for predicting a real cluster.",
            ]},
        ]},
    ],
    "takeaways": [
        "A join's cost is decided by what it moves: broadcast copies the small side to every executor, a shuffle moves both sides once.",
        "Spark broadcasts automatically below spark.sql.autoBroadcastJoinThreshold (10 MB); −1 turns that off.",
        "Outer joins restrict the build side: left outer can only broadcast the right table, full outer can never use broadcast hash.",
        "Sort-merge is the default for large equi-joins because sorting can spill; skewed keys make one partition the straggler.",
        "Bucketed tables join without an Exchange; that is the co-partitioned (\"map-side\") join, inside a sort-merge join.",
        "Without an equality only nested loops remain: |L| × |R| comparisons.",
    ],
    "glossary": [
        ["Build side", "The table a hash join loads into a hash table (and broadcasts, in a broadcast join)."],
        ["Probe (stream) side", "The table whose rows look up the hash table one by one."],
        ["Exchange", "Spark's shuffle operator: rows are repartitioned by a hash of the key."],
        ["Straggler", "The slowest task of a stage; the stage ends only when it does."],
        ["Bucketing", "Writing a table pre-partitioned by pmod(hash(column), n) into n files per directory."],
        ["AQE", "Adaptive query execution: Spark re-plans between stages using real partition sizes."],
        ["Equi-join", "A join whose condition is an equality between key columns."],
    ],
}, order=7)


# ===================================================================== SQL: figures
@figure("sql-pipeline")
def _fig_pipeline(p):
    r = S.run_sql(query="slide16")
    fired = {s["rule"]: s["fired"] for s in r["stages"]}
    sv = r["saving"]
    nodes = [{"id": "q", "x": 6, "y": 82, "w": 98, "h": 64, "title": "DataFrame", "sub": "slide-16 query"},
             {"id": "a", "x": 124, "y": 82, "w": 100, "h": 64, "title": "Analysed", "sub": "Filter above Join"}]
    edges = [{"from": "q", "to": "a", "width": 2}]
    for i, (k, name, _) in enumerate(S.SQL_RULES):
        nodes.append({"id": k, "x": 252, "y": 12 + i * 72, "w": 150, "h": 56, "title": name, "sub": "changed the plan" if fired.get(k) else "nothing to do", "hot": bool(fired.get(k))})
        edges.append({"from": "a", "to": k, "width": 2})
        edges.append({"from": k, "to": "p", "width": 2})
    nodes += [{"id": "p", "x": 430, "y": 82, "w": 92, "h": 64, "title": "Physical", "sub": "BroadcastHashJoin"},
              {"id": "r", "x": 536, "y": 82, "w": 58, "h": 64, "title": "Rows", "value": str(len(r["result"]["rows"]))}]
    edges.append({"from": "p", "to": "r", "width": 2})
    return {"kind": "flow", "height": 230, "nodes": nodes, "edges": edges,
            "notes": [f"The three rules cut the values read from {sv['none_cells']} to {r['metrics']['cells_read']} and the rows entering the join from "
                      f"{sv['none_join_rows']} to {r['metrics']['rows_into_join']}.",
                      f"The answer is unchanged: {', '.join(f'{row[1]} {row[2]}' for row in r['result']['rows'])}."]}


_QOPTS = [[k, v["label"]] for k, v in S.SQL_QUERIES.items()]
_VSHORT = {"No optimisation": "none", "ConstantFolding only": "fold", "PushDownPredicates only": "push", "ColumnPruning only": "prune", "All three rules": "all"}


@figure("sql-rules")
def _fig_rules(p):
    q = p.choice("query", "slide16", list(S.SQL_QUERIES))
    r = S.run_sql(query=q)
    V = r["variants"][:5]
    best = min(V, key=lambda v: v["cells_read"])
    return {"kind": "bars", "labels": [_VSHORT[v["label"]] for v in V], "ylabel": "Count",
            "series": [{"name": "values read", "color": "s1", "values": [v["cells_read"] for v in V]},
                       {"name": "rows into the join", "color": "s2", "values": [v["rows_into_join"] for v in V]},
                       {"name": "expression evaluations", "color": "s3", "values": [v["evals"] for v in V]}],
            "notes": [f"{S.SQL_QUERIES[q]['label']} ({S.SQL_QUERIES[q]['slide']}): {V[0]['cells_read']} values read with no rule, {best['cells_read']} at best; "
                      f"every variant returns the same {len(r['result']['rows'])} rows.",
                      "; ".join(f"{_VSHORT[v['label']]}: {v['cells_read']} values, {v['evals']} evaluations" for v in V[1:4]) + "."]}


@figure("sql-pushdown")
def _fig_pushdown(p):
    g = p.choice("gender", "F", ["F", "M"])
    q = S.build_query("slide16", gender=g)
    before = S.execute_plan(q["plan"])[3]
    opt, _ = S.optimise(q["plan"], {"pushdown": True})
    after = S.execute_plan(opt)[3]

    def counts(trace, op):
        return [t for t in trace if t["op"] == op]
    bj, aj = counts(before, "Join")[0], counts(after, "Join")[0]
    bf, af = counts(before, "Filter")[0], counts(after, "Filter")[0]
    nodes = [
        {"id": "e1", "x": 8, "y": 150, "w": 120, "h": 52, "title": "Scan employee", "value": f"{before[0]['rows_out']} rows"},
        {"id": "d1", "x": 150, "y": 150, "w": 120, "h": 52, "title": "Scan department", "value": f"{before[1]['rows_out']} rows"},
        {"id": "j1", "x": 78, "y": 82, "w": 120, "h": 52, "title": "Join", "sub": f"{bj['rows_in']} rows in → {bj['rows_out']}"},
        {"id": "f1", "x": 78, "y": 14, "w": 120, "h": 52, "title": "Filter", "sub": f"gender = '{g}': {bf['rows_in']} → {bf['rows_out']}"},
        {"id": "e2", "x": 330, "y": 150, "w": 120, "h": 52, "title": "Scan employee", "value": f"{after[0]['rows_out']} rows"},
        {"id": "f2", "x": 330, "y": 82, "w": 120, "h": 52, "title": "Filter", "sub": f"gender = '{g}': {af['rows_in']} → {af['rows_out']}", "hot": True},
        {"id": "d2", "x": 470, "y": 82, "w": 120, "h": 52, "title": "Scan department", "value": f"{after[2]['rows_out'] if after[2]['op'] == 'Scan' else 3} rows"},
        {"id": "j2", "x": 400, "y": 14, "w": 120, "h": 52, "title": "Join", "sub": f"{aj['rows_in']} rows in → {aj['rows_out']}"},
    ]
    edges = [{"from": "e1", "to": "j1"}, {"from": "d1", "to": "j1"}, {"from": "j1", "to": "f1"},
             {"from": "e2", "to": "f2"}, {"from": "f2", "to": "j2"}, {"from": "d2", "to": "j2"}]
    return {"kind": "flow", "height": 214, "nodes": nodes, "edges": [dict(e, width=2) for e in edges],
            "texts": [{"x": 140, "y": 210, "text": "as written", "cls": "n-sub"}, {"x": 460, "y": 210, "text": "after PushPredicateThroughJoin", "cls": "n-sub"}],
            "notes": [f"As written, all {bj['rows_in']} rows enter the join and the filter then throws away {bf['rows_in'] - bf['rows_out']} of its {bf['rows_in']} output rows.",
                      f"Pushed below the join, the filter keeps {af['rows_out']} of {af['rows_in']} employees first, so only {aj['rows_in']} rows enter the join; the result is the same."]}


@figure("sql-pruning")
def _fig_pruning(p):
    labels, off, on = [], [], []
    for k, v in S.SQL_QUERIES.items():
        r = S.run_sql(query=k)
        labels.append(v["label"])
        off.append(r["variants"][0]["cols_read"])
        on.append(r["variants"][3]["cols_read"])
    best = max(range(len(labels)), key=lambda i: off[i] - on[i])
    return {"kind": "bars", "labels": labels, "ylabel": "Columns read",
            "series": [{"name": "without ColumnPruning", "color": "neutral", "values": off}, {"name": "with ColumnPruning", "color": "s1", "values": on}],
            "notes": [f"Queries that SELECT * keep all columns; the others shed the rest. The biggest saving is {labels[best]}: {off[best]} columns down to {on[best]}.",
                      "Parquet stores each column in its own chunk, so a pruned column is never read from disk at all."]}


@figure("sql-folding")
def _fig_folding(p):
    ks = list(range(0, 101, 5))
    folded, unfolded = [], []
    for k in ks:
        q = S.build_query("filter", k=k)
        folded.append(S.execute_plan(S.optimise(q["plan"], {"folding": True, "pushdown": True})[0])[2]["rows_read"])
        unfolded.append(S.execute_plan(S.optimise(q["plan"], {"pushdown": True})[0])[2]["rows_read"])
    first = next((k for k, v in zip(ks, folded) if v == 0), None)
    mx = max(r[4] for r in S.EMPLOYEE)
    return {"kind": "lines", "x": ks, "xlabel": "k in WHERE salary > k * 1000", "ylabel": "Rows read from the file", "ymin": 0,
            "series": [{"name": "ConstantFolding + pushdown", "color": "s1", "y": folded},
                       {"name": "pushdown without folding", "color": "s2", "y": unfolded, "dash": True}],
            "markers": [{"x": mx / 1000, "label": f"max salary {mx:,}"}],
            "notes": [f"Without folding the filter stays salary > (k * 1000), which a file scan cannot evaluate, so all {unfolded[0]} rows are always read.",
                      f"Folded to a literal, the filter is pushed to the scan; from k = {first} the file's salary max ({mx:,}) proves no row can match and the row group is skipped."]}


_GRADES = ["AA", "AB", "BB", "BC", "CC", "CD", "DD", "DE", "F"]


@figure("sql-grades")
def _fig_grades(p):
    u = S.letter_grade_report()
    vals = [u["counts"].get(g, 0) for g in _GRADES]
    return {"kind": "bars", "labels": _GRADES, "xlabel": "Grade", "ylabel": "Students",
            "series": [{"name": "students", "color": "s1", "values": vals}],
            "notes": [f"The 19 marks of marks.csv graded by letterGrade as written: {', '.join(f'{g} {c}' for g, c in zip(_GRADES, vals) if c)}.",
                      f"The first ten grades are {' '.join(u['first10'])}, exactly the slide's output. DD covers {u['duplicate']['range']} "
                      f"through two branches, so {u['duplicate']['students']} students land in one wide band."]}


@figure("sql-udf-steps")
def _fig_udf_steps(p):
    mark = p.num("mark", 38, 0, 100)
    xs = [x / 2 for x in range(0, 201)]
    rank = {g: len(_GRADES) - i for i, g in enumerate(_GRADES)}
    ys = [rank[S.letter_grade(x)] for x in xs]
    g = S.letter_grade(mark)
    branch = next((f"marks {o} {b}" for o, b, gr in S.LETTER_GRADE_RULES if ((mark >= b) if o == ">=" else (mark > b))), "otherwise")
    return {"kind": "lines", "x": xs, "xlabel": "marks", "ylabel": "grade (F = 1 … AA = 9)", "ymin": 0, "ymax": 10,
            "series": [{"name": "letterGrade as written", "color": "s1", "y": ys}],
            "bands": [{"x0": 35, "x1": 45, "label": "DD twice"}],
            "points": [{"x": mark, "y": rank[g], "color": "accent", "label": g}],
            "notes": [f"{mark:g} marks → {g}, from the first branch that holds: {branch}.",
                      "The curve is flat across (35, 45]: the 'marks > 40' and 'marks > 35' branches both return DD, so the band is twice as wide as its neighbours."]}


_SQ = S.run_sql(query="slide16")
_SQ_OFF = S.run_sql(query="slide16", folding=False, pushdown=False, pruning=False)
_SQF = S.run_sql(query="filter", k=60)
_U = S.letter_grade_report()

explainer("bigdata/sql", {
    "title": "Spark SQL: from a query to a physical plan",
    "kicker": "Big Data Lab · Spark SQL and DataFrames",
    "summary": "A DataFrame chain and a SQL string both become the same logical plan, which Spark's Catalyst optimiser rewrites rule by rule "
               "before choosing physical operators. On the course's company tables this page applies constant folding, predicate pushdown and "
               "column pruning one at a time, counts what each saves, and runs the deck's letterGrade UDF exactly as written.",
    "sections": [
        {"id": "pipeline", "title": "One optimiser for SQL and DataFrames", "blocks": [
            {"type": "p", "text": "The SQL intro deck's slide 16 writes <code>employees.join(dept, …).where(gender = \"female\").groupBy(dept.id, dept.name).agg(count(name))</code>. "
                                  "Nothing runs when that line is evaluated: DataFrames are lazy, so Spark records a plan and optimises the whole of it when an action such as show() asks for rows."},
            {"type": "cards", "items": [
                {"title": "Analysed logical plan", "text": "What you asked for, with names resolved against the catalog: Filter above Join, every column of every table."},
                {"title": "Optimised logical plan", "text": "The same answer, rewritten by rules such as constant folding, predicate pushdown and column pruning."},
                {"title": "Physical plan", "text": "Concrete operators: a file scan with pushed filters, a broadcast hash join, a two-step hash aggregate."},
            ]},
            {"type": "figure", "figure": "sql-pipeline", "title": "The slide-16 query through Catalyst",
             "caption": "Lit rule boxes changed the plan for this query."},
        ]},
        {"id": "data", "title": "The course tables", "blocks": [
            {"type": "p", "text": f"The DataFrame deck's employee schema (p14) is the company database of the DB book: eno, name, dob, gender, salary, sup_eno, dno. "
                                  f"Its {len(S.EMPLOYEE)} employees are the course's EMP8 (RDD deck p28); the department table has {len(S.DEPARTMENT)} rows: "
                                  + ", ".join(f"{d[0]} {d[1]}" for d in S.DEPARTMENT) + "."},
            {"type": "callout", "tone": "info", "title": "Where the extra columns come from",
             "text": "dob, gender, sup_eno and the department names are not printed on these decks; they are the values of the same company database the deck cites. "
                     "Gender is coded M/F as in employee_m.csv, so the slide's gender === \"female\" becomes gender = 'F'."},
        ]},
        {"id": "folding", "title": "Rule 1: constant folding", "blocks": [
            {"type": "p", "text": "An expression built only from literals has the same value for every row. Catalyst evaluates it once, while planning. "
                                  "The live view writes the threshold as k * 1000 so that there is something to fold."},
            {"type": "eq", "text": "salary > (30 * 1000)  ⟶  salary > 30000\nevaluations saved = rows × folded operators",
             "where": [["30 * 1000", "a constant expression, computed once at planning time"], ["rows", "rows the filter would otherwise evaluate it for"]]},
            {"type": "figure", "figure": "sql-folding", "title": "Folding is what lets the filter reach the file",
             "caption": "Rows read from the employee file as the threshold rises, with and without constant folding (pushdown on in both)."},
            {"type": "try", "label": "A threshold above every salary", "params": {"query": "filter", "k": 60, "folding": True, "pushdown": True, "pruning": True}},
        ]},
        {"id": "pushdown", "title": "Rule 2: predicate pushdown", "blocks": [
            {"type": "p", "text": "A filter that only uses one table's columns can run before the join, on that table alone. "
                                  "If it compares a column with a literal it can go further, into the file scan itself (PushedFilters), "
                                  "where Parquet's min/max statistics can skip data without reading it."},
            {"type": "steps", "items": [
                "Split the WHERE clause into conjuncts joined by AND.",
                "A conjunct that references only employee columns moves below the join, onto the employee side.",
                "At the scan, conjuncts of the form column op literal become PushedFilters; anything else (arithmetic, a UDF) stays in a Filter.",
                "The Filter above the scan is kept: pushed filters only skip whole row groups, they do not check every row.",
            ]},
            {"type": "figure", "figure": "sql-pushdown", "title": "The slide-16 plan before and after pushdown",
             "caption": "Row counts at each operator, executed on the course tables. Switch the gender filter.",
             "controls": [{"k": "gender", "label": "gender =", "options": [["F", "F"], ["M", "M"]], "value": "F"}]},
            {"type": "quiz", "q": "Why can't Spark push WHERE grade = 'DD' (where grade = letterGrade(marks)) into the marks scan?",
             "options": ["CSV files have no statistics", "The predicate depends on a Python UDF the scan cannot evaluate", "Pushdown only works for joins"], "answer": 1,
             "why": "A file source only accepts simple comparisons between a column and a literal. A UDF is opaque to Catalyst, so the filter has to run after the UDF does."},
        ]},
        {"id": "pruning", "title": "Rule 3: column pruning", "blocks": [
            {"type": "p", "text": "Columns that no operator above the scan uses are dropped from the scan's read list. In a columnar format like Parquet a pruned column is never read."},
            {"type": "figure", "figure": "sql-pruning", "title": "Columns read per course query", "caption": "Without and with ColumnPruning, summed over the tables each query scans."},
            {"type": "figure", "figure": "sql-rules", "title": "What each rule saves on one query",
             "caption": "The query executed with no rule, each rule alone and all three. Pick a query.",
             "controls": [{"k": "query", "label": "Query", "options": _QOPTS, "value": "slide16"}]},
            {"type": "quiz", "q": f"With all rules on, the slide-16 query reads {_SQ['metrics']['cells_read']} values. With none?",
             "options": [str(_SQ_OFF["metrics"]["cells_read"]), str(_SQ["metrics"]["cells_read"]), str(_SQ["metrics"]["cells_read"] // 2)], "answer": 0,
             "why": f"Without pruning every column of both tables is read: 8 × 7 + 3 × 4 = {_SQ_OFF['metrics']['cells_read']} values. With pruning only name, gender and dno of employee and dno, name of department are read."},
        ]},
        {"id": "physical", "title": "Physical planning", "blocks": [
            {"type": "p", "text": "The optimised plan says what to compute; the physical plan says how. The department table is three rows, far below the 10 MB threshold, "
                                  "so the join is a BroadcastHashJoin with department as the build side. The aggregate runs in two steps: a partial count in each task, "
                                  "an Exchange by department, then the final count."},
            {"type": "compare", "items": [
                {"title": "Broadcast (the default here)", "points": ["department copied to every executor", "employee never shuffled", "one Exchange, for the aggregate"]},
                {"title": "Threshold −1", "points": ["both sides shuffled by dno", "both sides sorted", "SortMergeJoin, then the aggregate's Exchange"]},
            ]},
            {"type": "try", "label": "Turn auto-broadcast off", "params": {"query": "slide16", "broadcast": False}},
        ]},
        {"id": "udf", "title": "A user-defined function: letterGrade", "blocks": [
            {"type": "p", "text": "The DDL deck registers a Python function as a SQL function (p40) and calls it in a query (p41). Spark calls it once per row, shipping rows in batches to a "
                                  "Python worker (BatchEvalPython in the physical plan); the optimiser cannot look inside it."},
            {"type": "figure", "figure": "sql-udf-steps", "title": "letterGrade, as written on the slide",
             "caption": "The grade for each mark. Move the mark.",
             "controls": [{"k": "mark", "label": "marks", "min": 0, "max": 100, "step": 1, "value": 38}]},
            {"type": "figure", "figure": "sql-grades", "title": "The 19 marks of marks.csv, graded", "caption": "Counts per grade with the function exactly as on p40."},
            {"type": "callout", "tone": "warn", "title": "The slide's thresholds repeat \"DD\"",
             "text": "p40 has elif marks > 40: grade = \"DD\" followed by elif marks > 35: grade = \"DD\". Both branches return DD, so DD covers 35 < marks ≤ 45 while every "
                     f"other grade covers 5 marks; the second branch was probably meant to be another grade. The live view runs it as written, and its first ten grades "
                     f"({' '.join(_U['first10'])}) match the slide's output row for row."},
            {"type": "try", "label": "Run letterGrade on marks.csv", "params": {"query": "udf"}},
        ]},
        {"id": "slides", "title": "Where the slides need care", "blocks": [
            {"type": "callout", "tone": "warn", "title": "Corrections to the SQL decks",
             "text": "(1) The duplicate DD branch in letterGrade (DDL deck p40). (2) Intro deck p16 ends with agg(count(\"name\")) after joining two tables that both have a "
                     "name column; Spark would reject the bare name as ambiguous, so the live view counts employees(\"name\"). (3) DataFrame deck p13 prints the "
                     "schema with \"emp-no\" while the same file's header on p11 says empno. (4) DDL deck p41's \"update marks set grade=letterGrade(marks) from marks\" "
                     "is not valid Spark SQL: UPDATE is not supported on plain Parquet or CSV tables (it needs a table format such as Delta Lake), and the FROM clause does not belong there."},
            {"type": "p", "text": "Everything on this page is computed by a small re-implementation of the analyser, three Catalyst rules and the executor in plain Python; "
                                  "it follows Spark's documented behaviour but is not Spark."},
        ]},
    ],
    "takeaways": [
        "DataFrame code and SQL produce the same logical plan and share one optimiser.",
        "Constant folding computes literal-only expressions once, and makes filters simple enough to push into the scan.",
        "Predicate pushdown moves filters below joins and into Parquet scans, where min/max statistics can skip whole row groups.",
        "Column pruning reads only the columns a query uses; with Parquet the rest never leave the disk.",
        "Rules change how much is read, never the answer: every variant returns the same rows.",
        "A Python UDF is opaque to Catalyst; filters on its output cannot be pushed down.",
    ],
    "glossary": [
        ["Logical plan", "A tree of relational operators (Relation, Filter, Project, Join, Aggregate, Sort) saying what to compute."],
        ["Physical plan", "The operators that actually run: FileScan, BroadcastHashJoin, HashAggregate, Exchange…"],
        ["Catalyst", "Spark SQL's rule-based (and cost-based) query optimiser."],
        ["PushedFilters", "Predicates handed to the file source, which may use file statistics to skip data."],
        ["Row group", "A horizontal slice of a Parquet file, with min/max statistics per column."],
        ["UDF", "A user-defined function registered with Spark and called from SQL or DataFrames."],
    ],
}, order=8)


# ===================================================================== Tables: figures
@functools.lru_cache(maxsize=8)
def _emp(rows: int = 4000):
    return S.synthetic_employees(rows)


@functools.lru_cache(maxsize=64)
def _layout(**kw):
    return S.table_layout(**kw)


@figure("tbl-layout")
def _fig_layout(p):
    r = _layout()
    states = r["tree"]
    nodes = [{"id": "root", "x": 225, "y": 8, "w": 150, "h": 48, "title": "employees/", "sub": f"{r['rows']:,} rows"}]
    edges = []
    for i, st in enumerate(states):
        files = sum(g["files_total"] for g in st["children"])
        nodes.append({"id": st["name"], "x": 6 + i * 119, "y": 92, "w": 110, "h": 54, "title": st["name"], "sub": f"{st['rows']} rows · {files} files",
                      "hot": st["name"] == "state=CA"})
        edges.append({"from": "root", "to": st["name"], "width": 1.5 + 6 * st["rows"] / r["rows"]})
    ca = next(s for s in states if s["name"] == "state=CA")
    for j, g in enumerate(ca["children"]):
        nid = "CA/" + g["name"]
        nodes.append({"id": nid, "x": 8 + j * 122, "y": 178, "w": 112, "h": 50, "title": g["name"], "sub": f"{g['files_total']} files"})
        edges.append({"from": "state=CA", "to": nid, "width": 2})
    return {"kind": "flow", "height": 240, "nodes": nodes, "edges": edges,
            "notes": [f"{r['write']['dirs']} directories, one per (state, gender) value pair that occurs, holding {r['write']['files']} files.",
                      f"The state and gender values live only in the directory names: the files store {', '.join(S.DATA_COLS)}, so a filter on state is answered by listing folders."]}


@figure("tbl-buckets")
def _fig_buckets(p):
    n = p.num("buckets", 8, 1, 16, integer=True)
    emp = _emp()
    counts = np.bincount(S.partition_of(emp["empno"], n), minlength=n)
    lb = S.murmur3_int(S.LOOKUP_EMPNO) % n
    return {"kind": "bars", "labels": [str(i) for i in range(n)], "xlabel": "Bucket", "ylabel": "Employees",
            "series": [{"name": "employees", "color": "s1", "values": counts.tolist()}],
            "notes": [(f"pmod(murmur3(empno), {n}) spreads {emp['n']:,} employees from {counts.min()} to {counts.max()} per bucket: hashing balances unique keys well."
                       if n > 1 else f"With one bucket all {emp['n']:,} employees are in bucket 0: bucketing only pays off with several buckets."),
                      f"empno {S.LOOKUP_EMPNO} (row 2 of employee_m.csv) hashes to {S.murmur3_int(S.LOOKUP_EMPNO)}, so it is always in bucket {lb}: "
                      "a query for it can open one bucket file per directory."]}


@figure("tbl-skipping")
def _fig_skipping(p):
    rg = p.num("rg_rows", 16, 4, 128, integer=True)
    emp = _emp()
    ths = list(range(40000, 122001, 4000))
    out = {}
    for srt in (False, True):
        files = [f for f in S.write_table(emp, buckets=4, tasks=2, distribute=False, sort_salary=srt, rg_rows=rg) if f["state"] == "CA"]
        rgs = [g for f in files for g in f["row_groups"]]
        out[srt] = ([sum(1 for g in rgs if g["max_salary"] > t) / len(rgs) for t in ths], len(rgs))
    i90 = ths.index(88000)
    return {"kind": "lines", "x": ths, "xlabel": "WHERE salary > …", "ylabel": "Share of row groups read", "ymin": 0, "ymax": 1,
            "series": [{"name": "unsorted files", "color": "s2", "y": out[False][0]}, {"name": "SORTED BY (salary DESC)", "color": "s1", "y": out[True][0]}],
            "notes": [f"In the state=CA files ({out[False][1]} row groups of up to {rg} rows), salary > 88000 reads {100 * out[False][0][i90]:.0f}% of the row groups unsorted "
                      f"and {100 * out[True][0][i90]:.0f}% when each file is sorted by salary.",
                      ("Unsorted, almost every group contains a high salary somewhere, so its max passes the filter; sorting packs the high salaries into a few groups."
                       if out[True][0][i90] < out[False][0][i90] - 0.05 else
                       "With row groups this large each file is about one group, which always contains a high salary: statistics cannot skip anything, sorted or not.")]}


@figure("tbl-funnel")
def _fig_funnel(p):
    st = p.choice("state", "CA", ["all"] + S.TBL_STATES)
    r = _layout(f_state=st)
    b = r["read"]["bytes"]
    keys = [("total", "on disk"), ("partitions", "partitions"), ("buckets", "buckets"), ("row_groups", "row groups"), ("columns", "columns")]
    return {"kind": "bars", "labels": [k[1] for k in keys], "ylabel": "KB", "series": [{"name": "KB to read", "color": "s1", "values": [b[k] / 1024 for k, _ in keys]}],
            "notes": [f"{r['query']['sql']}: {_b(b['columns'])} of {_b(b['total'])} is read ({100 * b['columns'] / b['total']:.1f}%), "
                      f"{r['read']['dirs_read']} of {r['read']['dirs_total']} directories and {r['read']['row_groups_read']} row groups.",
                      f"The answer, {r['read']['result_rows']} rows, is identical to filtering the whole table: pruning never changes the result."]}


@figure("tbl-files")
def _fig_files(p):
    n = p.num("buckets", 8, 1, 16, integer=True)
    emp = _emp()
    plain, dist = [], []
    for t in range(1, 9):
        plain.append(len(S.write_table(emp, buckets=n, tasks=t, distribute=False, sort_salary=False, rg_rows=256)))
        dist.append(len(S.write_table(emp, buckets=n, tasks=t, distribute=True, sort_salary=False, rg_rows=256)))
    return {"kind": "bars", "labels": [str(t) for t in range(1, 9)], "xlabel": "Write tasks", "ylabel": "Files written",
            "series": [{"name": "plain INSERT", "color": "s2", "values": plain}, {"name": "DISTRIBUTE BY state, gender first", "color": "s1", "values": dist}],
            "notes": [f"With {n} bucket{'s' if n > 1 else ''}, 8 write tasks produce {plain[-1]} files, because every task writes its own file for every directory and bucket it holds rows of.",
                      f"Distributing by the partition columns first sends each directory to one task: {dist[-1]} files whatever the number of tasks."]}


@figure("tbl-drop")
def _fig_drop(p):
    kind = p.choice("kind", "managed", ["managed", "external"])
    d = _layout(kind=kind)["drop"]
    fs = d["before"]["fs"]
    kept = d["after"]["fs"] is not None
    nodes = [
        {"id": "m0", "x": 10, "y": 20, "w": 160, "h": 60, "title": "metastore", "sub": f"{d['before']['metastore']['type']} table"},
        {"id": "f0", "x": 10, "y": 140, "w": 160, "h": 64, "title": fs["name"].split("/")[-1] + "/", "value": f"{fs['files']} files", "sub": _b(fs["bytes"])},
        {"id": "x", "x": 240, "y": 82, "w": 120, "h": 56, "title": "DROP TABLE", "hot": True},
        {"id": "m1", "x": 430, "y": 20, "w": 160, "h": 60, "title": "metastore", "sub": "entry removed"},
        {"id": "f1", "x": 430, "y": 140, "w": 160, "h": 64, "title": fs["name"].split("/")[-1] + "/", "value": f"{fs['files']} files" if kept else "deleted",
         "sub": "untouched" if kept else "directory removed"},
    ]
    edges = [{"from": "m0", "to": "x", "width": 2}, {"from": "f0", "to": "x", "width": 2}, {"from": "x", "to": "m1", "width": 2},
             {"from": "x", "to": "f1", "width": 2, "color": "good" if kept else "bad"}]
    return {"kind": "flow", "height": 220, "nodes": nodes, "edges": edges,
            "notes": [d["note"], f"Before the drop the table's location {fs['name']} holds {fs['files']} files ({_b(fs['bytes'])}); "
                                 + ("they are still there afterwards and a new CREATE TABLE … LOCATION can use them again." if kept else "afterwards nothing is left.")]}


_CLAUSES = [["none", "none"], ["distribute", "DISTRIBUTE BY state"], ["distribute_sort", "+ SORT BY salary"], ["sort", "SORT BY salary"],
            ["cluster", "CLUSTER BY state"], ["order", "ORDER BY salary"]]


@figure("tbl-placement")
def _fig_placement(p):
    clause = p.choice("clause", "distribute", [c[0] for c in _CLAUSES])
    pl = S.place_rows(_emp(), clause, 24, 3)
    states = sorted({r[2] for part in pl["after"] for r in part})
    colors = ["s1", "s2", "s3", "s4", "neutral"]
    series = [{"name": st, "color": colors[i % 5], "values": [sum(1 for r in part if r[2] == st) for part in pl["after"]]} for i, st in enumerate(states)]
    c = pl["checks"]
    return {"kind": "bars", "labels": [f"partition {i}" for i in range(len(pl["after"]))], "ylabel": "Rows", "series": series,
            "notes": [f"{pl['sql']}: partitions of {', '.join(str(x) for x in c['sizes'])} rows; "
                      + ("every state sits in exactly one partition" if c["states_split"] == 0 else f"{c['states_split']} states are spread over several partitions") + ".",
                      ("Rows are sorted by salary inside each partition" if c["sorted_within"] else "Rows are not sorted inside the partitions")
                      + (", and the partitions in order form one globally sorted list." if c["globally_sorted"] else "; reading the partitions in order is not a global sort.")]}


_T0 = _layout()
_TS = _layout(sort_salary=True)
_TL = _layout(lookup=True, f_state="MA", f_salary=0)

explainer("bigdata/tables", {
    "title": "Tables on disk: partitions, buckets and Parquet",
    "kicker": "Big Data Lab · Spark SQL tables (DDL / DML)",
    "summary": "A Spark table is a metastore entry plus a directory of files. PARTITIONED BY makes one folder per value, CLUSTERED BY splits each folder "
               "into bucket files by a hash, and Parquet stores each file as row groups with per-column statistics. This page writes a table "
               "shaped like the course's employee_m.csv and follows a query as each layer lets it skip data.",
    "sections": [
        {"id": "tables", "title": "A table is metadata plus files", "blocks": [
            {"type": "p", "text": "Spark keeps every table's schema, format, partitioning and location in a metastore (often Hive's). The rows live in ordinary "
                                  "files: under the warehouse directory for a <b>managed</b> table, or at a LOCATION you give for an <b>external</b> (unmanaged) one."},
            {"type": "compare", "items": [
                {"title": "Managed", "points": ["Spark chooses the location (spark-warehouse/…)", "Spark owns metadata and files", "DROP TABLE deletes both"]},
                {"title": "External (LOCATION …)", "points": ["You choose the location, often existing data", "Spark owns only the metadata", "DROP TABLE leaves the files"]},
            ]},
            {"type": "figure", "figure": "tbl-drop", "title": "DROP TABLE", "caption": "The metastore and the file system before and after. Switch the table type.",
             "controls": [{"k": "kind", "label": "Table type", "options": [["managed", "Managed"], ["external", "External"]], "value": "managed"}]},
            {"type": "quiz", "q": "You created a table with LOCATION '/data/sales' over existing Parquet files and then ran DROP TABLE. What is left?",
             "options": ["Nothing", "The files in /data/sales", "The metastore entry"], "answer": 1,
             "why": "A table with a LOCATION is unmanaged: Spark registered only metadata, so dropping it removes only the metadata."},
        ]},
        {"id": "partition", "title": "PARTITIONED BY: one directory per value", "blocks": [
            {"type": "p", "text": f"The live view writes {_T0['rows']:,} employees shaped like employee_m.csv (the deck's three visible rows, then synthetic rows from a fixed seed) "
                                  "with <code>PARTITIONED BY (state, gender)</code>. Each (state, gender) pair becomes a Hive-style folder such as state=CA/gender=M/."},
            {"type": "eq", "text": "path = <table>/state=<s>/gender=<g>/part-<task>-<job>_<bucket>.c000.snappy.parquet",
             "where": [["state, gender", "partition columns: stored in the path, not inside the files"], ["task", "the write task that produced the file"],
                       ["bucket", "pmod(murmur3(empno), n) when the table is bucketed"]]},
            {"type": "figure", "figure": "tbl-layout", "title": "The directory tree", "caption": "Arrow widths follow the rows in each state; the CA folder is expanded."},
        ]},
        {"id": "buckets", "title": "CLUSTERED BY: bucket files inside each directory", "blocks": [
            {"type": "p", "text": "<code>CLUSTERED BY (empno) INTO n BUCKETS</code> sends each row to bucket pmod(murmur3(empno), n), Spark's own hash with seed 42. "
                                  "Rows with the same empno always land in the same bucket, in every directory and in every table bucketed the same way."},
            {"type": "figure", "figure": "tbl-buckets", "title": "Rows per bucket", "caption": "Change the number of buckets.",
             "controls": [{"k": "buckets", "label": "Buckets", "min": 1, "max": 16, "step": 1, "value": 8}]},
            {"type": "callout", "tone": "tip", "title": "Two payoffs (beyond the slides)",
             "text": f"Joins: two tables bucketed the same way on the join key join without a shuffle (see Spark joins). Lookups: WHERE empno = {S.LOOKUP_EMPNO} "
                     f"only needs bucket {_TL['query']['lookup_bucket']}, so Spark opens {_TL['read']['files_read']} of {_TL['read']['files_total']} files in the MA query."},
            {"type": "try", "label": f"Bucket pruning for empno {S.LOOKUP_EMPNO}", "params": {"state": "MA", "gender": "all", "salary": 0, "lookup": True}},
        ]},
        {"id": "parquet", "title": "Inside a Parquet file: row groups and min/max", "blocks": [
            {"type": "p", "text": "A Parquet file is split horizontally into row groups (parquet.block.size, 128 MB by default) and, inside each, vertically into one "
                                  "column chunk per column. The footer records each chunk's min, max and null count, so a reader can rule a row group out before decoding it."},
            {"type": "steps", "items": [
                "Read the footer of each file that survived partition and bucket pruning.",
                "For each row group, compare the filter with the column's min and max: salary > 90000 cannot match a group whose max is 88000.",
                "Decode only the surviving row groups, and only the columns the query uses.",
                "Apply the filter to the decoded rows: statistics skip groups, they do not select rows.",
            ]},
            {"type": "figure", "figure": "tbl-skipping", "title": "Sorting before the write makes statistics useful",
             "caption": "Share of the CA row groups read as the salary threshold rises. Change the row-group size.",
             "controls": [{"k": "rg_rows", "label": "Rows per row group", "min": 4, "max": 128, "step": 4, "value": 16}]},
            {"type": "try", "label": "Sort by salary before writing", "params": {"sorted": True, "state": "CA", "salary": 90000, "rg_rows": 16}},
        ]},
        {"id": "query", "title": "One query, layer by layer", "blocks": [
            {"type": "p", "text": f"<code>{_T0['query']['sql']}</code> is answered by reading as little as possible. Each layer prunes before the next one starts."},
            {"type": "figure", "figure": "tbl-funnel", "title": "Bytes still to read after each step", "caption": "Pick the state in the filter.",
             "controls": [{"k": "state", "label": "WHERE state =", "options": [["all", "all"]] + [[s, s] for s in S.TBL_STATES], "value": "CA"}]},
            {"type": "quiz", "q": f"Sorting by salary before the write changes which of these for the default query?",
             "options": ["The directories read", "The row groups read", "The rows returned"], "answer": 1,
             "why": f"Directories depend only on state and gender, and the answer never changes. Sorting narrows each row group's salary range, so the row groups read fall from "
                    f"{_T0['read']['row_groups_read']} to {_TS['read']['row_groups_read']}."},
        ]},
        {"id": "smallfiles", "title": "Why bucketed writes can make many small files", "blocks": [
            {"type": "p", "text": "The deck shows 8 bucket files per folder. In practice each <b>write task</b> writes its own file for every folder and bucket it has rows for, "
                                  "so files = tasks × folders × buckets in the worst case."},
            {"type": "figure", "figure": "tbl-files", "title": "Files written against write tasks", "caption": "Change the number of buckets.",
             "controls": [{"k": "buckets", "label": "Buckets", "min": 1, "max": 16, "step": 1, "value": 8}]},
            {"type": "callout", "tone": "info", "title": "DISTRIBUTE BY before writing",
             "text": "The DDL deck notes that DISTRIBUTE BY is useful before writing bucketed data. Distributing by the partition columns sends each folder's rows "
                     "to one task, so each folder gets at most one file per bucket."},
            {"type": "try", "label": "8 tasks, then DISTRIBUTE BY", "params": {"buckets": 8, "tasks": 8, "distribute": True}},
        ]},
        {"id": "placement", "title": "DISTRIBUTE BY, SORT BY, CLUSTER BY, ORDER BY", "blocks": [
            {"type": "compare", "items": [
                {"title": "DISTRIBUTE BY state", "points": ["shuffle: pmod(hash(state), n)", "all rows of a state in one partition", "no order inside"]},
                {"title": "SORT BY salary", "points": ["no shuffle", "each existing partition sorted", "not a global order"]},
                {"title": "CLUSTER BY state / ORDER BY salary", "points": ["CLUSTER BY = DISTRIBUTE BY + SORT BY on the same column", "ORDER BY: range partitioning, then sort", "only ORDER BY gives a global order"]},
            ]},
            {"type": "figure", "figure": "tbl-placement", "title": "Where 24 rows land in 3 partitions", "caption": "Bars count each state's rows per output partition. Pick the clause.",
             "controls": [{"k": "clause", "label": "Clause", "options": _CLAUSES, "value": "distribute"}]},
            {"type": "quiz", "q": "Which clause guarantees that reading the output partitions in order gives rows sorted by salary?",
             "options": ["SORT BY salary", "DISTRIBUTE BY state SORT BY salary", "ORDER BY salary"], "answer": 2,
             "why": "SORT BY only sorts within each partition. ORDER BY range-partitions by salary first, so partition 0 holds the highest salaries, partition 1 the next range, and so on."},
            {"type": "try", "label": "ORDER BY versus SORT BY", "params": {"clause": "order", "demo_rows": 12, "out_parts": 3}},
        ]},
        {"id": "slides", "title": "Where the slides need care", "blocks": [
            {"type": "callout", "tone": "warn", "title": "Corrections to the DDL/DML deck",
             "text": "(1) p21 labels the bucketing step \"CLUSTER BY\"; the CREATE TABLE clause is CLUSTERED BY, while CLUSTER BY is the SELECT clause that "
                     "distributes and sorts. (2) The bucket files are shown as part-00000-bucket.parquet … part-00007-bucket.parquet, one per bucket; Spark names "
                     "them part-<task>-<job>_<bucket>.c000.snappy.parquet, and every write task can write one per bucket, so 8 buckets often means many more than 8 files. "
                     "(3) p17 places min/max \"for each page of chunk\" in the row-group metadata; the footer keeps them per column chunk; page-level statistics are in the page "
                     "headers and the optional page index. (4) \"Parquet is a read-only format\" means its files are written once and never modified in place: tables still accept INSERTs as new files."},
            {"type": "p", "text": "Byte counts use a plain-encoding model (no compression, approximate metadata), so they compare layouts rather than predict real file sizes."},
        ]},
    ],
    "takeaways": [
        "A table is a metastore entry plus files; DROP TABLE deletes the files only for managed tables.",
        "PARTITIONED BY makes one directory per value; filters on partition columns skip whole directories.",
        "CLUSTERED BY … INTO n BUCKETS hashes rows into n files per directory with Spark's Murmur3; equal keys always share a bucket.",
        "Parquet row groups carry min/max per column; sorting before the write makes those ranges narrow and skipping effective.",
        "Every write task writes its own file per directory and bucket; DISTRIBUTE BY first avoids a flood of small files.",
        "Only ORDER BY sorts globally; SORT BY sorts each partition, CLUSTER BY = DISTRIBUTE BY + SORT BY.",
    ],
    "glossary": [
        ["Managed table", "A table whose files Spark owns, under the warehouse directory."],
        ["External table", "A table created with LOCATION; Spark owns only its metadata."],
        ["Partition pruning", "Skipping whole directories whose partition values fail the filter."],
        ["Bucket", "One of n files per directory holding the rows with pmod(hash(key), n) equal to its number."],
        ["Row group", "A horizontal slice of a Parquet file with per-column statistics."],
        ["Range partitioning", "Splitting rows by value ranges (as ORDER BY does) instead of by hash."],
    ],
}, order=9)
