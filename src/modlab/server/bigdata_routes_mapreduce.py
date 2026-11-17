"""/api/bigdata: HDFS blocks and replication, the MapReduce engine, and MapReduce joins."""
from __future__ import annotations

from ..bigdata import datasets as D
from ..bigdata import hdfs as H
from ..bigdata import mapreduce as M
from .api import choice, clean, endpoint, flag, num
from .api_bigdata import bp
from .cache import memo
from .catalog import section

BLOCK_SIZES = ["32", "64", "128", "256"]


@memo(maxsize=256)
def _hdfs(*args):
    return H.simulate(*args)


@memo(maxsize=16)
def _rack_loss(datanodes, racks, files, size_mb, block_mb):
    return H.rack_loss_table(datanodes, racks, files, size_mb, block_mb)


@bp.get("/hdfs")
@endpoint
def hdfs():
    n = num("datanodes", 4, 3, 12, integer=True)
    kill = num("kill", 0, 0, 12, integer=True)
    files, size_mb = num("files", 3, 1, 4, integer=True), num("size_mb", 384, 1, 4096)
    block_mb = float(choice("block_mb", "128", BLOCK_SIZES))
    racks = num("racks", 2, 1, 4, integer=True)
    args = (files, size_mb, block_mb, num("replication", 3, 1, 5, integer=True), n, racks,
            flag("rack_aware", True), kill if kill <= n else 0, flag("fail_rack", False),
            num("read_tb", 1, 0.001, 1000), num("mbps", 100, 10, 5000), num("readers", 1, 1, 10000, integer=True))
    out, cached = _hdfs(*args)
    loss, c2 = _rack_loss(n, racks, files, size_mb, block_mb)
    return dict(out, rack_loss=loss, kill_ignored=kill > n, _cached=cached and c2)


@memo(maxsize=256)
def _job(*args):
    return M.run_job(*args)


@bp.get("/mapreduce")
@endpoint
def mapreduce():
    out, cached = _job(choice("job", "avg", list(M.JOBS)), choice("combiner", "none", M.COMBINERS),
                       choice("partitioner", "range", M.PARTITIONERS), num("reducers", 2, 1, 6, integer=True),
                       flag("group_cmp", True), flag("natural_part", True), num("topn", 3, 1, 9, integer=True))
    return dict(out, _cached=cached)


@memo(maxsize=256)
def _join(*args):
    return M.run_join(*args)


@memo(maxsize=256)
def _scale(*args):
    return M.join_scale(*args)


@bp.get("/mr-joins")
@endpoint
def mr_joins():
    out, c1 = _join(choice("algo", "standard", M.JOIN_ALGOS), choice("join", "inner", M.JOIN_TYPES),
                    num("reducers", 3, 1, 4, integer=True), flag("extra", False), flag("group_cmp", True),
                    flag("natural_part", True), choice("partitioner", "hash", M.PARTITIONERS))
    sc, c2 = _scale(round(num("log_cust", 0, 0, 7), 2), round(num("log_orders", 0, 0, 9), 2),
                    num("nodes", 10, 1, 1000, integer=True), num("heap_mb", 1024, 64, 16384))
    extra, R = out["extra"], out["R"]
    compare = []
    for algo, label in (("standard", "Repartition"), ("improved", "Improved"), ("broadcast", "Broadcast")):
        rows = {}
        for j in M.JOIN_TYPES:
            r, _ = _join(algo, j, R, extra, True, True, out["partitioner"])
            rows[j] = r["check"]["rows"]
        r, _ = _join(algo, out["join"], R, extra, True, True, out["partitioner"])
        compare.append({"algo": algo, "label": label, "network_bytes": r["network"]["shuffled_bytes"] + r["network"]["broadcast_bytes"],
                        "peak_records": r["memory"]["peak_records"], "peak_bytes": r["memory"]["peak_bytes"], "rows_by_join": rows})
    return dict(out, scale=sc, compare=compare, _cached=c1 and c2)


@section("bigdata_mapreduce")
def catalog():
    """Dataset facts the three views describe: the slide data sizes, job descriptions, HDFS defaults."""
    exp = M.expected_avg()
    return clean({
        "hdfs": {"block_mb": D.HDFS_BLOCK_MB, "replication": D.HDFS_REPLICATION, "slide7_layout": D.HDFS_LAYOUT,
                 "slide7_files": D.HDFS_FILES, "slide7_copies": H.slide7_copies(), "block_sizes": [int(b) for b in BLOCK_SIZES]},
        "jobs": M.JOBS,
        "emp30": {"nodes": len(D.EMP30_NODES), "rows": sum(len(n) for n in D.EMP30_NODES),
                  "rows_per_node": [len(n) for n in D.EMP30_NODES], "departments": list(exp),
                  "expected_avg": {str(k): round(v, 2) for k, v in exp.items()}},
        "wordcount": {"splits": D.WORDCOUNT_SPLITS, "ranges": [f"{a}–{b}" for a, b in D.WORDCOUNT_RANGES],
                      "expected": M.expected_wordcount()},
        "topn": {"splits": [[list(r) for r in s] for s in D.TOPN_SPLITS]},
        "docs": M.DOCS,
        "joins": {"customers": [list(c) for c in D.MR_CUSTOMERS], "orders": [list(o) for o in D.MR_ORDERS],
                  "extra_customer": list(M.JOIN_EXTRA_CUSTOMER), "extra_order": list(M.JOIN_EXTRA_ORDER),
                  "joined_rows": len(M.expected_join("inner"))},
    })
