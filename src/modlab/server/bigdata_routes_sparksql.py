"""/api/bigdata/sparksql: Spark join strategies, Spark SQL's optimiser and table file layout.

Simulations of Spark's documented behaviour in plain Python (modlab.bigdata.spark_sql); nothing
here runs Spark. Every number the three views show is computed by these endpoints.
"""
from __future__ import annotations

from ..bigdata import spark_sql as S
from .api import choice, clean, endpoint, flag, num
from .api_bigdata import bp
from .cache import memo
from .catalog import section

_STATE_CHOICES = ["all"] + S.TBL_STATES


@memo(maxsize=128)
def _joins(**kw):
    return S.run_join(**kw)


@bp.get("/sparksql/joins")
@endpoint
def sparksql_joins():
    """Plan and run customers ⋈ orders with Spark's join selection, partition by partition."""
    out, cached = _joins(
        source=choice("source", "course", list(S.COURSE_JOIN_SOURCES) + ["synthetic"]),
        join=choice("join", "inner", list(S.JOIN_TYPES)), cond=choice("cond", "eq", ["eq", "band"]),
        hint=choice("hint", "none", list(S.HINTS)), layout=choice("layout", "plain", ["plain", "bucketed"]),
        threshold_mb=num("threshold", S.AUTO_BROADCAST_DEFAULT_MB, -1, 200),
        partitions=num("partitions", 8, 1, 400, integer=True), executors=num("executors", 4, 1, 64, integer=True),
        prefer_smj=flag("prefer_smj", True), aqe=flag("aqe", False),
        lrows_exp=num("lrows", 5, 1, 9), rrows_exp=num("rrows", 8, 1, 10),
        lbytes=num("lbytes", 100, 8, 2048, integer=True), rbytes=num("rbytes", 50, 8, 2048, integer=True),
        skew=num("skew", 0, 0, 2.5))
    return dict(out, _cached=cached)


@memo(maxsize=128)
def _sql(**kw):
    return S.run_sql(**kw)


@bp.get("/sparksql/query")
@endpoint
def sparksql_query():
    """Optimise a course query rule by rule, plan it physically and execute it."""
    out, cached = _sql(query=choice("query", "slide16", list(S.SQL_QUERIES)), k=num("k", 30, 0, 100, integer=True),
                       dno=int(choice("dno", "5", ["1", "4", "5"])), gender=choice("gender", "F", ["F", "M"]),
                       folding=flag("folding", True), pushdown=flag("pushdown", True), pruning=flag("pruning", True),
                       broadcast=flag("broadcast", True))
    return dict(out, _cached=cached)


@memo(maxsize=64)
def _tables(**kw):
    return S.table_layout(**kw)


@bp.get("/sparksql/tables")
@endpoint
def sparksql_tables():
    """Write a partitioned, bucketed Parquet table, query it with pruning, drop it, and place rows."""
    out, cached = _tables(
        rows=num("rows", 4000, 100, 20000, integer=True), seed=num("seed", 494, 0, 99999, integer=True),
        buckets=num("buckets", 4, 1, 16, integer=True), tasks=num("tasks", 2, 1, 8, integer=True),
        distribute=flag("distribute", False), sort_salary=flag("sorted", False), rg_rows=num("rg_rows", 16, 4, 256, integer=True),
        f_state=choice("state", "CA", _STATE_CHOICES), f_gender=choice("gender", "all", ["all", "M", "F"]),
        f_salary=num("salary", 90000, 0, 130000, integer=True), lookup=flag("lookup", False),
        select=choice("select", "two", ["two", "all"]), kind=choice("kind", "managed", ["managed", "external"]),
        clause=choice("clause", "distribute", ["none", "distribute", "distribute_sort", "sort", "cluster", "order"]),
        demo_rows=num("demo_rows", 12, 4, 24, integer=True), out_parts=num("out_parts", 3, 1, 6, integer=True))
    return dict(out, _cached=cached)


@section("bigdata_sparksql")
def catalog():
    """Fixed facts the three views describe before a request: labels, defaults, course rows."""
    return clean({
        "joins": {
            "sources": {k: v["label"] for k, v in S.COURSE_JOIN_SOURCES.items()} | {"synthetic": "Synthetic scale-up"},
            "strategies": {k: v for k, v in S.STRATEGY.items()}, "order": S.STRATEGIES,
            "join_sql": S.JOIN_SQL, "hints": {k: v for k, v in S.HINTS.items() if v},
            "threshold_default_mb": S.AUTO_BROADCAST_DEFAULT_MB, "shuffle_partitions_default": 200,
            "broadcast_limit_bytes": S.BROADCAST_HARD_LIMIT, "max_partition_bytes": S.MAX_PARTITION_BYTES,
            "skew": {"factor": S.SKEW_FACTOR, "min_bytes": S.SKEW_MIN_BYTES, "advisory_bytes": S.ADVISORY_BYTES},
            "cost": {"net_bps": S.NET_BPS, "hash_rate": S.HASH_RATE, "sort_rate": S.SORT_RATE, "nl_rate": S.NL_RATE},
            "sample": {"left": S.SAMPLE_LEFT, "right": S.SAMPLE_RIGHT},
            "source_rows": {k: [len(v["left"]["rows"]), len(v["right"]["rows"])] for k, v in S.COURSE_JOIN_SOURCES.items()},
            "hash_check": {"value": 1, "hash": S.murmur3_int(1)},
        },
        "sql": {"queries": {k: {"label": v["label"], "slide": v["slide"]} for k, v in S.SQL_QUERIES.items()},
                "rules": [{"key": k, "name": n} for k, n, _ in S.SQL_RULES],
                "letter_grade": [{"test": f"marks {o} {b}", "grade": g} for o, b, g in S.LETTER_GRADE_RULES],
                "slide_first10": S.SLIDE_GRADES_FIRST10, "marks": len(S.MARKS), "employees": len(S.EMPLOYEE),
                "departments": [[d[0], d[1]] for d in S.DEPARTMENT]},
        "tables": {"states": S.TBL_STATES, "lookup_empno": S.LOOKUP_EMPNO, "data_cols": S.DATA_COLS,
                   "deck_rows": [list(r) for r in S.EMPLOYEE_M_ROWS], "pool_size": sum(len(n) for n in S.EMP30_NODES),
                   "parquet_block_default_bytes": 128 * S.MB},
    })
