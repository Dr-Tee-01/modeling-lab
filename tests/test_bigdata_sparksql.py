"""Big Data Lab · Spark SQL: join strategies, the optimiser and table layout (modlab.bigdata.spark_sql)."""
import itertools
import time

import numpy as np
import pandas as pd
import pytest

from modlab.bigdata import spark_sql as S
from modlab.bigdata.datasets import MR_CUSTOMERS, MR_ORDERS, SPARK_CUSTOMERS, SPARK_ORDERS


# ---------------------------------------------------------------- Spark's hash
def test_murmur3_matches_spark():
    assert S.murmur3_int(1) == -559580957                      # SELECT hash(1)
    h = S.murmur3_str("Spark")                                  # SELECT hash('Spark', array(123), 2), Spark docs
    h = S.murmur3_int(123, h)
    assert S.murmur3_int(2, h) == -1321691492
    keys = np.arange(-50, 5000)
    assert S.murmur3_np(keys).tolist() == [S.murmur3_int(int(k)) for k in keys]
    assert set(S.partition_of(keys, 7).tolist()) <= set(range(7))


# ---------------------------------------------------------------- joins: every strategy gives the right rows
def _pandas_pairs(left, right, how):
    """(cust_id, order_id) pairs of the join computed by pandas, independent of the simulator."""
    l = pd.DataFrame({"cust_id": [r[0] for r in left], "c": 1})
    r = pd.DataFrame({"order_id": [o[0] for o in right], "cust_id": [o[1] for o in right]})
    m = l.merge(r, on="cust_id", how={"inner": "inner", "left": "left", "full": "outer"}[how])
    return sorted((int(a), -1 if pd.isna(b) else int(b)) for a, b in zip(m["cust_id"], m["order_id"]))


@pytest.mark.parametrize("source", ["course", "mr"])
@pytest.mark.parametrize("join", ["inner", "left", "full"])
def test_every_strategy_matches_pandas(source, join):
    src = S.COURSE_JOIN_SOURCES[source]
    left, right = src["left"]["rows"], src["right"]["rows"]
    lk = np.array([r[0] for r in left])
    rk = np.array([r[1] for r in right])
    want = _pandas_pairs(left, right, join)
    sizes = {"left_bytes": 1, "right_bytes": 2}
    ran = 0
    for strat in S.STRATEGIES:
        ok, _ = S._feasible(strat, join, "eq")
        if not ok:
            continue
        builds = S._BUILD.get(strat, {}).get(join, (None,)) or (None,)
        for b in builds:                              # every allowed build side, not only the planner's choice
            for parts in (1, 3, 8):
                out = S.execute_join(strat, b, join, "eq", lk, rk, parts, 4)["pairs"]
                got = sorted((int(lk[a]) if a is not None else int(rk[c]), int(right[c][0]) if c is not None else -1) for a, c in out)
                assert got == want, (strat, b, parts)
                ran += 1
    assert ran >= 9
    # the full run checks every feasible strategy against its own all-pairs reference too
    r = S.run_join(source=source, join=join, threshold_mb=-1)
    assert all(v["same_as_reference"] for v in r["validation"] if v["feasible"])
    assert r["result"]["total"] == len(want)


def test_course_join_rows():
    r = S.run_join(source="course", join="left", threshold_mb=-1)
    assert r["result"]["total"] == len(SPARK_ORDERS) + 2
    assert r["result"]["unmatched_left_ids"] == [116, 117]          # customers with no order in the RDD deck
    inner = S.run_join(source="mr")
    assert inner["result"]["total"] == len(MR_ORDERS)


@pytest.mark.parametrize("join", ["inner", "left", "full"])
def test_band_join_strategies_agree(join):
    r = S.run_join(source="course", join=join, cond="band", threshold_mb=10)
    ref = [(c[0], o[0]) for c in SPARK_CUSTOMERS for o in SPARK_ORDERS if abs(c[0] - o[1]) <= 1]
    assert r["result"]["total"] >= len(ref)
    feasible = [v for v in r["validation"] if v["feasible"]]
    assert {v["strategy"] for v in feasible} == ({"bnlj", "cartesian"} if join == "inner" else {"bnlj"})
    assert all(v["same_as_reference"] for v in feasible)
    if join == "inner":
        assert r["result"]["total"] == len(ref)
        assert r["metrics"]["nl_comparisons"] == len(SPARK_CUSTOMERS) * len(SPARK_ORDERS)


def test_synthetic_sample_and_skew():
    for skew in (0, 1.5):
        r = S.run_join(source="synthetic", skew=skew, threshold_mb=-1, partitions=64, join="full")
        assert all(v["same_as_reference"] for v in r["validation"] if v["feasible"])
        assert r["result"]["unmatched_right"] > 0                      # the 1% orphan orders
    flat = S.run_join(source="synthetic", skew=0, threshold_mb=-1, partitions=200, lrows_exp=6, rrows_exp=9)
    skewed = S.run_join(source="synthetic", skew=1.2, threshold_mb=-1, partitions=200, lrows_exp=6, rrows_exp=9)
    assert skewed["metrics"]["straggler_ratio"] > 10 * flat["metrics"]["straggler_ratio"]
    aqe = S.run_join(source="synthetic", skew=1.2, threshold_mb=-1, partitions=200, lrows_exp=6, rrows_exp=9, aqe=True)
    assert aqe["metrics"]["skew_splits"] > 0 and aqe["metrics"]["extra_read_bytes"] > 0
    assert aqe["metrics"]["straggler_ratio"] < skewed["metrics"]["straggler_ratio"]
    assert aqe["metrics"]["peak_mem_bytes"] < skewed["metrics"]["peak_mem_bytes"]


# ---------------------------------------------------------------- joins: the planner
def _plan(join="inner", cond="eq", hint="none", thr=10, lb=1 * S.MB, rb=5 * S.GB, prefer_smj=True, bucketed=False, partitions=200):
    return S.plan_join(join=join, cond=cond, hint=hint, threshold_mb=thr, partitions=partitions, prefer_smj=prefer_smj,
                       bucketed=bucketed, sizes={"left_bytes": lb, "right_bytes": rb})


def test_broadcast_threshold_boundary():
    thr = 10 * S.MB
    assert _plan(lb=thr)["strategy"] == "bhj"                         # "at most" the threshold
    assert _plan(lb=thr + 1)["strategy"] == "smj"
    assert _plan(lb=thr, thr=-1)["strategy"] == "smj"                  # −1 disables auto-broadcast
    p = _plan(lb=3 * S.MB, rb=2 * S.MB)
    assert p["strategy"] == "bhj" and p["build"] == "right"            # both small: the smaller is built
    # the same boundary through the full run: 100,000 customers × 104 B is under 10 MiB, × 105 B is over
    assert S.run_join(source="synthetic", lrows_exp=5, lbytes=104)["plan"]["strategy"] == "bhj"
    assert S.run_join(source="synthetic", lrows_exp=5, lbytes=105)["plan"]["strategy"] == "smj"


def test_outer_joins_restrict_the_build_side():
    assert _plan(join="left")["strategy"] == "smj"                     # small customers, but LEFT can only build right
    p = _plan(join="left", lb=5 * S.GB, rb=1 * S.MB)
    assert p["strategy"] == "bhj" and p["build"] == "right"
    assert _plan(join="full", lb=1, rb=1)["strategy"] == "smj"        # never broadcast hash for FULL OUTER
    assert _plan(join="full", hint="broadcast")["strategy"] == "smj"  # the hint is ignored
    assert _plan(join="full", hint="shuffle_hash")["strategy"] == "shj"


def test_hints_preferences_and_buckets():
    assert _plan(hint="merge")["strategy"] == "smj"
    p = _plan(hint="broadcast", lb=50 * S.GB, rb=60 * S.GB)
    assert p["strategy"] == "bhj" and any("8 GB" in c["detail"] for c in p["checks"])
    assert _plan(hint="shuffle_replicate_nl")["strategy"] == "cartesian"
    # shuffle hash only when preferSortMergeJoin is off, the side fits a partition and is 3x smaller
    assert _plan(lb=20 * S.MB, prefer_smj=True)["strategy"] == "smj"
    p = _plan(lb=20 * S.MB, prefer_smj=False)
    assert p["strategy"] == "shj" and p["build"] == "left"
    assert _plan(lb=20 * S.MB, rb=40 * S.MB, prefer_smj=False)["strategy"] == "smj"
    p = _plan(lb=20 * S.MB, bucketed=True)
    assert p["strategy"] == "copart" and not p["exchange"]
    r = S.run_join(source="course", layout="bucketed", threshold_mb=-1)
    assert r["metrics"]["shuffle_bytes"] == 0 and r["plan"]["tree"]["label"] == "SortMergeJoin"


def test_non_equi_planning():
    assert _plan(cond="band")["strategy"] == "bnlj"
    assert _plan(cond="band", thr=-1)["strategy"] == "cartesian"
    p = _plan(cond="band", thr=-1, join="left")
    assert p["strategy"] == "bnlj" and p["build"] == "left"           # last resort: the smaller side, whatever the join type
    small_right = _plan(cond="band", join="left", lb=5 * S.GB, rb=1 * S.MB)
    assert small_right["build"] == "right" and small_right["checks"][-1]["rule"] != "Fallback"   # by size: right only
    assert _plan(cond="band", join="left")["checks"][-1]["rule"] == "Fallback"   # small customers cannot be broadcast by size
    assert _plan(cond="band", join="full", lb=1, rb=2)["checks"][-1]["rule"] == "Fallback"


def test_costs_follow_the_definitions():
    r = S.run_join(source="synthetic", lrows_exp=5, lbytes=100, rrows_exp=8, rbytes=50, executors=6)
    c = {x["strategy"]: x for x in r["compare"]}
    L, R = r["tables"]["left"]["bytes"], r["tables"]["right"]["bytes"]
    assert c["bhj"]["broadcast_bytes"] == pytest.approx(L * 6)
    assert c["smj"]["shuffle_bytes"] == pytest.approx(L + R)
    assert c["copart"]["shuffle_bytes"] == 0
    assert c["bnlj"]["nl_comparisons"] == pytest.approx(r["tables"]["left"]["rows"] * r["tables"]["right"]["rows"])


# ---------------------------------------------------------------- SQL: the optimiser preserves results
@pytest.mark.parametrize("query", list(S.SQL_QUERIES))
def test_every_rule_combination_gives_the_same_result(query):
    base = S.run_sql(query=query, folding=False, pushdown=False, pruning=False)
    for f, p, c in itertools.product([False, True], repeat=3):
        r = S.run_sql(query=query, folding=f, pushdown=p, pruning=c)
        assert r["result"]["same_as_unoptimised"]
        if query == "order":
            assert r["result"]["rows"] == base["result"]["rows"]
        else:
            assert sorted(map(str, r["result"]["rows"])) == sorted(map(str, base["result"]["rows"]))
    assert all(v["same_result"] for v in base["variants"])


def test_course_query_results():
    rows = lambda q, **kw: S.run_sql(query=q, **kw)["result"]["rows"]
    assert sorted(r[0] for r in rows("filter", k=30)) == [102, 104, 105, 106]           # salary > 30000
    assert [r[0] for r in rows("filter_and", k=40, dno=5)] == [102]                    # salary >= 40000 AND dno = 5
    assert sorted(rows("slide16")) == [[4, "Administration", 2], [5, "Research", 1]]
    assert sorted(rows("join_agg")) == [["Administration", 93000], ["Headquarters", 55000], ["Research", 133000]]
    agg = {r[0]: (r[1], r[2]) for r in rows("agg")}
    assert agg == {5: (33250.0, 40000), 4: (31000.0, 43000), 1: (55000.0, 55000)}
    order = rows("order")
    keys = [(r[6], -r[4]) for r in order]
    assert keys == sorted(keys)


def test_each_rule_saves_what_it_should():
    on = S.run_sql(query="slide16")
    off = S.run_sql(query="slide16", folding=False, pushdown=False, pruning=False)
    assert off["metrics"]["cells_read"] == 8 * 7 + 3 * 4 and on["metrics"]["cells_read"] == 8 * 3 + 3 * 2
    assert off["metrics"]["rows_into_join"] == 11 and on["metrics"]["rows_into_join"] == 3 + 3
    pushed = [s for s in on["stages"] if s["rule"] == "pushdown"][0]
    assert any("moved below the Join" in line for line in pushed["log"])
    # folding: 30 * 1000 is folded once instead of being evaluated for each of the 8 rows
    f = S.run_sql(query="filter", k=30)
    nf = S.run_sql(query="filter", k=30, folding=False)
    assert nf["metrics"]["evals"] - f["metrics"]["evals"] == 8
    # folding makes the filter pushable; above the max salary the row group is skipped entirely
    assert S.run_sql(query="filter", k=60)["metrics"]["rows_read"] == 0
    assert S.run_sql(query="filter", k=60, folding=False)["metrics"]["rows_read"] == 8
    phys = S.run_sql(query="filter", k=30)["stages"][-1]["tree"]
    scan = phys
    while scan.get("children"):
        scan = scan["children"][0]
    assert "GreaterThan(salary,30000)" in scan["detail"]


def test_physical_plans():
    def labels(t):
        return [t["label"]] + [x for c in t.get("children", []) for x in labels(c)]
    b = S.run_sql(query="slide16")["stages"][-1]["tree"]
    assert "BroadcastHashJoin" in labels(b) and "BroadcastExchange" in labels(b)
    m = S.run_sql(query="slide16", broadcast=False)["stages"][-1]["tree"]
    assert "SortMergeJoin" in labels(m) and labels(m).count("Sort") == 2
    u = S.run_sql(query="udf")["stages"][-1]["tree"]
    assert "BatchEvalPython" in labels(u)


def test_letter_grade_as_written():
    rep = S.letter_grade_report()
    assert rep["first10"] == ["DD", "F", "DD", "DE", "BB", "DD", "BC", "BB", "BB", "DD"]
    assert rep["matches_slide"]
    assert S.letter_grade(32) == "DE"                                   # the slide's print(letterGrade(32))
    assert S.letter_grade(36) == S.letter_grade(44) == "DD"             # the duplicated branch
    assert rep["duplicate"]["branches"] == ["marks > 40", "marks > 35"]
    assert sum(rep["counts"].values()) == len(S.MARKS) == 19
    r = S.run_sql(query="udf")
    assert [row[2] for row in r["result"]["rows"][:10]] == rep["first10"]


def test_read_sources_infer_types():
    src = S.read_sources()
    assert src["no_header"]["cols"][:2] == ["_c0", "_c1"] and set(src["no_header"]["types"]) == {"string"}
    assert src["infer"]["types"] == ["integer", "string", "integer", "integer", "string", "string"]
    assert src["json"]["types"] == ["string", "long"]


# ---------------------------------------------------------------- tables: pruning gives the same answer with fewer bytes
@pytest.mark.parametrize("kw", [{}, {"sort_salary": True}, {"f_state": "all", "f_salary": 60000}, {"f_state": "NY", "f_gender": "F"},
                                {"lookup": True, "f_state": "MA", "f_salary": 0}, {"buckets": 1, "rg_rows": 256}, {"select": "all", "rows": 20000}])
def test_pruning_is_correct_and_reads_less(kw):
    r = S.table_layout(**kw)
    rd, b = r["read"], r["read"]["bytes"]
    assert rd["same_as_full_scan"]
    assert b["total"] >= b["partitions"] >= b["buckets"] >= b["row_groups"] >= b["columns"]
    assert rd["rows"]["result"] <= rd["rows"]["row_groups"] <= rd["rows"]["buckets"] <= rd["rows"]["partitions"] <= rd["rows"]["total"]
    if kw.get("f_state", "CA") != "all":
        assert b["partitions"] < b["total"] and rd["dirs_read"] < rd["dirs_total"]


def test_layout_details():
    r = S.table_layout()
    emp = S.synthetic_employees(4000)
    assert [tuple(row[:6]) for row in r["sample"][:3]] == [tuple(x) for x in S.EMPLOYEE_M_ROWS]   # the deck's rows first
    assert all(row[7] == S.murmur3_int(row[0]) % 4 for row in r["sample"])
    assert sum(r["write"]["bucket_rows"]) == emp["n"]
    files = S.write_table(emp, buckets=4, tasks=2, distribute=False, sort_salary=False, rg_rows=16)
    assert sum(len(f["rows"]) for f in files) == emp["n"]
    for f in files:
        assert all(emp["state"][i] == f["state"] and emp["gender"][i] == f["gender"] for i in f["rows"])
        assert all(S.murmur3_int(int(emp["empno"][i])) % 4 == f["bucket"] for i in f["rows"])
    # sorting by salary before the write lets min/max skip more row groups; DISTRIBUTE BY cuts the file count
    assert S.table_layout(sort_salary=True)["read"]["row_groups_read"] < r["read"]["row_groups_read"]
    many = S.table_layout(tasks=8, buckets=8)["write"]
    dist = S.table_layout(tasks=8, buckets=8, distribute=True)["write"]
    assert dist["files"] == dist["ideal_files"] < many["files"]
    lk = S.table_layout(lookup=True, f_state="MA", f_salary=0)
    assert lk["read"]["files_read"] < S.table_layout(f_state="MA", f_salary=0)["read"]["files_read"]
    assert lk["read"]["result_rows"] == 1


def test_drop_managed_and_external():
    m = S.table_layout(kind="managed")["drop"]
    e = S.table_layout(kind="external")["drop"]
    assert m["after"]["metastore"] is None and m["after"]["fs"] is None
    assert e["after"]["metastore"] is None and e["after"]["fs"] == e["before"]["fs"]
    assert e["before"]["metastore"]["type"] == "EXTERNAL"


def test_distribute_sort_cluster_order():
    emp = S.synthetic_employees(400)
    p = lambda c: S.place_rows(emp, c, 24, 3)
    assert p("distribute")["checks"]["states_split"] == 0
    assert p("cluster")["checks"]["states_split"] == 0
    assert p("distribute_sort")["checks"]["sorted_within"]
    assert p("sort")["checks"]["sorted_within"] and not p("sort")["checks"]["globally_sorted"]
    assert p("order")["checks"]["globally_sorted"]
    for c in ("none", "distribute", "distribute_sort", "sort", "cluster", "order"):
        out = p(c)
        assert sorted(r[0] for part in out["after"] for r in part) == sorted(r[0] for part in out["before"] for r in part)


# ---------------------------------------------------------------- API
def test_api_endpoints(client):
    for url, q in [("/api/bigdata/sparksql/joins", {}), ("/api/bigdata/sparksql/joins", {"source": "synthetic", "skew": 1.2, "aqe": "true", "threshold": -1}),
                   ("/api/bigdata/sparksql/query", {"query": "slide16"}), ("/api/bigdata/sparksql/query", {"query": "udf"}),
                   ("/api/bigdata/sparksql/tables", {}), ("/api/bigdata/sparksql/tables", {"clause": "order", "kind": "external"})]:
        t0 = time.perf_counter()
        r = client.get(url, query_string=q)
        assert r.status_code == 200, r.get_json()
        assert time.perf_counter() - t0 < 1.5
        assert "compute_ms" in r.get_json()["meta"]
    again = client.get("/api/bigdata/sparksql/query", query_string={"query": "slide16"}).get_json()
    assert again["meta"]["cached"] is True
    for url, q in [("/api/bigdata/sparksql/joins", {"join": "cross"}), ("/api/bigdata/sparksql/joins", {"partitions": 0}),
                   ("/api/bigdata/sparksql/query", {"query": "drop"}), ("/api/bigdata/sparksql/tables", {"buckets": 99})]:
        r = client.get(url, query_string=q)
        assert r.status_code == 400 and r.get_json()["error"]["code"] in ("invalid_parameter", "out_of_range")
    cat = client.get("/api/catalog").get_json()["bigdata_sparksql"]
    assert cat["joins"]["threshold_default_mb"] == 10 and cat["joins"]["hash_check"]["hash"] == -559580957
    assert cat["sql"]["slide_first10"][:2] == ["DD", "F"] and cat["tables"]["lookup_empno"] == 10084


def test_explainers_registered(client):
    for key in ("spark-joins", "sql", "tables"):
        r = client.get(f"/api/explain/bigdata/{key}")
        assert r.status_code == 200
        page = r.get_json()
        figs = [b for s in page["sections"] for b in s["blocks"] if b["type"] == "figure"]
        assert 6 <= len(page["sections"]) <= 9 and len(figs) >= 5
        assert sum(1 for f in figs if f.get("controls")) >= 2
        assert all(f["data"].get("notes") for f in figs)
