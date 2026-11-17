"""Big Data Lab: RPC semantics, the mini RDD engine and Spark stage cutting reproduce the slides."""
import pytest

from modlab.bigdata import rpc as R
from modlab.bigdata import spark_core as S


def as_dict(rows):
    return {k: v for k, v in rows}


def final_records(r):
    return [x for p in r["steps"][-1]["partitions"] for x in p["records"]]


# --------------------------------------------------------------------------- RPC
def test_rpc_is_deterministic_for_a_seed():
    a, b = R.simulate(seed=11), R.simulate(seed=11)
    assert a["timeline"] == b["timeline"] and a["final_balance"] == b["final_balance"]


def test_perfect_network_debits_once_per_call():
    for pol in R.POLICIES:
        r = R.simulate(loss=0, dup=0, policy=pol, detail=False)
        assert r["executions"] == 10 and r["final_balance"] == r["expected_balance"] == 1000
        assert r["messages"] == 20 and r["double_debits"] == 0


def test_retry_without_key_double_debits_and_idempotency_fixes_it():
    retry = R.simulate(policy="retry", seed=4, detail=False)
    idem = R.simulate(policy="idempotent", seed=4, detail=False)
    assert retry["double_debits"] > 0 and retry["final_balance"] < retry["expected_balance"]
    assert idem["double_debits"] == 0 and idem["final_balance"] == idem["expected_balance"]
    assert idem["dedup_hits"] >= retry["double_debits"]          # every would-be double was answered from the table


def test_idempotency_never_double_debits_over_many_seeds():
    for s in range(40):
        r = R.simulate(policy="idempotent", loss=0.3, dup=0.2, seed=s, detail=False)
        assert r["double_debits"] == 0
        assert r["final_balance"] >= r["expected_balance"]


def test_no_retry_is_at_most_once_only_on_a_non_duplicating_network():
    for s in range(30):
        assert R.simulate(policy="none", loss=0.3, dup=0.0, seed=s, detail=False)["double_debits"] == 0
    assert any(R.simulate(policy="none", loss=0.0, dup=0.3, seed=s, detail=False)["double_debits"] for s in range(10))


def test_marshalling_bytes():
    m = R.marshal(1001, 100.0)
    assert m["int32"]["be"] == ["00", "00", "03", "E9"] and m["int32"]["le"] == ["E9", "03", "00", "00"]
    assert m["float64"]["be"] == ["40", "59", "00", "00", "00", "00", "00", "00"]
    assert m["int32"]["misread"] == -385679360


# --------------------------------------------------------------------------- RDD engine and the slides' numbers
def test_dept_sums_and_averages():
    r = S.rdd_lab("avg_reduce", 2, False, 1)
    sums = as_dict(eval(x) for x in r["steps"][3]["partitions"][0]["records"] + r["steps"][3]["partitions"][1]["records"])
    assert {k: v[0] for k, v in sums.items()} == {"4": 93000, "5": 133000, "1": 55000}
    for pipe in ("avg_reduce", "avg_group", "avg_aggregate"):
        out = as_dict(eval(x) for x in final_records(S.rdd_lab(pipe, 2, False, 1)))
        assert out == {"4": 31000.0, "5": 33250.0, "1": 55000.0}, pipe


def test_reduce_by_key_result_order_matches_slide_on_two_partitions():
    assert S.rdd_lab("avg_reduce", 2, False, 1)["result"] == "[('4', 31000.0), ('5', 33250.0), ('1', 55000.0)]"


def test_stats_variant():
    r = S.rdd_lab("avg_aggregate", 2, False, 1, True)
    agg = {x.split(",")[0].strip("('"): x for p in r["steps"][3]["partitions"] for x in p["records"]}
    assert "Stats(sum=133000.0, count=4, min=25000, max=40000)" in agg["5"]
    assert "Stats(sum=93000.0, count=3, min=25000, max=43000)" in agg["4"]
    assert r["correct"]


@pytest.mark.parametrize("parts,expected", [(1, 3), (2, 3), (3, 4), (4, 5)])
def test_shuffle_counts(parts, expected):
    assert S.rdd_lab("avg_group", parts, False, 1)["shuffled_records"] == 8
    assert S.rdd_lab("avg_reduce", parts, False, 1)["shuffled_records"] == expected
    assert S.rdd_lab("avg_aggregate", parts, False, 1)["shuffled_records"] == expected


def test_join_state_totals():
    r = S.rdd_lab("join", 2, False, 1)
    assert as_dict(eval(x) for x in final_records(r)) == {"Gujarat": 3302, "Texas": 2223, "Maharashtra": 2188,
                                                           "Scotland": 2333, "Washington": 2064}
    assert "Karnataka" not in r["result"] and "'117'" not in str(r["steps"][4]["partitions"])
    join_shuffles = r["steps"][4]["shuffles"]
    assert sum(sh["records"] for sh in join_shuffles) == 7 + 20


def test_regions_and_map_partitions():
    assert as_dict(eval(x) for x in final_records(S.rdd_lab("regions", 3, False, 1))) == {"Northeast": 4, "West": 3, "South": 2, "Midwest": 3}
    r = S.rdd_lab("mappartitions", 3, False, 1)
    assert eval(r["result"]) == [10, 200, 30, 40, 500, 60, 70, 800, 90]
    assert eval(S.rdd_lab("mappartitions", 2, False, 1)["result"]) == [10, 200, 30, 400, 50, 600, 70, 800, 90]


def test_broadcast_and_accumulator():
    assert eval(S.rdd_lab("broadcast", 2, False, 1)["result"]) == [3, 4, 5, 6]
    one = S.rdd_lab("accumulator", 2, False, 1)
    assert eval(one["result"]) == [1, 4, 9, 16] and one["counter"] == 4
    assert S.rdd_lab("accumulator", 2, False, 2)["counter"] == 8          # recomputed without persist: double count
    assert S.rdd_lab("accumulator", 2, True, 2)["counter"] == 4


def test_laziness_and_persist():
    lazy = S.rdd_lab("avg_reduce", 2, False, 0)
    assert lazy["lazy"] and lazy["partitions_computed"] == 0 and all(st["partitions"] is None for st in lazy["steps"])
    no = S.rdd_lab("avg_reduce", 2, False, 2)
    yes = S.rdd_lab("avg_reduce", 2, True, 2)
    assert no["recomputations"] > 0 and yes["recomputations"] == 0
    assert no["jobs"][1]["stages_skipped"] == 1                     # the shuffle files are reused by the second job


# --------------------------------------------------------------------------- stages
def test_quiz_has_four_stages():
    r = S.dag_lab("quiz", 3)
    assert r["n_stages"] == 4 and r["tasks_total"] == 12
    assert [s["kind"] for s in r["stages"]] == ["shuffle_map", "shuffle_map", "shuffle_map", "result"]
    assert r["stages"][3]["rdds"][0] == "join"
    assert S.dag_lab("quiz", 3, copart=True)["n_stages"] == 3


def test_paper_figure_skips_cached_b():
    r = S.dag_lab("paper", 3, cache=True, actions=2, reuse=False)
    assert [s["rdds"] for s in r["stages"]] == [["A"], ["C", "D", "E", "F"], ["B", "G"]]
    second = r["jobs"][1]["stages"]
    assert [s["id"] for s in second if s["skipped"]] == [1]          # "B is already in RAM, so we run stage 2 and then 3"


def test_copartitioned_join_is_narrow():
    r = S.dag_lab("join", 3, copart=True)
    joined = next(x for x in r["rdds"] if x["name"] == "joined")
    assert [d["kind"] for d in joined["deps"]] == ["narrow", "narrow"]
    plain = S.dag_lab("join", 3)
    assert [d["kind"] for d in next(x for x in plain["rdds"] if x["name"] == "joined")["deps"]] == ["shuffle", "shuffle"]


def test_lost_partition_recomputes_only_its_lineage():
    r = S.dag_lab("quiz", 3, lose=1)
    rec = r["recovery"]["recomputed"]
    assert {x["p"] for x in rec} == {1} and len(rec) == 8               # one partition of each of the 8 RDDs
    logs = S.dag_lab("logmine", 3, cache=True, actions=3, lose=0)
    assert {x["rdd"] for x in logs["recovery"]["recomputed"]} == {"lines", "errors", "php", "times"}
    assert {x["p"] for x in logs["recovery"]["recomputed"]} == {0}
    assert logs["recovery"]["lost_cache"] == [{"rdd": "errors", "p": 0}]          # not restored from a replica


def test_log_mining_cache_saves_work():
    assert S.dag_lab("logmine", 3, cache=True, actions=3)["partitions_computed"] < S.dag_lab("logmine", 3, actions=3)["partitions_computed"]


def test_repartition_vs_coalesce_and_hash():
    rep = S.dag_lab("resize", 3, target=2, how="repartition")
    coa = S.dag_lab("resize", 3, target=2, how="coalesce")
    assert rep["n_stages"] == 2 and rep["shuffle_records"] == 12
    assert coa["n_stages"] == 1 and coa["shuffle_records"] == 0 and coa["sizes"] == [4, 8]
    assert S.dag_lab("resize", 3, target=5, how="coalesce")["out_partitions"] == 3      # coalesce cannot grow
    h = S.dag_lab("hash", 3, target=4)
    assert {p["key"]: p["partition"] for p in h["placement"]} == {k: S.java_hash(k) % 4 for k in ("Northeast", "West", "South", "Midwest")}
    assert S.java_hash("Northeast") == -1847737758


def test_iterative_cost_contradicts_100x():
    it = S.iterative_cost(10, 100)
    assert it["mapreduce"][9] == 1100 and it["spark"][9] == 119
    assert 9 < it["speedup_10"] < 10 and it["per_iter_ratio"] == 110


# --------------------------------------------------------------------------- API
def test_endpoints(client):
    r = client.get("/api/bigdata/rpc", query_string={"policy": "idempotent"}).get_json()
    assert r["final_balance"] == r["expected_balance"] and set(r["compare"]) == set(R.POLICIES) and r["sweep"]["loss"][0] == 0
    d = client.get("/api/bigdata/rdd", query_string={"pipe": "avg_group", "parts": 2, "step": 4}).get_json()
    assert d["shuffled_records"] == 8 and d["step"] == 4 and d["correct"]
    g = client.get("/api/bigdata/dag", query_string={"pipe": "quiz", "stage": 8}).get_json()
    assert g["n_stages"] == 4 and g["stage"] == 4 and g["iter"]["speedup_10"] < 10
    assert client.get("/api/bigdata/rpc", query_string={"policy": "maybe"}).status_code == 400
    assert client.get("/api/bigdata/rdd", query_string={"parts": 9}).status_code == 400
    cat = client.get("/api/catalog").get_json()["bigdata_spark"]
    assert cat["dag"]["executors"] == 3 and len(cat["rdd"]["pipes"]) == 8
