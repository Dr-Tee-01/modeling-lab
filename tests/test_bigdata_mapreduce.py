"""Big Data Lab: HDFS, the MapReduce engine and MR joins reproduce the numbers on the IE494 slides."""
import pytest

from modlab.bigdata import hdfs as H
from modlab.bigdata import mapreduce as M


# ------------------------------------------------------------------ HDFS
def test_spark_p4_blocks_and_placement():
    s = H.simulate(files=1, size_mb=500, block_mb=128, replication=3, datanodes=3, racks=1)
    assert [b["mb"] for b in s["blocks"]] == [128, 128, 128, 116]
    assert s["storage"]["raw_mb"] == 1500
    assert s["blocks"][0]["replicas"] == ["DN1", "DN2", "DN3"]          # Spark deck p4: Block 0 -> DN1, DN2, DN3
    assert s["blocks"][1]["replicas"] == ["DN2", "DN3", "DN1"]          # Block 1 -> DN2, DN3, DN1


def test_slide7_layout_replicas_distinct_and_rack_aware():
    s = H.simulate()                                                     # 3 files x 384 MB = b1..b9
    assert s["n_blocks"] == 9 and [b["id"] for b in s["blocks"]] == [f"b{i}" for i in range(1, 10)]
    for b in s["blocks"]:
        assert len(set(b["replicas"])) == 3 and len(b["racks"]) == 2
    assert sum(len(n["blocks"]) for n in s["nodes"]["place"]) == 27


def test_kill_datanode_rereplicates():
    s = H.simulate(kill=1)
    f = s["failure"]
    assert s["dead"] == ["DN1"] and f["lost"] == 0 and f["under"] > 0
    assert s["rereplication"]["copies"] == f["under"]                   # one copy per missing replica
    assert all(b["status_heal"] == "ok" for b in s["blocks"])
    assert s["rereplication"]["mb"] == 128 * f["under"]


def test_rack_failure_loses_blocks_only_without_rack_awareness():
    naive = H.simulate(datanodes=6, racks=2, rack_aware=False, kill=1, fail_rack=True)
    aware = H.simulate(datanodes=6, racks=2, rack_aware=True, kill=1, fail_rack=True)
    assert naive["failure"]["lost"] > 0 and aware["failure"]["lost"] == 0
    t = H.rack_loss_table()
    assert all(v == 0 for v in t["aware"][1:])


def test_read_time_l1_slide12_corrected():
    r = H.read_time(1, 100, 1)
    assert r["seconds_one"] == pytest.approx(2 ** 40 / (100 * 2 ** 20))
    assert r["hours_one"] == pytest.approx(2.9127, abs=1e-3)
    assert r["slide_formula_value"] == pytest.approx(2 ** 40 / (100 * 3600))    # what the slide literally computes
    assert r["slide_formula_value"] / r["hours_one"] == pytest.approx(2 ** 20)
    assert H.read_time(100, 100, 1)["days_one"] == pytest.approx(12.14, abs=0.01)
    assert H.read_time(100, 100, 1000)["hours"] == pytest.approx(0.2913, abs=1e-3)


# ------------------------------------------------------------------ MapReduce
def test_group_by_average_matches_slides():
    r = M.run_job("avg", "none", "range", 2)
    got = dict(r["output"])
    assert got == {"1": "84683.5", "3": "92111.2", "4": "95660", "5": "58741.64"}
    assert r["check"]["all_ok"]
    assert r["load"]["reducer_records"] == [8, 22]                       # MR deck 1 p6: {1, 3, 4} | {5}
    assert r["load"]["hot_key"] == "5" and r["load"]["hot_key_records"] == 22


def test_naive_avg_combiner_bug_and_sum_count_fix():
    bug = M.run_job("avg", "reducer", "range", 2)
    assert not bug["check"]["all_ok"]
    got = dict(bug["output"])
    assert got["3"] == "94028.17" and got["5"] == "58840.43"            # average of per-node averages
    fix = M.run_job("avg", "proper", "range", 2)
    assert fix["check"]["all_ok"]
    assert fix["shuffle"]["records"] == 8 and fix["shuffle"]["records_no_combiner"] == 30
    assert fix["shuffle"]["saved_bytes"] > 0


def test_wordcount_and_combiner_lists():
    r = M.run_job("wordcount", "none", "range", 3)
    assert dict(r["output"]) == {"Hadoop": "3", "Map": "7", "Key": "2", "Reduce": "2", "Value": "2"}
    assert r["partitions"] == ["A–I", "J–Q", "R–Z"]
    assert r["shuffle"]["records"] == 16
    c = M.run_job("wordcount", "proper", "range", 3)
    groups = {g["key"]: g["values"] for red in c["stages"]["shuffle"] for g in red}
    assert groups["Hadoop"] == ["2", "1"] and groups["Map"] == ["4", "3"]   # MR deck 1 p30
    assert c["shuffle"]["records"] == 8 and c["check"]["all_ok"]


def test_inverted_index():
    r = M.run_job("index", "proper", "range", 3)
    out = dict(r["output"])
    assert out["hdfs"] == "d1,d3" and out["big"] == "d1,d2"
    assert r["check"]["all_ok"] and r["shuffle"]["saved_records"] == 1


def test_topn_min_heap_single_reducer():
    r = M.run_job("topn", "proper", "hash", 1, topn=3)
    assert [v for _, v in r["output"]] == ["<C6,40000>", "<C3,31000>", "<C8,21000>"]
    assert r["shuffle"]["records"] == 6 and r["shuffle"]["records_no_combiner"] == 9
    assert M.expected_topn(3) == [("C6", 40000), ("C3", 31000), ("C8", 21000)]
    many = M.run_job("topn", "proper", "hash", 3, topn=3)
    assert many["load"]["idle_reducers"] == 2                           # NULL key -> one reducer


def test_secondary_sort_needs_grouping_comparator_and_natural_partition():
    ok = M.run_job("secsort", "none", "hash", 2, True, True)
    assert ok["check"]["all_ok"] and sum(ok["load"]["reduce_calls"]) == 4
    five = dict(ok["output"])["5"]
    vals = [int(x) for x in five.strip("[]").split(",")]
    assert vals == sorted(vals, reverse=True) and len(vals) == 22
    assert not M.run_job("secsort", "none", "hash", 2, False, True)["check"]["all_ok"]
    assert not M.run_job("secsort", "none", "hash", 2, True, False)["check"]["all_ok"]


def test_hash_partitioner_is_java_compatible():
    assert M.java_hash("a") == 97 and M.java_hash("ab") == 97 * 31 + 98
    assert M.java_hash("Aa") == M.java_hash("BB")                        # the classic Java collision
    assert M.hash_partition(5, 2) == 1 and M.hash_partition(4, 2) == 0


# ------------------------------------------------------------------ joins
@pytest.mark.parametrize("algo", M.JOIN_ALGOS)
def test_joins_produce_the_nine_rows(algo):
    r = M.run_join(algo, "inner", 3)
    assert r["check"]["ok"] and len(r["output"]) == 9
    assert ["111", "India", "Gujarat", "1", "550"] in r["output"]


def test_join_memory_and_shuffle():
    std, imp, bc = (M.run_join(a) for a in M.JOIN_ALGOS)
    assert std["network"]["shuffled_records"] == 12 and bc["network"]["shuffled_records"] == 0
    assert std["memory"]["peak_records"] == 4 and imp["memory"]["peak_records"] == 1
    assert bc["network"]["broadcast_bytes"] > 0
    # p12: the C record arrives mid-list for key 112
    vals = next(g["values"] for red in std["stages"]["shuffle"] for g in red if g["key"] == "112")
    assert vals == ["<O,2,780>", "<O,12,799>", "<C,USA,Texas>", "<O,17,324>"]


def test_outer_joins_and_broadcast_limits():
    counts = {j: M.run_join("standard", j, 3, extra=True)["check"]["rows"] for j in M.JOIN_TYPES}
    assert counts == {"inner": 9, "left": 10, "right": 10, "full": 11}
    for j in M.JOIN_TYPES:
        assert M.run_join("improved", j, 3, extra=True)["check"]["ok"]
    assert not M.run_join("broadcast", "full", 3, extra=True)["check"]["ok"]
    assert M.run_join("broadcast", "left", 3, extra=True)["check"]["ok"]


def test_improved_join_without_grouping_comparator_breaks():
    r = M.run_join("improved", "inner", 3, group_cmp=False)
    assert r["check"]["rows"] == 0 and len(r["check"]["missing"]) == 9


def test_join_scale_crossover():
    small = M.join_scale(0, 0)
    assert small["winner"] == "broadcast"
    s = M.join_scale(3, 8, 100, 1024)
    assert s["winner"] == "broadcast" and s["crossover_customers"] > 1e6
    huge = M.join_scale(7, 8.5, 100, 1024)
    assert huge["winner"] == "repartition"


# ------------------------------------------------------------------ API
def test_api_endpoints(client):
    r = client.get("/api/bigdata/hdfs", query_string={"kill": 2, "datanodes": 6})
    assert r.status_code == 200
    d = r.get_json()
    assert d["dead"] == ["DN2"] and "rack_loss" in d and d["meta"]["compute_ms"] < 300
    r = client.get("/api/bigdata/mapreduce", query_string={"job": "wordcount", "combiner": "proper", "reducers": 3})
    d = r.get_json()
    assert r.status_code == 200 and d["check"]["all_ok"] and d["shuffle"]["records"] == 8
    r = client.get("/api/bigdata/mr-joins", query_string={"algo": "improved"})
    d = r.get_json()
    assert r.status_code == 200 and d["check"]["rows"] == 9 and len(d["compare"]) == 3
    assert client.get("/api/bigdata/mapreduce", query_string={"job": "nope"}).status_code == 400
    assert client.get("/api/bigdata/hdfs", query_string={"replication": 9}).status_code == 400
    again = client.get("/api/bigdata/mapreduce", query_string={"job": "wordcount", "combiner": "proper", "reducers": 3}).get_json()
    assert again["meta"]["cached"]


def test_catalog_section(client):
    c = client.get("/api/catalog").get_json()["bigdata_mapreduce"]
    assert c["emp30"]["rows"] == 30 and c["emp30"]["expected_avg"]["5"] == pytest.approx(58741.64)
    assert c["joins"]["joined_rows"] == 9
    assert set(c["jobs"]) == {"avg", "wordcount", "index", "topn", "secsort"}
