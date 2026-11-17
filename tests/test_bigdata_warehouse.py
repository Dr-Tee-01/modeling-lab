"""Big Data Lab, warehouse topic: OLAP operations, SQL grouping operators and the cuboid lattice
reproduce the numbers printed on the IE494 slides, and the API serves them."""
import itertools

import pytest

from modlab.bigdata import datasets as D
from modlab.bigdata import warehouse as W


def cells(step):
    return W._cells_of(step)


# ------------------------------------------------------------------ datasets and totals
def test_dataset_totals_match_the_slides():
    assert sum(p for *_, p in D.FACT9) == 735
    assert sum(a for *_, a in D.FACT30) == 3350
    assert sum(map(sum, D.ENROL.values())) == 863
    assert W.olap("fact30", "drill", "item", "apex")["steps"][0]["total"] == 3350


# ------------------------------------------------------------------ OLAP: slice and dice on the enrolment table
def test_bit_slice_is_the_slide_30_column():
    r = W.olap("enrol", "slice", "degree", member="BIT")
    after = cells(r["steps"][-1])
    assert after == {(c,): D.ENROL[c][D.DEGREES.index("BIT")] for c in D.ENROL}
    assert r["after_total"] == 217
    assert all(c["ok"] for c in r["checks"]) and r["checks"]


def test_dice_bit_bcom_by_three_countries():
    r = W.olap("enrol", "dice", "degree", dice="degree:BIT|BCom;country:India|Australia|Malaysia")
    got = cells(r["steps"][-1])
    assert got == {("Australia", "BCom"): 103, ("Australia", "BIT"): 21, ("India", "BCom"): 47, ("India", "BIT"): 30,
                   ("Malaysia", "BCom"): 31, ("Malaysia", "BIT"): 43}
    assert r["after_total"] == 275
    assert r["pick"]["selected"] == {"degree": ["BCom", "BIT"], "country": ["Australia", "India", "Malaysia"]}


def test_dice_star_keeps_the_whole_cube_and_bad_members_fall_back_to_the_default():
    assert W.olap("enrol", "dice", "degree", dice="*")["after_total"] == 863
    assert W.olap("enrol", "dice", "degree", dice="planet:Mars")["after_total"] == 275


def test_vancouver_slice_is_the_slide_17_table():
    r = W.olap("hk", "slice", "location", member="Vancouver")
    got = cells(r["steps"][-1])
    assert [[got[(q, it)] for it in D.HK_ITEMS] for q in D.HK_QUARTERS] == W.SLIDE_EXPECT["vancouver"]["cells"]
    assert r["checks"][0]["ok"]
    assert "s.city = 'Vancouver'" in r["steps"][-1]["sql"]


# ------------------------------------------------------------------ OLAP: roll-up and drill-down
def test_rollup_location_city_province_country_all_is_correct():
    r = W.olap("hk", "rollup", "location", "dim")
    assert [s["cuboid"] for s in r["steps"]] == ["C(i,j,k)", "C(i′,j,k)", "C(i″,j,k)", "C(0,j,k)"]
    country = cells(r["steps"][2])
    for q, it in itertools.product(D.HK_QUARTERS, D.HK_ITEMS):
        qi, ii = D.HK_QUARTERS.index(q), D.HK_ITEMS.index(it)
        canada = sum(D.HK_SALES[c][qi][ii] for c in D.HK_SALES if D.LOCATION[c][1] == "Canada")
        usa = sum(D.HK_SALES[c][qi][ii] for c in D.HK_SALES if D.LOCATION[c][1] == "USA")
        assert country[(q, "Canada", it)] == canada and country[(q, "USA", it)] == usa   # rows = time (slide 18 layout)
    assert len({s["total"] for s in r["steps"]}) == 1 and r["preserved"]
    assert "JOIN location l ON l.city = s.city" in r["steps"][2]["sql"]


def test_fact30_drill_down_reproduces_slides_27_to_29():
    r = W.olap("fact30", "drill", "item", "apex")
    assert [s["cuboid"] for s in r["steps"]] == ["C(0,0,0)", "C(i,0,0)", "C(i,j,0)", "C(i,j,k)"]
    assert cells(r["steps"][1]) == {(k,): v for k, v in W.SLIDE_EXPECT["f30_items"]["values"].items()}
    assert cells(r["steps"][2]) == W.SLIDE_EXPECT["f30_item_store"]["values"]
    assert r["steps"][3]["cells"] == 30
    assert all(c["ok"] for c in r["checks"]) and len(r["checks"]) == 3


def test_rollup_is_the_reverse_of_drill_down():
    up = W.olap("fact30", "rollup", "item", "apex")["steps"]
    down = W.olap("fact30", "drill", "item", "apex")["steps"]
    assert [s["node"] for s in up] == [s["node"] for s in reversed(down)]


def test_pivot_rotates_and_lab_supplies_match_slide_32():
    r = W.olap("lab", "pivot", "item")
    assert r["steps"][0]["table"]["view"] == "relation" and r["steps"][0]["cells"] == 36
    assert r["steps"][-1]["table"]["rows"]["dim"] == "item"
    assert r["checks"] and r["checks"][0]["ok"]
    assert "SUM(CASE WHEN" in r["steps"][-1]["sql"]


def test_lattice_of_the_hk_cube_counts_prod_levels_plus_one():
    r = W.olap("hk", "rollup", "location")
    assert r["lattice"]["count"] == (3 + 1) * (2 + 1) * (1 + 1)
    assert r["lattice"]["visited"][0] == "0-0-0" and r["lattice"]["final"] == "3-0-0"


# ------------------------------------------------------------------ windows
def test_windows_rank_and_lag():
    w = W.window("tumbling", 3, "sum", "CSCL")
    cscl = D.PIVOT_SALES["CSCL"]
    assert [x["values"]["CSCL"] for x in w["windows"]] == [sum(cscl[0:3]), sum(cscl[3:6]), sum(cscl[6:9])]
    s = W.window("sliding", 3, "avg", "Phenol")
    ph = D.PIVOT_SALES["Phenol"]
    assert s["rows"][4]["agg"] == pytest.approx(sum(ph[2:5]) / 3, rel=1e-4)
    assert s["rows"][0]["frame_rows"] == 1 and s["partial"] == 2
    assert s["rows"][3]["lag"] == ph[2] and s["rows"][0]["lag"] is None
    # CSCL is the largest item every month
    assert all(r == 1 for r in s["rank"][0])


# ------------------------------------------------------------------ SQL grouping operators
def test_fact9_cube_has_33_rows_and_total_735():
    g = W.grouping("fact9", "cube", ["store", "item", "cust"])
    assert g["row_count"] == 33 and g["apex"] == 735
    assert [s["rows"] for s in g["sets"]] == [9, 6, 4, 6, 2, 3, 2, 1]


def test_fact9_rollup_item_store_cust_has_19_rows():
    g = W.grouping("fact9", "rollup", ["item", "store", "cust"])
    assert g["row_count"] == 19 and g["apex"] == 735
    assert [s["label"] for s in g["sets"]] == ["(item, store, cust)", "(item, store)", "(item)", "()"]


def test_cube_equals_union_of_group_bys_and_of_rotated_rollups():
    for fk in W.FACTS:
        for order in itertools.permutations(W.FACTS[fk].dims):
            c = W.grouping(fk, "cube", list(order))["constructions"]
            assert c["union_equal"] and c["rot_equal"]
            assert c["rot_union"] == c["cube_rows"] < c["rot_union_all"]
    assert W.rotated_rollup_coverage(3)["covered"] == 8
    assert W.rotated_rollup_coverage(4) == {"n": 4, "covered": 14, "all": 16, "missing": ["ac", "bd"]}


def test_grouping_bits_and_the_real_null_pitfall():
    plain = W.grouping("fact9", "cube")
    assert all(any(r["grouping"]) or r["gid"] == 0 for r in plain["rows"]) and plain["null_rows"] == 0
    g = W.grouping("fact9", "cube", with_null=True)
    assert g["row_count"] == 37 and g["fact"]["total"] == 755
    apex_like = [r for r in g["rows"] if r["vals"] == [None, None, None]]
    assert len(apex_like) == 2 and sorted(r["amount"] for r in apex_like) == [20, 755]
    assert sorted(tuple(r["grouping"]) for r in apex_like) == [(1, 1, 0), (1, 1, 1)]
    assert len(g["ambiguous"]) == 4


def test_deck_sql_bugs_and_their_fixes():
    bugs = {b["query"]: b for b in W.deck_bugs()}
    assert (bugs["item axis"]["deck_rows"], bugs["item axis"]["fixed_rows"]) == (4, 3)
    assert (bugs["item × store"]["deck_rows"], bugs["item × store"]["fixed_rows"]) == (12, 6)
    assert (bugs["item × store × customer"]["deck_rows"], bugs["item × store × customer"]["fixed_rows"]) == (33, 9)
    assert bugs["dice store s1, item i2"]["deck_rows"] == 0 and bugs["dice store s1, item i2"]["fixed_rows"] == 2
    assert bugs["dice from the fact table"]["deck_rows"] == "error"


def test_star_versus_snowflake_join_counts():
    s = W.schema()
    assert s["star"]["joins"]["state"] == 1 and s["snowflake"]["joins"]["state"] == 3
    assert set(s["star"]["tables"]["Sales"]) == {"storeID", "itemID", "custID", "price", "qty"}


# ------------------------------------------------------------------ lattice, hierarchies, HRU
@pytest.mark.parametrize("ck", ["hk", "fact30"])
def test_cuboid_count_formula(ck):
    for n in range(2, 6):
        L = W.lattice(ck, n, 0)
        assert L["cuboids"] == 2 ** n == len(L["lattice"]["nodes"])
        h = L["hierarchy"]
        assert h["count"] == pytest.approx(__import__("math").prod(x["L"] + 1 for x in h["levels"]))
        assert sum(x["cuboids"] for x in h["by_depth"]) == h["count"]
    assert W.lattice("hk", 5, 0)["hierarchy"]["count"] == 3 * 2 * 4 * 2 * 2


def test_synthetic_extension_preserves_sums():
    hk = W.lattice_frame("hk")
    assert int(hk["sales"].sum()) == sum(sum(map(sum, g)) for g in D.HK_SALES.values())
    assert int(W.lattice_frame("fact30")["amount"].sum()) == 3350
    assert W.lattice_sizes("hk", 3)[frozenset({"time", "item", "location"})] == 64


@pytest.mark.parametrize("ck,n", [("hk", 4), ("hk", 5), ("fact30", 4), ("fact30", 5)])
def test_hru_benefits_are_monotone(ck, n):
    H = W.hru(W.lattice_sizes(ck, n), 8)
    bens = [r["benefit"] for r in H["rounds"]]
    assert all(a >= b for a, b in zip(bens, bens[1:]))
    assert all(a >= b for a, b in zip(H["totals"], H["totals"][1:]))
    assert H["totals"][0] - H["totals"][-1] == sum(bens)


def test_layer_by_layer_uses_smallest_parents_and_router_follows_kylin():
    L = W.lattice("hk", 4, 3, ["item", "location"], where=True)
    assert L["query"]["route"]["chosen"] == ["item", "location", "time"] and L["query"]["route"]["cost"] == 64
    L2 = W.lattice("hk", 4, 3, ["item", "location"], where=False)
    assert L2["query"]["route"]["chosen"] == ["item", "location"] and L2["query"]["route"]["cost"] == 16
    sizes = W.lattice_sizes("hk", 4)
    for row in L["layers"][1:]:
        parents = [frozenset(row["cuboid"]) | {d} for d in L["dims"] if d not in row["cuboid"]]
        assert row["parent_rows"] == min(sizes[p] for p in parents)
    assert L["layer_reads"] < L["base_reads"] < L["naive_reads"]


# ------------------------------------------------------------------ API
def test_api_endpoints(client):
    r = client.get("/api/bigdata/olap", query_string={"ds": "enrol", "op": "slice", "dim": "degree", "member": "BIT"}).get_json()
    assert r["after_total"] == 217 and r["meta"]["compute_ms"] >= 0
    w = client.get("/api/bigdata/olap", query_string={"op": "window", "wkind": "tumbling", "wsize": 3}).get_json()
    assert w["window"]["windows"][0]["from"] == "Mar"
    c = client.get("/api/bigdata/cube", query_string={"fact": "fact9", "op": "rollup", "order": "item,store,cust"}).get_json()
    assert c["row_count"] == 19
    L = client.get("/api/bigdata/lattice", query_string={"ds": "hk", "n": 4, "where": "true", "q": "item,location"}).get_json()
    assert L["query"]["route"]["cost"] == 64
    assert client.get("/api/bigdata/olap", query_string={"op": "spin"}).status_code == 400
    assert client.get("/api/bigdata/cube", query_string={"order": "item,item"}).status_code == 400
    assert client.get("/api/bigdata/lattice", query_string={"n": 9}).status_code == 400
    cat = client.get("/api/catalog").get_json()["bigdata_warehouse"]
    assert cat["cubes"]["enrol"]["total"] == 863 and cat["facts"]["fact9"]["total"] == 735
