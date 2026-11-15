def get(client, url, status=200):
    r = client.get(url)
    assert r.status_code == status, r.get_json()
    return r.get_json()


def test_random_walk_study(client):
    j = get(client, "/api/crossover/random-walk?sims=500&perms=60")
    assert j["tails"]["beyond4"] > 20 * j["tails"]["expected4"]           # fat tails
    assert j["acf_abs"][0] > 0.2 and abs(j["acf_r"][0]) < 0.15              # size has memory, sign barely
    vr10 = next(v for v in j["variance_ratio"] if v["k"] == 10)
    assert vr10["vr"] < 1
    assert 0 < j["drawdown"]["share_worse_normal"] < 1
    again = get(client, "/api/crossover/random-walk?sims=500&perms=60")
    assert again["meta"]["cached"] and again["tails"] == j["tails"]


def test_vix_two_compartments_beat_one(client):
    j = get(client, "/api/crossover/vix-events")
    assert len(j["events"]) >= 15
    assert j["two"]["r2"] > j["one"]["r2"] + 0.2
    assert j["two"]["hl_fast"] < 5 < 20 < j["two"]["hl_slow"]


def test_vix_model_fit_improves_r2(client):
    fixed = get(client, "/api/crossover/vix-model?half_life=60&threshold=3.4")
    fitted = get(client, "/api/crossover/vix-model?threshold=3.4&fit=true")
    assert fitted["r2"] > fixed["r2"] and 0.4 < fitted["r2"] < 0.8
    assert len(fitted["dates"]) == len(fitted["vix"]) == len(fitted["model"]) <= 1500


def test_vix_model_window(client):
    j = get(client, "/api/crossover/vix-model?window=2020")
    assert j["dates"][0] >= "2019-10-01" and j["dates"][-1] < "2021-01-01"


def test_crash_shapes(client):
    j = get(client, "/api/crossover/crash-shapes")
    assert len(j["episodes"]) >= 50
    assert 0.5 < j["cumulative"][2] < 0.75
    assert j["correlations"]["mode2_trough_at"] > 0.5                       # mode 2 = timing of the bottom
    s = get(client, "/api/crossover/crash-sample?b1=0&b2=0&b3=0")
    assert s["curve"][0] == 0 and min(s["curve"]) < -0.4


def test_crossover_validation(client):
    assert get(client, "/api/crossover/crash-shapes?spy_depth=0.5&stock_depth=0.8", 422)["error"]["code"] == "insufficient_data"
    assert get(client, "/api/crossover/vix-events?ratio=9", 400)["error"]["code"] == "out_of_range"
    assert get(client, "/api/crossover/crash-shapes?tickers=SPY,FOO", 400)["error"]["code"] == "invalid_parameter"


# ---------------------------------------------------------------------------- catalog and summaries
import numpy as np  # noqa: E402
import pytest  # noqa: E402


def test_catalog_crossover(client):
    c = get(client, "/api/catalog")["crossover"]
    assert [t["id"] for t in c["tickers"]] == ["SPY", "AAPL", "MSFT", "NVDA", "JPM", "XOM"]
    assert c["first_year"] == 1993 and c["last_year"] >= 2026
    w = c["vix_windows"]
    assert w[0]["id"] == "all" and w[0]["label"] == f"{c['vix']['start'][:4]}–{c['vix']['end'][:4]}"
    assert [x["id"] for x in w[1:]] == ["2008", "2020", "2025"]
    for x in w[1:]:                                     # every window the catalog offers is accepted
        assert get(client, f"/api/crossover/vix-model?window={x['id']}")["window"] == x["id"]


def test_random_walk_tests_and_bands(client):
    j = get(client, "/api/crossover/random-walk?sims=500&perms=60")
    t = {x["id"]: x for x in j["tests"]}
    assert list(t) == ["normal", "reversion", "memory", "crash"]
    T, M = j["tails"], j["drawdown"]
    assert t["normal"]["fail"] == (T["beyond4"] > 3 * max(1, T["expected4"]))
    r10 = next(r for r in j["robust"] if r["k"] == 10)
    assert t["reversion"]["fail"] == (abs(r10["z"]) > 1.96) and t["reversion"]["vr"] == r10["vr"]
    assert t["memory"]["fail"] == (j["acf_abs"][0] > 2 * j["acf_band"])
    assert t["crash"]["fail"] == (M["share_worse_normal"] < 0.05)
    d = j["decades"]
    assert j["decades_weaker"] == (abs(d[-1]["z10"]) < abs(d[0]["z10"]))
    H = j["hist"]
    assert len(H["expected_raw"]) == len(H["raw"]) and sum(H["expected_raw"]) == pytest.approx(H["n_raw"], rel=1e-3)
    assert len(H["normal_curve"]["z"]) == len(H["normal_curve"]["raw"]) == 321
    w = get(client, "/api/crossover/walks?n=8&seed=3")
    walks = np.array(w["walks"])
    assert np.allclose(w["walk_lo"], walks.min(axis=0), atol=1e-4) and np.allclose(w["walk_hi"], walks.max(axis=0), atol=1e-4)


def test_vix_summaries(client):
    e = get(client, "/api/crossover/vix-events")
    assert e["n_events"] == len(e["events"])
    assert e["two_better"] == (e["two"]["r2"] > e["one"]["r2"] + 0.1)
    assert e["two"]["a"] + e["two"]["a_slow"] == pytest.approx(1)
    assert all(x["ratio"] == pytest.approx(x["peak"] / x["base"], rel=1e-4) for x in e["events"])
    m = get(client, "/api/crossover/vix-model")
    assert m["dose_share"] == pytest.approx(m["n_doses"] / m["n_days"], rel=1e-4)
    assert list(m["windows"]) == ["2008", "2020", "2025"]


def test_crash_shape_summaries(client):
    j = get(client, "/api/crossover/crash-shapes")
    eps = j["episodes"]
    spy = [i for i, e in enumerate(eps) if e["ticker"] == "SPY"]
    assert j["counts"] == {"total": len(eps), "spy": len(spy), "stocks": len(eps) - len(spy)}
    assert j["main_spy"] == max(spy, key=lambda i: eps[i]["depth"])
    labels = {e["peak"]: e["label"] for e in eps if e["label"]}
    assert labels == {"2000-03-24": "SPY dot-com", "2007-10-09": "SPY 2008", "2020-02-19": "SPY 2020"}
    assert len(j["mode_bands"]) == 3
    mean, sd, mode = np.array(j["mean"]), j["sd"][0], np.array(j["modes"][0])
    assert np.allclose(j["mode_bands"][0]["plus"], mean + 2 * sd * mode, atol=1e-4)
    curves = np.sort(np.array([e["curve"] for e in eps]), axis=0)
    assert np.allclose(j["q25"], curves[int(0.25 * (len(eps) - 1))], atol=1e-5)
    c1 = j["correlations"]["mode1_depth"]
    assert j["depth_lean"] == ("U" if c1 < -0.2 else "V" if c1 > 0.2 else None)
    assert 0.8 < j["reference"]["jaw_three_modes"] < 1
