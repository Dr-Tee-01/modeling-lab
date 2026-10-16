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
