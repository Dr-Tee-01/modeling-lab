"""Markets API: contract, plausibility of the walk-forward results, validation,
caching, the fast GARCH against stocklab's loop, and the PyTorch jobs."""
from __future__ import annotations

import importlib.util
import time

import numpy as np
import pandas as pd
import pytest

TORCH = importlib.util.find_spec("torch") is not None


def get(client, url, status=200):
    r = client.get(url)
    assert r.status_code == status, r.get_json()
    return r.get_json()


def wait_job(client, body, timeout=90):
    r = client.post("/api/jobs", json=body)
    assert r.status_code == 200, r.get_json()
    job = r.get_json()["job"]
    t0 = time.time()
    while job["status"] in ("queued", "running"):
        assert time.time() - t0 < timeout, job
        time.sleep(0.25)
        job = get(client, f"/api/jobs/{job['id']}")["job"]
    return job


# ----------------------------------------------------------------------------
def test_tickers(client):
    j = get(client, "/api/markets/tickers")
    assert j["tickers"] == ["SPY", "AAPL", "MSFT", "NVDA", "JPM", "XOM"]
    assert j["test_years"] == list(range(2015, 2027))
    assert j["start"] < "1994" and j["end"] >= "2026-10-01"


def test_prices(client):
    j = get(client, "/api/markets/prices?ticker=MSFT&every=5")
    assert len(j["dates"]) == len(j["price"]) == len(j["log_return"]) > 1000
    assert j["log_return"][0] is None and abs(np.nanmean(np.array(j["log_return"][1:], float))) < 0.02
    assert j["dates"] == sorted(j["dates"])


def test_stylized(client):
    j = get(client, "/api/markets/stylized?ticker=SPY&max_lag=50")
    assert len(j["acf_r"]) == len(j["acf_abs"]) == 50
    assert j["kurtosis"] > 3                                    # fat tails
    assert np.mean(j["acf_abs"][:10]) > 5 * j["band"]           # volatility clustering
    assert abs(np.mean(j["acf_r"][:10])) < 3 * j["band"]
    assert len(j["hist"]["edges"]) == len(j["hist"]["counts"]) + 1 == len(j["normal_expected"]) + 1


def test_volatility_spy(client):
    j = get(client, "/api/markets/volatility?ticker=SPY")
    m = j["models"]
    assert set(m) == {"hist21", "ewma", "garch", "har", "har_x"}
    assert 0.4 < m["har_x"]["r2_log"] < 0.8
    assert m["har_x"]["r2_log"] > m["har"]["r2_log"] > m["hist21"]["r2_log"] > 0.2
    assert all(0 < v["qlike"] < 1 for v in m.values())
    s = j["series"]
    assert len(s["dates"]) == len(s["realized"]) == len(s["forecasts"]["garch"]) > 2500
    v = j["var"]
    assert v["model"] == "har_x" and v["alpha"] == 0.01 and v["n"] == len(s["dates"])
    assert 0.005 < v["fhs"]["rate"] < 0.02 and 0 <= v["normal"]["kupiec_p"] <= 1
    assert set(j["params"]["garch"]) == {str(y) for y in j["years"]}
    mu, w, a, b = j["params"]["garch"]["2015"]
    assert w > 0 and 0 < a < 1 and 0 < b < 1 and a + b < 1


def test_volatility_subset_and_cache(client):
    url = "/api/markets/volatility?ticker=XOM&models=ewma,har&start_year=2018&end_year=2020&var_model=garch&var_alpha=0.05"
    j1 = get(client, url)
    assert set(j1["models"]) == {"ewma", "har"} and j1["years"] == [2018, 2019, 2020]
    assert j1["var"]["model"] == "garch" and j1["series"]["dates"][0] >= "2018"
    j2 = get(client, url)
    assert j2["meta"]["cached"] is True and j2["models"] == j1["models"]


def test_returns_spy(client):
    j = get(client, "/api/markets/returns?ticker=SPY&models=hist_mean,ridge,gbm")
    m = j["models"]
    assert set(m) == {"zero", "hist_mean", "ridge", "gbm"}
    assert m["zero"]["r2_oos"] == 0
    for k in ("hist_mean", "ridge", "gbm"):
        assert abs(m[k]["r2_oos"]) < 0.05
        assert 0.4 < m[k]["hit_rate"] < 0.6 and 0 <= m[k]["dm_p"] <= 1 and m[k]["n"] > 2500
    assert 0.5 < j["always_up_hit_rate"] < 0.6
    bt = j["backtest"]
    assert set(bt) == {"hist_mean", "ridge", "gbm", "buy_hold"}
    assert bt["buy_hold"]["final_multiple"] > 2 and -0.6 < bt["buy_hold"]["max_dd"] < 0
    eq = j["equity"]
    assert len(eq["dates"]) == len(eq["curves"]["gbm"]) == len(eq["curves"]["buy_hold"]) > 500
    assert set(j["hyperparams"]["ridge"]) == {str(y) for y in range(2015, 2027)}
    j2 = get(client, "/api/markets/returns?ticker=SPY&models=gbm,ridge,hist_mean")   # same set, other order
    assert j2["meta"]["cached"] is True


def test_returns_arima_short(client):
    j = get(client, "/api/markets/returns?ticker=JPM&models=arima&start_year=2025&end_year=2026")
    assert abs(j["models"]["arima"]["r2_oos"]) < 0.05
    assert len(j["hyperparams"]["arima"]["order"]) == 3


@pytest.mark.parametrize("url,code", [
    ("/api/markets/prices?ticker=TSLA", "invalid_parameter"),
    ("/api/markets/prices?ticker=SPY&every=21", "out_of_range"),
    ("/api/markets/volatility?models=har,lstm", "invalid_parameter"),
    ("/api/markets/volatility?start_year=2004", "out_of_range"),
    ("/api/markets/volatility?start_year=2020&end_year=2016", "invalid_parameter"),
    ("/api/markets/volatility?var_alpha=0.5", "out_of_range"),
    ("/api/markets/volatility?ticker=pooled", "invalid_parameter"),
    ("/api/markets/returns?models=ridge,transformer", "invalid_parameter"),
    ("/api/markets/returns?ticker=pooled&models=arima", "invalid_parameter"),
    ("/api/markets/returns?start_year=abc", "invalid_parameter"),
    ("/api/markets/stylized?max_lag=1000", "out_of_range"),
])
def test_validation(client, url, code):
    j = get(client, url, status=400)
    assert j["error"]["code"] == code and j["error"]["message"]


def test_fast_garch_matches_stocklab_loop():
    from stocklab import data as D, models as M
    from modlab.markets import garch as G
    r = (D.daily_panel("SPY")["ret"] * 100)
    w = r[(r.index >= "2010-01-01") & (r.index < "2014-01-01")].values
    th_loop, th_fast = M.garch_fit(w), G.garch_fit(w)
    assert np.allclose(th_loop, th_fast, atol=1e-3, rtol=1e-3)
    v0 = np.var(w)
    assert np.allclose(M.garch_filter(w, th_loop, v0), G.garch_filter(w, th_loop, v0), rtol=1e-10)
    ewma_ref = M.ewma_var(r.iloc[:2000])
    from modlab.markets.volatility import ewma_var
    assert np.allclose(ewma_ref.values, ewma_var(r.iloc[:2000]).values, rtol=1e-10)


def test_job_validation(client):
    r = client.post("/api/jobs", json={"kind": "lstm_trap", "epochs": 0})
    assert r.status_code == 400 and r.get_json()["error"]["code"] == "out_of_range"
    r = client.post("/api/jobs", json={"kind": "rnn_returns", "model": "transformer"})
    assert r.status_code == 400


def test_lstm_trap_job(client):
    job = wait_job(client, {"kind": "lstm_trap", "ticker": "SPY", "epochs": 1, "window": 20})
    if not TORCH:
        assert job["status"] == "failed"
        assert "PyTorch is not installed. Install it with: pip install -e '.[rnn]'" in job["error"]
        return
    assert job["status"] == "done", job["error"]
    res = job["result"]
    assert len(res["dates"]) == len(res["actual"]) == len(res["leaky"]) == len(res["clean"]) <= 800
    assert res["scores"]["naive"]["r2"] > 0.99
    assert set(res["xcorr"]) == {str(k) for k in range(6)}


def test_rnn_returns_job(client):
    pytest.importorskip("torch")
    job = wait_job(client, {"kind": "rnn_returns", "ticker": "XOM", "model": "gru", "epochs": 1,
                            "start_year": 2025, "end_year": 2026})
    assert job["status"] == "done", job["error"]
    res = job["result"]
    assert abs(res["models"]["gru"]["r2_oos"]) < 0.05
    assert set(res["epochs_used"]) == {"2025", "2026"}
    assert "gru" in res["equity"]["curves"]


def test_returns_pooled(client):
    j = get(client, "/api/markets/returns?ticker=pooled&models=ridge&start_year=2024&end_year=2026")
    assert set(j["models"]) == {"zero", "ridge"} and j["models"]["ridge"]["n"] > 3000
    assert abs(j["models"]["ridge"]["r2_oos"]) < 0.05
    assert set(j["backtest"]) == {"ridge", "buy_hold"}


def test_jobs_fail_cleanly_without_torch(client, monkeypatch):
    """Simulate a missing PyTorch (sys.modules[name] = None makes the import raise)."""
    import sys
    monkeypatch.setitem(sys.modules, "torch", None)
    msg = "PyTorch is not installed. Install it with: pip install -e '.[rnn]'"
    job = wait_job(client, {"kind": "lstm_trap", "ticker": "AAPL", "epochs": 2, "window": 21})
    assert job["status"] == "failed" and msg in job["error"]
    job = wait_job(client, {"kind": "rnn_returns", "ticker": "AAPL", "model": "lstm", "epochs": 2,
                            "start_year": 2026, "end_year": 2026})
    assert job["status"] == "failed" and msg in job["error"]


# ---------------------------------------------------------------------------- catalog and summaries
def test_catalog_markets(client):
    c = get(client, "/api/catalog")["markets"]
    ids = [t["id"] for t in c["tickers"]]
    assert ids == ["SPY", "AAPL", "MSFT", "NVDA", "JPM", "XOM"]
    by = {t["id"]: t for t in c["tickers"]}
    assert by["SPY"]["kind"] == "fund" and by["AAPL"]["kind"] == "stock" and by["NVDA"]["name"] == "Nvidia"
    assert by["NVDA"]["start"][:4] == "1999" and by["SPY"]["start"][:4] == "1993"      # read from the CSVs
    assert c["first_year"] == 1993 and c["end_year"] == min(c["last_year"], c["year_bounds"][1])
    assert c["val_years"] == 2
    for m in c["vol_models"] + c["return_models"] + c["rnn_models"] + ["zero", "buy_hold"]:
        assert c["models"][m]["name"] and c["models"][m]["short"]
    assert c["models"]["har_x"] == {"name": "HAR-RV + VIX", "short": "HAR + VIX"}
    assert [m["id"] for m in c["trap_models"]] == ["naive", "leaky", "clean"]
    assert c["trap_split"]["train"] + c["trap_split"]["test"] == pytest.approx(1)


def test_volatility_summary(client):
    j = get(client, "/api/markets/volatility?ticker=SPY")
    s, m = j["summary"], j["models"]
    assert s["best"] == max(m, key=lambda k: m[k]["r2_log"]) and s["baseline"] == "hist21"
    assert s["gain_over_baseline"] == pytest.approx(m[s["best"]]["r2_log"] - m["hist21"]["r2_log"], abs=1e-5)
    last = max(j["params"]["har"]["har_x"])
    assert s["har_latest"]["har_x"] == {"year": int(last), "coef": j["params"]["har"]["har_x"][last]}
    S = j["series"]
    r, vn = np.array(S["next_return"]), np.array(S["var_normal"])
    assert len(S["breach_normal"]) == j["var"]["normal"]["exceed"] and all(r[i] < vn[i] for i in S["breach_normal"])
    assert len(S["breach_fhs"]) == j["var"]["fhs"]["exceed"]
    for x in (j["var"]["normal"], j["var"]["fhs"]):
        assert x["verdict"] == ("ok" if x["kupiec_p"] >= 0.05 else "too_often" if x["rate"] > 0.01 else "too_rarely")
    one = get(client, "/api/markets/volatility?ticker=SPY&models=ewma&start_year=2024")["summary"]
    assert one["best"] == "ewma" and one["baseline"] is None and one["gain_over_baseline"] is None


def test_returns_summary(client):
    j = get(client, "/api/markets/returns?ticker=SPY&models=hist_mean,ridge,gbm")
    s, m, bt = j["summary"], j["models"], j["backtest"]
    ks = ["hist_mean", "ridge", "gbm"]
    assert s["best"]["r2"] == max(ks, key=lambda k: m[k]["r2_oos"])
    assert s["always_long"] == [k for k in ks if bt[k]["time_in_market"] > 0.999]
    timers = [k for k in ks if k not in s["always_long"]] or ks
    assert s["best"]["hit"] == max(timers, key=lambda k: m[k]["hit_rate"])
    assert s["best"]["sharpe"] == max(timers, key=lambda k: bt[k]["sharpe"])
    alphas = [v["alpha"] for v in j["hyperparams"]["ridge"].values()]
    assert s["ridge_alpha"] == {"max": max(alphas), "years_at_max": alphas.count(max(alphas)), "years": len(alphas)}
    assert j["cost_bps"] == 5


def test_lstm_trap_summary(client):
    pytest.importorskip("torch")
    job = wait_job(client, {"kind": "lstm_trap", "ticker": "SPY", "epochs": 1, "window": 20})
    res = job["result"]
    assert res["best"] == max(res["scores"], key=lambda k: res["scores"][k]["r2"])
    assert res["xcorr_peak_lag"] == int(max(res["xcorr"], key=res["xcorr"].get))
    assert res["test_over_train_max"] == pytest.approx(res["test_max"] / res["train_max"], rel=1e-4)
    assert res["train_share"] + res["test_share"] == pytest.approx(1)


def test_rnn_returns_summary(client):
    pytest.importorskip("torch")
    job = wait_job(client, {"kind": "rnn_returns", "ticker": "XOM", "model": "gru", "epochs": 1,
                            "start_year": 2025, "end_year": 2026})
    res = job["result"]
    assert res["summary"]["best"]["r2"] == "gru" and res["cost_bps"] == 5
    assert res["summary"]["epochs_used_mean"] == pytest.approx(np.mean(list(res["epochs_used"].values())))
