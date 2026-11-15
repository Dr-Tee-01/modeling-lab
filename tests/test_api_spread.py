import numpy as np


def get(client, url, status=200):
    r = client.get(url)
    assert r.status_code == status, r.get_json()
    return r.get_json()


def test_sir_fit_matches_1978_data(client):
    j = get(client, "/api/spread/sir/fit")
    assert abs(j["beta"] - 1.67) < 0.02 and abs(j["alpha"] - 0.44) < 0.01 and j["rmse"] < 18


def test_sir_conserves_population(client):
    j = get(client, "/api/spread/sir?beta=2&alpha=0.4")
    total = np.array(j["S"]) + np.array(j["I"]) + np.array(j["R"]) + np.array(j["V"])
    assert np.allclose(total, 763, atol=1e-6)
    assert j["meta"]["compute_ms"] >= 0


def test_lockdown_lowers_the_peak(client):
    free = get(client, "/api/spread/sir")["peak"]
    locked = get(client, "/api/spread/sir?lockdown=true&lock_start=2&lock_days=10&lock_cut=60")["peak"]
    assert locked < free


def test_dosing_one_compartment_plateau(client):
    j = get(client, "/api/spread/dosing?dose=100&tau=8&half_life=22&absorbed=0.12")
    p = np.exp(-np.log(2) / 22 * 8)
    assert np.isclose(j["steady_peak"], 4 / (1 - p), rtol=1e-5)          # API rounds to 6 significant digits
    assert max(j["blood"]) <= j["steady_peak"] + 1e-6          # levels approach the plateau from below


def test_dosing_validation(client):
    assert get(client, "/api/spread/dosing?mec=30&mtc=20", 400)["error"]["code"] == "invalid_parameter"
    assert get(client, "/api/spread/dosing?dose=abc", 400)["error"]["code"] == "invalid_parameter"
    assert get(client, "/api/spread/dosing?tau=0", 400)["error"]["code"] == "out_of_range"


def test_bass_solver_matches_closed_form(client):
    j = get(client, "/api/spread/bass?p=0.02&q=0.5")
    assert j["max_solver_error"] < 1e-4


def test_rumour_peak_matches_theory(client):
    j = get(client, "/api/spread/rumour?beta=0.4&gamma=0.1")
    assert abs(j["peak_spreaders"] - j["theory"]["s_max"]) < 1
    assert abs(j["reach"] - j["theory"]["r_inf"]) < 0.01 and j["conserved_drift"] < 1e-6


def test_decay_mean_tracks_exponential(client):
    j = get(client, "/api/spread/decay?lam=0.1&runs=400")
    assert j["max_gap_to_theory"] < 6 and abs(j["half_life_measured"] - j["half_life_theory"]) <= 1


def test_walk_moments_match_theory(client):
    j = get(client, "/api/spread/walk?p=0.6&steps=100&walkers=5000")
    assert abs(j["mean"][-1] - 20) < 1 and abs(j["var"][-1] / 96 - 1) < 0.08


def test_walk_budget(client):
    assert get(client, "/api/spread/walk?steps=2000&walkers=20000", 400)["error"]["code"] == "too_large"


def test_walk2d_msd(client):
    j = get(client, "/api/spread/walk2d?walkers=2000&steps=200")
    assert abs(j["msd"][-1] / j["msd_theory"][-1] - 1) < 0.08


def test_coins_converge(client):
    j = get(client, "/api/spread/coins?n=20000")
    assert all(abs(f - 0.5) < 0.03 for f in j["final"])


# ------------------------------------------------ values the views show, served by the backend
def test_catalog_spread_section(client):
    c = get(client, "/api/catalog")["spread"]
    flu = c["flu"]
    assert (flu["year"], flu["month"], flu["population"], flu["initial_cases"]) == (1978, "January", 763, 1)
    assert (flu["first_day"], flu["last_day"], flu["observations"]) == (1, 14, 14)
    fit = get(client, "/api/spread/sir/fit")
    assert flu["best_fit"] == {"beta": round(fit["beta"], 2), "alpha": round(fit["alpha"], 2)}
    assert c["dosing"]["volume_l"] == 3 and c["dosing"]["step_h"] == 0.05
    assert c["sir"]["step_day"] == 0.01 and c["rumour"]["step_day"] == 0.05 and c["rumour"]["days"] == 100
    assert c["decay"]["n0"] == 1000 and c["coins"]["sequences"] == 6
    assert c["walk"]["paths_shown"] == 40 and c["walk2d"]["paths_shown"] == 12 and c["walk2d"]["msd_slope"] == 1.5


def test_dosing_reports_horizon_and_solver(client):
    j = get(client, "/api/spread/dosing")
    assert j["horizon_days"] == 10 and j["hours"] == 240 and j["volume_ml"] == 3000
    assert np.isclose(j["k"], np.log(2) / 22, rtol=1e-5) and j["step_h"] == 0.05


def test_bass_summary(client):
    j = get(client, "/api/spread/bass?p=0.03&q=0.38&market=100&years=25")
    assert j["word_of_mouth_takes_over"] is True
    assert j["final_adopters"] == j["adopters"][-1] and np.isclose(j["final_share"], j["adopters"][-1] / 100, rtol=1e-5)
    assert get(client, "/api/spread/bass?p=0.1&q=0.05")["word_of_mouth_takes_over"] is False


def test_sir_summary_fields(client):
    j = get(client, "/api/spread/sir?beta=1.67&alpha=0.44&lockdown=true&lock_start=2&lock_days=10&lock_cut=60")
    assert j["population"] == 763 and j["days"] == 20 and j["rmse_days"] == [1, 14]
    assert np.isclose(j["at_peak"]["I"], j["peak"]) and np.isclose(sum(j["at_peak"].values()), 763, atol=1)
    assert np.isclose(j["days_ill"], 1 / 0.44, rtol=1e-5)
    assert j["lockdown"] == {"start": 2, "end": 12, "factor": 0.4}
    assert get(client, "/api/spread/sir")["lockdown"] is None


def test_rumour_at_peak(client):
    j = get(client, "/api/spread/rumour")
    assert np.isclose(j["at_peak"]["spreader"], j["peak_spreaders"]) and j["days"] == 100 and j["population"] == 1000


def test_decay_summary_fields(client):
    j = get(client, "/api/spread/decay?lam=0.05&runs=50")
    th = j["half_life_step"]
    assert th == round(np.log(2) / 0.05) and j["n0"] == 1000 and j["t_max"] == 100
    assert j["first_run_left_at_half_life"] == sum(t > th for t in j["first_run_decay_times"])
    assert np.allclose(np.array(j["band_hi"]) - np.array(j["band_lo"]), 2 * np.array(j["sd"]), atol=1e-3)


def test_coins_summary_fields(client):
    j = get(client, "/api/spread/coins?n=2000")
    assert j["final_min"] == min(j["final"]) and j["final_max"] == max(j["final"]) and j["sequences"] == 6
    assert np.allclose(np.array(j["band_hi"]) - 0.5, j["band"], atol=1e-5) and j["p_heads"] == 0.5


def test_walk_summary_fields(client):
    j = get(client, "/api/spread/walk?p=0.6&steps=100&walkers=2000")
    m = j["summary"]
    assert m["mean"] == j["mean"][-1] and m["var"] == j["var"][-1] and m["steps"] == 100
    assert np.isclose(m["mean_theory"], 20) and np.isclose(m["var_theory"], 96)
    assert np.isclose(m["typical_distance"], np.sqrt(j["var"][-1]), rtol=1e-5)
    assert np.isclose(m["var_rel_error"], abs(m["var"] / 96 - 1), rtol=1e-4)
    assert np.allclose(m["range95"], [20 - 2 * np.sqrt(96), 20 + 2 * np.sqrt(96)], rtol=1e-5)
    assert np.isclose(m["p_left"], 0.4) and len(j["paths"]) == 40
    assert np.allclose(j["band_hi"], np.array(j["mean_theory"]) + 2 * np.sqrt(j["var_theory"]), atol=1e-3)
    assert len(j["final_edges"]) == len(j["final_values"]) + 1
    b = j["bell"]
    assert len(b["x"]) == len(b["y"]) and np.isclose(max(b["y"]), 1, atol=1e-3)
    assert np.isclose(b["x"][int(np.argmax(b["y"]))], m["mean_theory"])


def test_walk_degenerate_has_no_ratio(client):
    m = get(client, "/api/spread/walk?p=1&steps=50&walkers=100")["summary"]
    assert m["var_rel_error"] is None and m["range95"] == [50, 50]


def test_walk2d_summary(client):
    j = get(client, "/api/spread/walk2d?walkers=200&steps=100")
    assert j["summary"]["msd"] == j["msd"][-1] and j["summary"]["msd_theory"] == 150 and len(j["paths"]) == 12
