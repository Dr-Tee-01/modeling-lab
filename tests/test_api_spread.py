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
