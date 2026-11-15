"""/api/shape: contract, correctness against ShapeLab's documented results, validation, caching."""
from __future__ import annotations

import math

import numpy as np
import pytest

BASE = "/api/shape"


def ok(client, url):
    r = client.get(BASE + url)
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert {"compute_ms", "cached"} <= set(body["meta"])
    return body


def err(client, url, status=400):
    r = client.get(BASE + url)
    assert r.status_code == status, (url, r.status_code, r.get_json())
    body = r.get_json()
    assert set(body["error"]) == {"code", "message"}
    return body["error"]


# ---------------------------------------------------------------- outlines / Fourier


def test_outlines(client):
    body = ok(client, "/outlines")
    names = {o["name"]: o["n_points"] for o in body["outlines"]}
    assert names == {"bird1": 843, "bird2": 793}


def test_fourier_ten_percent_needs_six_coefficients(client):
    b = ok(client, "/fourier?outline=bird1")
    assert b["coefficients_for_10pct"] == 6
    assert b["keep"] == 6  # default keep is the 10% count
    k = ok(client, f"/fourier?outline=bird1&keep={b['coefficients_for_10pct']}")
    assert k["error"] <= 0.10
    assert k["share"] == pytest.approx(6 / 843, rel=1e-4)
    assert len(k["original"]) == len(k["reconstruction"]) == 843
    assert len(k["error_curve"]) == 201 and k["error_curve"][0] == pytest.approx(1.0)
    assert k["error"] == pytest.approx(k["error_curve"][6])
    assert all(a >= b - 1e-9 for a, b in zip(k["error_curve"], k["error_curve"][1:]))
    # five coefficients are not enough
    assert ok(client, "/fourier?outline=bird1&keep=5")["error"] > 0.10


def test_fourier_full_keep_is_exact(client):
    b = ok(client, "/fourier?outline=bird2&keep=793")
    assert b["error"] == pytest.approx(0, abs=1e-9)
    assert np.allclose(b["original"], b["reconstruction"], atol=1e-4)


@pytest.mark.parametrize("q", ["outline=nope", "outline=../bird1", "keep=0", "keep=844", "keep=2.5", "keep=abc"])
def test_fourier_validation(client, q):
    err(client, f"/fourier?{q}")


# ---------------------------------------------------------------- jaw model


def test_jaw_model(client):
    b = ok(client, "/jaw-model?n_modes=3")
    assert b["n_shapes"] == 300 and b["n_landmarks"] == 41
    assert np.array(b["mean"]).shape == (41, 2)
    assert np.array(b["modes"]).shape == (3, 41, 2)
    assert len(b["sd"]) == len(b["variance_explained"]) == len(b["cumulative"]) == 3
    assert b["cumulative"][-1] == pytest.approx(0.904, abs=0.01)
    assert np.array(b["aligned_sample"]).shape == (24, 41, 2)
    assert np.linalg.norm(b["mean"]) == pytest.approx(1.0, abs=1e-4)  # Procrustes mean has unit size


def test_jaw_sample(client):
    mean = np.array(ok(client, "/jaw-model")["mean"])
    zero = ok(client, "/jaw-sample")
    assert np.allclose(zero["shape"], mean, atol=1e-5)
    assert zero["b"] == [0, 0, 0, 0, 0, 0]
    m = ok(client, "/jaw-model?n_modes=1")
    s = ok(client, "/jaw-sample?b1=2")
    expected = mean + 2 * m["sd"][0] * np.array(m["modes"][0])
    assert np.allclose(s["shape"], expected, atol=1e-5)


@pytest.mark.parametrize("q", ["/jaw-model?n_modes=0", "/jaw-model?n_modes=11", "/jaw-sample?b1=3.5", "/jaw-sample?b6=-4"])
def test_jaw_validation(client, q):
    err(client, q)


# ---------------------------------------------------------------- meshes


def test_meshes(client):
    b = ok(client, "/meshes")
    got = {m["name"]: (m["n_vertices"], m["n_faces"]) for m in b["meshes"]}
    assert got["cat4"] == (3400, 6775)
    assert set(got) == {"bunny", "cactus", "cat4", "cylinder"}


def test_mesh(client):
    b = ok(client, "/mesh/cylinder")
    assert len(b["vertices"]) == 3 * b["n_vertices"] and len(b["faces"]) == 3 * b["n_faces"]
    assert b["bbox"] == {"min": [-1, -1, 0], "max": [1, 1, 12]}
    assert max(b["faces"]) < b["n_vertices"]


@pytest.mark.parametrize("name", ["nope", "..", "cat4.off", "..%2Fmeshes%2Fcat4", "%2E%2E"])
def test_mesh_unknown_or_path_like(client, name):
    e = err(client, f"/mesh/{name}", 404)
    assert e["code"] in ("not_found",)


# ---------------------------------------------------------------- spectral


def test_spectral_cat_within_ten_percent_at_15(client):
    b = ok(client, "/spectral?mesh=cat4&basis=120&k=15")
    assert b["error"] <= 0.10
    assert b["coefficients_for_10pct"] == 15
    assert len(b["eigenvalues"]) == len(b["error_curve"]) == 120
    assert b["eigenvalues"][0] == pytest.approx(0, abs=1e-6)
    assert all(x <= y + 1e-12 for x, y in zip(b["eigenvalues"], b["eigenvalues"][1:]))
    assert b["error"] == pytest.approx(b["error_curve"][14])
    assert len(b["vertices"]) == 3 * 3400


def test_spectral_second_call_cached_and_fast(client):
    ok(client, "/spectral?mesh=cat4&basis=120&k=30")
    b = ok(client, "/spectral?mesh=cat4&basis=120&k=31")
    assert b["meta"]["cached"] is True
    assert b["meta"]["compute_ms"] < 100


@pytest.mark.parametrize(
    "q",
    ["mesh=nope", "mesh=../cat4", "basis=9", "basis=301", "basis=120&k=121", "k=0", "mesh=bunny&basis=151"],
)
def test_spectral_validation(client, q):
    err(client, f"/spectral?{q}")


def test_segment(client):
    b = ok(client, "/segment?mesh=cat4&parts=5")
    assert len(b["labels"]) == 3400
    assert set(b["labels"]) == set(range(5))
    assert b["sizes"] == sorted(b["sizes"], reverse=True) and sum(b["sizes"]) == 3400
    err(client, "/segment?parts=1")
    err(client, "/segment?parts=11")
    err(client, "/segment?basis=20")


# ---------------------------------------------------------------- curvature


@pytest.mark.parametrize("mesh", ["cat4", "cactus"])
def test_gauss_bonnet(client, mesh):
    b = ok(client, f"/curvature?mesh={mesh}")
    gb = b["gauss_bonnet"]
    assert gb["expected"] == pytest.approx(2 * math.pi * gb["euler_characteristic"], rel=1e-5)  # JSON keeps 6 significant digits
    assert gb["integral"] == pytest.approx(gb["expected"], abs=1e-4)
    assert len(b["gaussian"]) == len(b["mean"])


def test_surface_curvature(client):
    s = ok(client, "/surface-curvature?surface=saddle&n=21")
    assert len(s["u"]) == len(s["v"]) == 21 and np.array(s["K"]).shape == (21, 21)
    assert s["at_origin"]["K"] == pytest.approx(-4) and s["at_origin"]["H"] == pytest.approx(0, abs=1e-12)
    assert np.allclose(s["K"], s["closed_form"]["K"], rtol=1e-4, atol=1e-6)
    m = ok(client, "/surface-curvature?surface=monkey")
    assert m["at_origin"]["k1"] == pytest.approx(0, abs=1e-12) and m["at_origin"]["k2"] == pytest.approx(0, abs=1e-12)
    assert np.allclose(m["K"], m["closed_form"]["K"], rtol=1e-4, atol=1e-6)
    p = ok(client, "/surface-curvature?surface=paraboloid")
    assert p["at_origin"]["K"] == pytest.approx(4) and p["at_origin"]["H"] == pytest.approx(2)
    q = ok(client, "/surface-curvature?surface=quadratic&a1=0.5&a3=1&a4=2&a5=0.5")
    assert q["closed_form"]["max_abs_diff"] < 1e-9
    assert q["at_origin"]["K"] == pytest.approx((4 * 1 * 0.5 - 4) / (1 + 0.25) ** 2)


@pytest.mark.parametrize("q", ["surface=torus", "n=10", "n=82", "surface=quadratic&a3=6"])
def test_surface_validation(client, q):
    err(client, f"/surface-curvature?{q}")


# ---------------------------------------------------------------- skinning


def test_skin_twist_candy_wrapper(client):
    d = ok(client, "/skin?mesh=cylinder&method=dqs&angle=180")
    assert d["cross_section_ratio"] == pytest.approx(1.0, abs=1e-3)
    assert len(d["vertices"]) == 3 * d["n_vertices"] and len(d["faces"]) == 3 * d["n_faces"]
    lbs = ok(client, "/skin?mesh=cylinder&method=lbs&angle=180")
    assert lbs["cross_section_ratio"] < 0.05
    for m in ("nlerp", "slerp"):
        assert ok(client, f"/skin?mesh=cylinder&method={m}&angle=180")["cross_section_ratio"] == pytest.approx(1, abs=1e-3)
    rest = ok(client, "/skin?mesh=cylinder&method=lbs&angle=0")
    mesh = ok(client, "/mesh/cylinder")
    assert np.allclose(rest["vertices"], mesh["vertices"], atol=1e-5)
    assert rest["volume_ratio"] == pytest.approx(1)


def test_skin_cactus(client):
    b = ok(client, "/skin?mesh=cactus&method=dqs")
    assert b["angle"] == pytest.approx(math.degrees(0.31), abs=1e-3)  # README pose by default
    assert b["handles"] == [2232, 4557, 3019, 1795]  # 0-based handle indices
    assert 0.5 < b["volume_ratio"] < 1.5
    assert "cross_section_ratio" not in b


@pytest.mark.parametrize("q", ["mesh=cat4", "method=foo", "angle=361", "angle=-400"])
def test_skin_validation(client, q):
    err(client, f"/skin?{q}")


# ---------------------------------------------------------------- catalogue and derived fields
# The browser holds no shape data: lists, labels, sizes and every derived number come from here.


def test_catalog_shape_section(client):
    r = client.get("/api/catalog")
    assert r.status_code == 200
    c = r.get_json()["shape"]
    meshes = ok(client, "/meshes")["meshes"]
    assert {m["name"]: (m["n_vertices"], m["n_faces"]) for m in meshes} == {
        k: (v["n_vertices"], v["n_faces"]) for k, v in c["meshes"].items()
    }
    assert c["meshes"]["cat4"]["label"] == "Cat" and c["meshes"]["bunny"]["label"] == "Bunny"
    assert c["meshes"]["cat4"]["closed"] is False and c["meshes"]["cat4"]["boundary_vertices"] > 0
    assert c["meshes"]["bunny"]["closed"] is True and c["meshes"]["bunny"]["euler_characteristic"] == 2
    assert c["outlines"] == {
        "bird1": {"label": "Bird 1", "n_points": 843, "coefficients_for_10pct": 6},
        "bird2": {"label": "Bird 2", "n_points": 793, "coefficients_for_10pct": 5},
    }
    jaw = c["jaw"]
    assert (jaw["n_shapes"], jaw["n_landmarks"], jaw["n_coordinates"], jaw["max_modes"]) == (300, 41, 82, 10)
    model = ok(client, "/jaw-model?n_modes=10")
    assert jaw["variance_explained"] == pytest.approx(model["variance_explained"])
    assert jaw["sample_count"] == len(model["aligned_sample"])
    sp = c["spectral"]
    assert sp["meshes"] == ["cat4", "bunny"] and sp["basis"] == {"cat4": 120, "bunny": 100}
    assert sp["gps_dims"] == 20 and sp["max_parts"] == 10
    assert set(c["curvature"]["meshes"]) <= set(c["meshes"])
    rigs = c["skinning"]["rigs"]
    assert set(rigs) == {"cylinder", "cactus"}
    assert rigs["cactus"]["n_handles"] == 4 and rigs["cactus"]["pose_ratios"] == pytest.approx([1, 0.26 / 0.31], rel=1e-5)
    assert rigs["cylinder"]["pose"] == "twist" and rigs["cactus"]["pose"] == "bend"


def test_fourier_summary(client):
    b = ok(client, "/fourier?outline=bird1&keep=6")
    s = b["summary"]
    assert s["share_for_10pct"] == pytest.approx(6 / 843, rel=1e-4)
    assert s["compression"] == pytest.approx(843 / 6, rel=1e-4)
    assert s["doubled"]["keep"] == 12 and s["doubled"]["error"] == pytest.approx(b["error_curve"][12])
    gaps = np.linalg.norm(np.array(b["original"]) - np.array(b["reconstruction"]), axis=1)
    assert np.allclose(b["gaps"], gaps, atol=1e-3)
    ec = np.array(b["error_curve"])
    # spectrum[i] is the size of coefficient i: dropping it lowers the squared error by spectrum[i]^2
    assert len(b["spectrum"]) == 200
    assert np.allclose(np.array(b["spectrum"][:50]) ** 2, ec[:50] ** 2 - ec[1:51] ** 2, atol=1e-5)
    assert ok(client, "/fourier?outline=bird2&keep=793")["summary"]["doubled"] is None


def test_jaw_sample_summary(client):
    mean = np.array(ok(client, "/jaw-model")["mean"])
    s = ok(client, "/jaw-sample?b1=2&b2=-1")
    shape = np.array(s["shape"])
    shifts = np.linalg.norm(shape - mean, axis=1)
    sm = s["summary"]
    assert sm["distance_sd"] == pytest.approx(math.sqrt(5))
    assert np.allclose(sm["shifts"], shifts, atol=1e-5)
    j = int(np.argmax(shifts))
    assert sm["max_shift"]["landmark"] == j + 1
    assert sm["max_shift"]["share_of_width"] == pytest.approx(shifts[j] / np.ptp(mean[:, 0]), rel=1e-4)
    assert ok(client, "/jaw-sample")["summary"]["distance_sd"] == 0


def test_spectral_and_segment_summary(client):
    b = ok(client, "/spectral?mesh=cat4&k=30")
    assert b["basis"] == 120 and b["n_vertices"] == 3400  # default basis is the one the interface uses
    assert b["summary"]["share_kept"] == pytest.approx(30 / 3400, rel=1e-4)
    g = ok(client, "/segment?mesh=cat4&parts=5")
    assert g["n_vertices"] == 3400 and g["restarts"] == 10
    assert g["shares"] == pytest.approx([x / 3400 for x in g["sizes"]], rel=1e-5)
    assert sum(g["shares"]) == pytest.approx(1)
    sm = g["summary"]
    assert (sm["largest"], sm["smallest"]) == (max(g["sizes"]), min(g["sizes"]))
    assert sm["largest_share"] == pytest.approx(max(g["sizes"]) / 3400, rel=1e-5)
    assert sm["smallest_share"] == pytest.approx(min(g["sizes"]) / 3400, rel=1e-5)


@pytest.mark.parametrize("mesh", ["cat4", "bunny"])
def test_curvature_summary(client, mesh):
    b = ok(client, f"/curvature?mesh={mesh}")
    for f in ("gaussian", "mean"):
        x = np.array([np.nan if v is None else v for v in b[f]], dtype=float)
        assert b["summary"][f]["negative_share"] == pytest.approx(np.mean(x < 0), abs=1e-6)
        assert b["summary"][f]["positive_share"] == pytest.approx(np.mean(x > 0), abs=1e-6)
        assert b["range_limit"][f] == pytest.approx(max(map(abs, b["range"][f])))
    gb = b["gauss_bonnet"]
    assert gb["matches"] is True
    assert gb["expected_over_pi"] == 2 * gb["euler_characteristic"]
    assert b["range_percentiles"] == [2, 98]
    s = ok(client, "/surface-curvature?surface=saddle&n=21")
    assert s["K_abs_max"] == pytest.approx(np.max(np.abs(s["K"])), rel=1e-5)


def test_skin_derived_fields(client):
    d = ok(client, "/skin?mesh=cylinder&method=lbs&angle=90")
    assert d["pose"] == "twist" and d["pose_angles"] == [90]
    assert d["lbs_midpoint_scale"] == pytest.approx(math.cos(math.radians(45)))
    assert d["full_turn"] is False
    assert ok(client, "/skin?mesh=cylinder&method=lbs&angle=360")["full_turn"] is True
    assert ok(client, "/skin?mesh=cylinder&method=lbs&angle=0")["full_turn"] is False
    c = ok(client, "/skin?mesh=cactus&method=dqs&angle=20")
    assert c["pose"] == "bend" and c["pose_angles"] == pytest.approx([20, 20 * 0.26 / 0.31], rel=1e-5)
    assert "lbs_midpoint_scale" not in c
