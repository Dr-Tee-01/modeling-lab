"""/api/shape: ShapeLab geometry (Fourier outlines, jaw shape model, spectral
meshes, curvature, skinning), computed live from the raw files in data/shape.

Expensive pure results (eigenbases, the jaw model, parsed meshes) are memoised
in memory by :mod:`modlab.shape_service`; ``meta.cached`` reports whether the
expensive part of a response came from that cache.
"""
from __future__ import annotations

import numpy as np
from flask import Blueprint

from .. import shape_service as S
from .api import ApiError, choice, clean, endpoint, num
from .catalog import section

bp = Blueprint("shape", __name__, url_prefix="/api/shape")

FOURIER_CURVE_MAX = 200
JAW_SAMPLE_MODES = 6


def _flat(a: np.ndarray) -> np.ndarray:
    return np.asarray(a).reshape(-1)


def _mesh_param(name: str = "mesh", default: str = "cat4") -> str:
    return choice(name, default, S.mesh_names())


# ---------------------------------------------------------------- catalogue


@section("shape")
def shape_catalog() -> dict:
    """Meshes, outlines, the jaw model's facts and each view's mesh list, read from data/shape."""
    jm, _ = S.jaw_model()
    outlines = {}
    for name in S.outline_names():
        f, _ = S.fourier_summary(name)
        outlines[name] = {"label": S.outline_label(name), "n_points": len(f.points), "coefficients_for_10pct": f.coefficients_for_10pct}
    rigs = {}
    for name in S.SKIN_MESHES:
        (_, _, _, handles, _), _ = S.skin_rig(name)
        rigs[name] = {"label": S.mesh_label(name), "pose": S.SKIN_POSE[name], "n_handles": len(handles), "pose_ratios": S.skin_pose_ratios(name)}
    return clean({
        "meshes": {name: S.mesh_info(name)[0] for name in S.mesh_names()},
        "outlines": outlines,
        "fourier": {"curve_max": FOURIER_CURVE_MAX},
        "jaw": {
            "n_shapes": jm.n_shapes,
            "n_landmarks": jm.n_landmarks,
            "n_coordinates": int(jm.model.mean.size),
            "max_modes": S.JAW_MAX_MODES,
            "sample_modes": JAW_SAMPLE_MODES,
            "sample_count": int(len(jm.aligned_sample)),
            "variance_explained": jm.model.explained,
        },
        "spectral": {
            "meshes": list(S.SPECTRAL_MESHES),
            "basis": {name: S.spectral_basis(name) for name in S.SPECTRAL_MESHES},
            "gps_dims": S.GPS_DIMS,
            "restarts": S.SEGMENT_RESTARTS,
            "max_parts": S.SEGMENT_MAX_PARTS,
        },
        "curvature": {"meshes": list(S.CURVATURE_MESHES), "range_percentiles": list(S.CURVATURE_RANGE_PERCENTILES)},
        "skinning": {"rigs": rigs},
    })


# ---------------------------------------------------------------- outlines


@bp.get("/outlines")
@endpoint
def outlines():
    items, cached = [], True
    for name in S.outline_names():
        b, hit = S.load_outline(name)
        cached &= hit
        items.append({"name": name, "n_points": len(b)})
    return {"outlines": items, "_cached": cached}


@bp.get("/fourier")
@endpoint
def fourier():
    name = choice("outline", "bird1", S.outline_names())
    summary, hit1 = S.fourier_summary(name)
    N = len(summary.points)
    keep = num("keep", summary.coefficients_for_10pct, 1, N, integer=True)
    recon, hit2 = S.fourier_reconstruction(name, keep)
    br, abs_err = recon
    top = min(FOURIER_CURVE_MAX, N)
    c10 = summary.coefficients_for_10pct
    doubled = min(2 * keep, top)
    return {
        "outline": name,
        "n_points": N,
        "keep": keep,
        "share": keep / N,
        "error": float(summary.error_curve[keep]),
        "abs_error": abs_err,
        "coefficients_for_10pct": c10,
        "original": summary.points,
        "reconstruction": br,
        "gaps": np.linalg.norm(summary.points - br, axis=1),  # per-point distance original -> rebuilt
        "error_curve": summary.error_curve[: top + 1],
        "spectrum": summary.spectrum[:top],  # relative size of each coefficient, in keep order
        "summary": {
            "share_for_10pct": c10 / N,
            "compression": N / keep,
            "doubled": {"keep": doubled, "error": float(summary.error_curve[doubled])} if doubled > keep else None,
        },
        "_cached": hit1 and hit2,
    }


# ---------------------------------------------------------------- jaw shape model


@bp.get("/jaw-model")
@endpoint
def jaw_model():
    k = num("n_modes", 3, 1, S.JAW_MAX_MODES, integer=True)
    jm, hit = S.jaw_model()
    m = jm.model
    explained = m.explained[:k]
    return {
        "n_shapes": jm.n_shapes,
        "n_landmarks": jm.n_landmarks,
        "n_coordinates": int(m.mean.size),
        "n_modes": k,
        "mean": m.mean,
        "modes": m.modes[:k],
        "sd": np.sqrt(m.variances[:k]),
        "variance_explained": explained,
        "cumulative": np.cumsum(explained),
        "aligned_sample": jm.aligned_sample,
        "_cached": hit,
    }


@bp.get("/jaw-sample")
@endpoint
def jaw_sample():
    b = [num(f"b{i}", 0.0, -3, 3) for i in range(1, JAW_SAMPLE_MODES + 1)]
    jm, hit = S.jaw_model()
    mean = jm.model.mean
    shape = jm.model.sample(b)
    shifts = np.linalg.norm(shape - mean, axis=1)
    j = int(np.argmax(shifts))
    return {
        "shape": shape,
        "b": b,
        "summary": {
            "distance_sd": float(np.linalg.norm(b)),  # Mahalanobis distance from the mean jaw
            "shifts": shifts,  # per landmark, in unit-size jaw units
            "max_shift": {"landmark": j + 1, "value": float(shifts[j]), "share_of_width": float(shifts[j] / np.ptp(mean[:, 0]))},
        },
        "_cached": hit,
    }


# ---------------------------------------------------------------- meshes


@bp.get("/meshes")
@endpoint
def meshes():
    items, cached = [], True
    for name in S.mesh_names():
        (V, F), hit = S.load_mesh(name)
        cached &= hit
        items.append({"name": name, "n_vertices": len(V), "n_faces": len(F)})
    return {"meshes": items, "_cached": cached}


@bp.get("/mesh/<name>")
@endpoint
def mesh(name: str):
    if name not in S.mesh_names():  # whitelist: only the files present, so no path tricks
        raise ApiError(f"Unknown mesh {name!r}; available: {', '.join(S.mesh_names())}.", "not_found", 404)
    (V, F), hit = S.load_mesh(name)
    return {
        "name": name,
        "vertices": _flat(V),
        "faces": _flat(F),
        "n_vertices": len(V),
        "n_faces": len(F),
        "bbox": {"min": V.min(axis=0), "max": V.max(axis=0)},
        "_cached": hit,
    }


# ---------------------------------------------------------------- spectral


def _basis_param(mesh_name: str, default: int, lo: int) -> int:
    k = num("basis", default, lo, S.SPECTRAL_DEFAULT_MAX, integer=True)
    cap = S.spectral_max_basis(mesh_name)
    if k > cap:
        raise ApiError(
            f"Parameter 'basis' is limited to {cap} for mesh '{mesh_name}' to keep the eigensolve fast; got {k}.",
            "out_of_range",
        )
    return k


@bp.get("/spectral")
@endpoint
def spectral():
    name = _mesh_param()
    basis_k = _basis_param(name, S.spectral_basis(name), 10)
    k = num("k", min(30, basis_k), 1, basis_k, integer=True)
    sp, hit = S.eigenbasis(name, basis_k)
    errors = sp.errors
    within = np.flatnonzero(errors <= 0.1)
    n = len(sp.basis.vectors)
    return {
        "mesh": name,
        "n_vertices": n,
        "k": k,
        "basis": basis_k,
        "max_basis": S.spectral_max_basis(name),
        "eigenvalues": sp.basis.values,
        "error": float(errors[k - 1]),
        "error_curve": errors,
        "coefficients_for_10pct": int(within[0]) + 1 if len(within) else None,
        "vertices": _flat(S.spectral_reconstruct(sp, k)),
        "summary": {"share_kept": k / n},  # k x 3 coefficients instead of n x 3 coordinates
        "_cached": hit,
    }


@bp.get("/segment")
@endpoint
def segment():
    name = _mesh_param()
    basis_k = _basis_param(name, S.spectral_basis(name), S.GPS_DIMS + 1)
    parts = num("parts", 5, 2, S.SEGMENT_MAX_PARTS, integer=True)
    labels, hit = S.segmentation(name, basis_k, parts)
    sizes = np.bincount(labels, minlength=parts)
    n = int(sizes.sum())
    shares = sizes / n
    lo = int(np.argmin(sizes))
    return {
        "mesh": name,
        "n_vertices": n,
        "parts": parts,
        "basis": basis_k,
        "gps_dims": S.GPS_DIMS,
        "restarts": S.SEGMENT_RESTARTS,
        "labels": labels,
        "sizes": sizes,
        "shares": shares,
        "summary": {
            "largest": int(sizes.max()),
            "largest_share": float(shares.max()),
            "smallest": int(sizes[lo]),
            "smallest_share": float(shares[lo]),
        },
        "_cached": hit,
    }


# ---------------------------------------------------------------- curvature


def _range(x: np.ndarray) -> list[float]:
    """Robust colour range (2nd/98th percentile) so a few spiky vertices do not wash out the map."""
    lo, hi = np.percentile(x[np.isfinite(x)], S.CURVATURE_RANGE_PERCENTILES)
    return [float(lo), float(hi)]


def _signs(x: np.ndarray) -> dict:
    return {"negative_share": float(np.mean(x < 0)), "positive_share": float(np.mean(x > 0))}


@bp.get("/curvature")
@endpoint
def mesh_curvature():
    name = _mesh_param()
    c, hit = S.mesh_curvature(name)
    ranges = {"gaussian": _range(c.gaussian), "mean": _range(c.mean)}
    expected = 2 * np.pi * c.euler_characteristic
    return {
        "mesh": name,
        "gaussian": c.gaussian,
        "mean": c.mean,
        "range": ranges,
        "range_limit": {f: max(abs(lo), abs(hi)) for f, (lo, hi) in ranges.items()},  # symmetric colour limit
        "range_percentiles": list(S.CURVATURE_RANGE_PERCENTILES),
        "gauss_bonnet": {
            "integral": c.integral,
            "expected": expected,
            "expected_over_pi": 2 * c.euler_characteristic,
            "matches": bool(abs(c.integral - expected) < 1e-6 + 1e-4 * abs(expected)),
            "euler_characteristic": c.euler_characteristic,
            "closed": c.boundary_vertices == 0,
            "boundary_vertices": c.boundary_vertices,
        },
        "summary": {"gaussian": _signs(c.gaussian), "mean": _signs(c.mean)},
        "_cached": hit,
    }


@bp.get("/surface-curvature")
@endpoint
def surface_curvature():
    kind = choice("surface", "saddle", S.SURFACES)
    n = num("n", 41, 11, 81, integer=True)
    extent = num("extent", 1.0, 0.1, 3.0)
    a: tuple[float, ...] = ()
    if kind == "quadratic":
        defaults = (0.0, 0.0, 0.0, 1.0, 0.0, 1.0)
        a = tuple(float(num(f"a{i}", defaults[i], -5, 5)) for i in range(6))
    out, hit = S.surface_curvature(kind, a, n, float(extent))
    return {"surface": kind, "n": n, "extent": extent, **out, "_cached": hit}


# ---------------------------------------------------------------- skinning


@bp.get("/skin")
@endpoint
def skin():
    name = choice("mesh", "cylinder", S.SKIN_MESHES)
    method = choice("method", "dqs", S.SKIN_METHODS)
    angle = float(num("angle", S.SKIN_DEFAULT_ANGLE[name], -360, 360))
    out, hit = S.skin(name, method, angle)
    (V, F, W, handles, _), _ = S.skin_rig(name)
    body = {
        "mesh": name,
        "method": method,
        "angle": angle,
        "n_vertices": len(V),
        "n_faces": len(F),
        "vertices": _flat(out["vertices"]),
        "faces": _flat(F),
        "handles": handles,
        "handle_positions": V[handles],
        "volume_ratio": out["volume_ratio"],
        "pose": S.SKIN_POSE[name],
        "pose_angles": [r * angle for r in S.skin_pose_ratios(name)],  # degrees, per moving handle
        "full_turn": angle != 0 and abs(angle) % 360 == 0,  # both handles end with the same rotation
        "_cached": hit,
    }
    if name == "cylinder":
        # Halfway along, LBS averages R(0) and R(angle): that matrix scales a ring by |cos(angle / 2)|.
        body["lbs_midpoint_scale"] = abs(float(np.cos(np.radians(angle) / 2)))
    for key in ("cross_section_ratio", "mid_cross_section_ratio", "cross_section_profile"):
        if key in out:
            body[key] = out[key]
    return body
