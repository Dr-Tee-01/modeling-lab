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
from .api import ApiError, choice, endpoint, num

bp = Blueprint("shape", __name__, url_prefix="/api/shape")

FOURIER_CURVE_MAX = 200
JAW_SAMPLE_MODES = 6


def _flat(a: np.ndarray) -> np.ndarray:
    return np.asarray(a).reshape(-1)


def _mesh_param(name: str = "mesh", default: str = "cat4") -> str:
    return choice(name, default, S.mesh_names())


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
    return {
        "outline": name,
        "n_points": N,
        "keep": keep,
        "share": keep / N,
        "error": float(summary.error_curve[keep]),
        "abs_error": abs_err,
        "coefficients_for_10pct": summary.coefficients_for_10pct,
        "original": summary.points,
        "reconstruction": br,
        "error_curve": summary.error_curve[: min(FOURIER_CURVE_MAX, N) + 1],
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
    return {"shape": jm.model.sample(b), "b": b, "_cached": hit}


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
    basis_k = _basis_param(name, 120, 10)
    k = num("k", min(30, basis_k), 1, basis_k, integer=True)
    sp, hit = S.eigenbasis(name, basis_k)
    errors = sp.errors
    within = np.flatnonzero(errors <= 0.1)
    return {
        "mesh": name,
        "k": k,
        "basis": basis_k,
        "max_basis": S.spectral_max_basis(name),
        "eigenvalues": sp.basis.values,
        "error": float(errors[k - 1]),
        "error_curve": errors,
        "coefficients_for_10pct": int(within[0]) + 1 if len(within) else None,
        "vertices": _flat(S.spectral_reconstruct(sp, k)),
        "_cached": hit,
    }


@bp.get("/segment")
@endpoint
def segment():
    name = _mesh_param()
    basis_k = _basis_param(name, 120, S.GPS_DIMS + 1)
    parts = num("parts", 5, 2, 10, integer=True)
    labels, hit = S.segmentation(name, basis_k, parts)
    return {
        "mesh": name,
        "parts": parts,
        "basis": basis_k,
        "gps_dims": S.GPS_DIMS,
        "labels": labels,
        "sizes": np.bincount(labels, minlength=parts),
        "_cached": hit,
    }


# ---------------------------------------------------------------- curvature


def _range(x: np.ndarray) -> list[float]:
    """Robust colour range (2nd/98th percentile) so a few spiky vertices do not wash out the map."""
    lo, hi = np.percentile(x[np.isfinite(x)], [2, 98])
    return [float(lo), float(hi)]


@bp.get("/curvature")
@endpoint
def mesh_curvature():
    name = _mesh_param()
    c, hit = S.mesh_curvature(name)
    return {
        "mesh": name,
        "gaussian": c.gaussian,
        "mean": c.mean,
        "range": {"gaussian": _range(c.gaussian), "mean": _range(c.mean)},
        "gauss_bonnet": {
            "integral": c.integral,
            "expected": 2 * np.pi * c.euler_characteristic,
            "euler_characteristic": c.euler_characteristic,
            "closed": c.boundary_vertices == 0,
            "boundary_vertices": c.boundary_vertices,
        },
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
        "_cached": hit,
    }
    for key in ("cross_section_ratio", "mid_cross_section_ratio", "cross_section_profile"):
        if key in out:
            body[key] = out[key]
    return body
