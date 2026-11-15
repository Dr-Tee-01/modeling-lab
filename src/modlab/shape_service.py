"""Pure, memoised shape computations behind /api/shape.

Everything here is computed from the raw files in ``data/shape`` on first use
and kept in memory (``modlab.server.cache.memo``); nothing is precomputed on
disk. Each memoised function returns ``(value, was_cached)``. Cached arrays are
marked read-only so a caller cannot corrupt the cache by mutating them.

The geometry itself lives in :mod:`shapelab`; this module only wires the data
files to it and derives the few extra numbers the API reports.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shapelab import curvature, fourier, io, procrustes, skinning, spectral

from . import DATA
from .server.cache import memo

SHAPE_DIR = DATA / "shape"
OUTLINE_DIR = SHAPE_DIR / "outlines"
MESH_DIR = SHAPE_DIR / "meshes"
WEIGHTS_DIR = SHAPE_DIR / "weights"
LANDMARK_DIR = SHAPE_DIR / "landmarks"

# Display names for data files whose stem is not a readable name; others are derived from the stem.
MESH_LABELS = {"cat4": "Cat"}

# Largest eigenbasis per mesh (bunny: 14k vertices, ~2.6 s for 150 eigenpairs).
SPECTRAL_MAX_BASIS = {"bunny": 150}
SPECTRAL_DEFAULT_MAX = 300
# Basis the interface uses per mesh (and the API default): the bunny keeps 100 so its first solve stays near one second.
SPECTRAL_BASIS = {"bunny": 100}
SPECTRAL_BASIS_DEFAULT = 120
SPECTRAL_MESHES = ("cat4", "bunny")
GPS_DIMS = 20  # segment() uses phi_1..phi_20, so the basis needs >= 21 eigenpairs
SEGMENT_RESTARTS = 10  # k-means++ starts; the lowest-inertia result is kept
SEGMENT_MAX_PARTS = 10

CURVATURE_MESHES = ("cat4", "bunny", "cactus")
CURVATURE_RANGE_PERCENTILES = (2, 98)

JAW_MAX_MODES = 10
JAW_SAMPLE_COUNT = 24

# Cactus pose from the ShapeLab README: arm handles 1 and 3 bend about x by
# 0.31 and 0.26 rad. The API's ``angle`` is the bend of handle 1 in degrees and
# handle 3 keeps the README's 0.26/0.31 proportion.
CACTUS_AXES = [[0, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]
CACTUS_PROPORTIONS = np.array([0.0, 1.0, 0.0, 0.26 / 0.31])
CYLINDER_AXIS = [0.0, 0.0, 1.0]
SKIN_MESHES = ("cylinder", "cactus")
SKIN_POSE = {"cylinder": "twist", "cactus": "bend"}
SKIN_DEFAULT_ANGLE = {"cylinder": 180.0, "cactus": float(np.degrees(0.31))}


def _ro(*arrays: np.ndarray) -> None:
    for a in arrays:
        a.setflags(write=False)


class _Quiet:
    """numpy 2.0 + Accelerate raises spurious FP warnings from matmul; results are checked."""

    def __enter__(self):
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)
        self._e = np.errstate(all="ignore")
        self._e.__enter__()
        return self

    def __exit__(self, *exc):
        self._e.__exit__(*exc)
        self._w.__exit__(*exc)
        return False


quiet = _Quiet


# ---------------------------------------------------------------- catalogue


def _names(directory: Path, suffix: str) -> list[str]:
    return sorted(p.stem for p in directory.glob(f"*{suffix}") if p.is_file())


def outline_names() -> list[str]:
    return _names(OUTLINE_DIR, ".csv")


def mesh_names() -> list[str]:
    return _names(MESH_DIR, ".off")


def mesh_label(name: str) -> str:
    return MESH_LABELS.get(name, name.capitalize())


def outline_label(name: str) -> str:
    """bird1 -> "Bird 1"."""
    m = re.fullmatch(r"(\D+?)(\d+)", name)
    return f"{m[1].capitalize()} {m[2]}" if m else name.capitalize()


# ---------------------------------------------------------------- outlines (Fourier)


@memo(maxsize=16)
def load_outline(name: str) -> np.ndarray:
    if name not in outline_names():
        raise KeyError(name)
    b = io.load_outline(OUTLINE_DIR / f"{name}.csv")
    _ro(b)
    return b


@dataclass(frozen=True)
class FourierSummary:
    points: np.ndarray
    error_curve: np.ndarray  # relative error for K = 0..N
    spectrum: np.ndarray  # |Z_k| / (sqrt(N) ||b||) in keep order (0, +1, -1, +2, ...)
    coefficients_for_10pct: int


@memo(maxsize=16)
def fourier_summary(name: str) -> FourierSummary:
    b, _ = load_outline(name)
    curve = fourier.error_curve(b)
    N = len(b)
    spectrum = np.abs(fourier.descriptors(b)[fourier.frequency_order(N)]) / (np.sqrt(N) * np.linalg.norm(b))
    _ro(curve, spectrum)
    return FourierSummary(b, curve, spectrum, fourier.coefficients_for_error(b, 0.1))


@memo(maxsize=256)
def fourier_reconstruction(name: str, keep: int) -> tuple[np.ndarray, float]:
    b, _ = load_outline(name)
    br, abs_err = fourier.reconstruct(b, int(keep))
    _ro(br)
    return br, abs_err


# ---------------------------------------------------------------- jaw shape model


@memo(maxsize=2)
def jaw_landmarks() -> np.ndarray:
    jaws = io.load_landmarks(LANDMARK_DIR / "jaw_x.npy", LANDMARK_DIR / "jaw_y.npy")
    _ro(jaws)
    return jaws


@dataclass(frozen=True)
class JawModel:
    model: procrustes.ShapeModel
    aligned_sample: np.ndarray  # (JAW_SAMPLE_COUNT, m, 2)
    n_shapes: int
    n_landmarks: int


@memo(maxsize=2)
def jaw_model() -> JawModel:
    """Generalised Procrustes + PCA on all jawlines, keeping JAW_MAX_MODES modes.

    ``explained`` is relative to the total variance of all modes, so slicing the
    first k modes gives exactly what ``fit_shape_model(jaws, n_modes=k)`` would.
    """
    jaws, _ = jaw_landmarks()
    model = procrustes.fit_shape_model(jaws, n_modes=JAW_MAX_MODES)
    idx = np.linspace(0, len(jaws) - 1, JAW_SAMPLE_COUNT).round().astype(int)
    aligned = procrustes.align(model.mean, jaws[idx])
    _ro(model.mean, model.modes, model.variances, model.explained, aligned)
    return JawModel(model, aligned, int(jaws.shape[0]), int(jaws.shape[1]))


# ---------------------------------------------------------------- meshes


@memo(maxsize=8)
def load_mesh(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name not in mesh_names():
        raise KeyError(name)
    V, F = io.read_off(MESH_DIR / f"{name}.off")
    _ro(V, F)
    return V, F


@memo(maxsize=8)
def mesh_info(name: str) -> dict:
    """Size and topology of a mesh: what the interface lists before it loads the mesh itself."""
    (V, F), _ = load_mesh(name)
    nb = int(curvature.boundary_vertices(F, len(V)).sum())
    return {
        "label": mesh_label(name),
        "n_vertices": int(len(V)),
        "n_faces": int(len(F)),
        "closed": nb == 0,
        "boundary_vertices": nb,
        "euler_characteristic": euler_characteristic(V, F),
    }


def euler_characteristic(V: np.ndarray, F: np.ndarray) -> int:
    E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
    used = len(np.unique(F))  # isolated vertices do not belong to the surface
    return int(used - len(E) + len(F))


def signed_volume(V: np.ndarray, F: np.ndarray) -> float:
    """Enclosed volume of a closed, consistently oriented mesh (divergence theorem)."""
    return float(np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum() / 6.0)


@dataclass(frozen=True)
class MeshCurvature:
    gaussian: np.ndarray
    mean: np.ndarray
    integral: float
    euler_characteristic: int
    boundary_vertices: int


@memo(maxsize=8)
def mesh_curvature(name: str) -> MeshCurvature:
    (V, F), _ = load_mesh(name)
    with quiet():
        c = curvature.mesh_curvature(V, F)
        integral = curvature.total_gaussian_curvature(V, F)
    nb = int(curvature.boundary_vertices(F, len(V)).sum())
    _ro(c.K, c.H)
    return MeshCurvature(c.K, c.H, integral, euler_characteristic(V, F), nb)


# ---------------------------------------------------------------- spectral


def spectral_max_basis(name: str) -> int:
    (V, _), _ = load_mesh(name)
    return min(SPECTRAL_MAX_BASIS.get(name, SPECTRAL_DEFAULT_MAX), len(V) - 1)


def spectral_basis(name: str) -> int:
    """The basis size the interface uses for this mesh (also the API default)."""
    return min(SPECTRAL_BASIS.get(name, SPECTRAL_BASIS_DEFAULT), spectral_max_basis(name))


@dataclass(frozen=True)
class Spectral:
    basis: spectral.Eigenbasis
    coeffs: np.ndarray  # (basis, 3) spectral coefficients of the xyz coordinates
    errors: np.ndarray  # relative error for k = 1..basis


@memo(maxsize=8)
def eigenbasis(name: str, k: int) -> Spectral:
    """Laplace-Beltrami eigenbasis per (mesh, basis size): the expensive step."""
    (V, F), _ = load_mesh(name)
    with quiet():
        basis = spectral.eigenbasis(V, F, k=int(k))
        coeffs = basis.project(V)
        errors = spectral.reconstruction_errors(V, basis)
    if not (np.all(np.isfinite(basis.vectors)) and np.all(np.isfinite(errors))):
        raise FloatingPointError("eigenbasis is not finite")
    _ro(basis.values, basis.vectors, basis.mass, coeffs, errors)
    return Spectral(basis, coeffs, errors)


def spectral_reconstruct(sp: Spectral, k: int) -> np.ndarray:
    """First k terms of the projection, reusing the cached coefficients (O(n k))."""
    with quiet():
        return sp.basis.vectors[:, :k] @ sp.coeffs[:k]


@memo(maxsize=64)
def segmentation(name: str, basis_k: int, parts: int) -> np.ndarray:
    sp, _ = eigenbasis(name, basis_k)
    with quiet():
        labels = spectral.segment(sp.basis, int(parts), n=GPS_DIMS, restarts=SEGMENT_RESTARTS)
    _ro(labels)
    return labels


# ---------------------------------------------------------------- parametric surfaces

SURFACES = ("saddle", "monkey", "paraboloid", "quadratic")


def make_surface(kind: str, a: tuple[float, ...]) -> tuple[curvature.PolynomialSurface, tuple[float, ...], str]:
    """(surface, lab coefficients a0..a5 or (), formula). Monkey saddle is cubic."""
    if kind == "saddle":
        a = (0.0, 0.0, 0.0, 1.0, 0.0, -1.0)
        return curvature.PolynomialSurface.from_lab(a), a, "z = u^2 - v^2"
    if kind == "paraboloid":
        a = (0.0, 0.0, 0.0, 1.0, 0.0, 1.0)
        return curvature.PolynomialSurface.from_lab(a), a, "z = u^2 + v^2"
    if kind == "monkey":
        return curvature.PolynomialSurface.monkey_saddle(), (), "z = u^3 - 3 u v^2"
    if kind == "quadratic":
        formula = "z = a0 + a1 u + a2 v + a3 u^2 + a4 u v + a5 v^2"
        return curvature.PolynomialSurface.from_lab(a), tuple(a), formula
    raise ValueError(kind)


def closed_form_K(kind: str, a: tuple[float, ...], U: np.ndarray, Vg: np.ndarray) -> tuple[np.ndarray, str]:
    """Gaussian curvature of z = f(u, v) written out by hand, K = (f_uu f_vv - f_uv^2) / (1 + f_u^2 + f_v^2)^2."""
    if kind == "monkey":
        r2 = U**2 + Vg**2
        return -36 * r2 / (1 + 9 * r2**2) ** 2, "K = -36 (u^2+v^2) / (1 + 9 (u^2+v^2)^2)^2"
    _, a1, a2, a3, a4, a5 = a
    fu = a1 + 2 * a3 * U + a4 * Vg
    fv = a2 + a4 * U + 2 * a5 * Vg
    return (4 * a3 * a5 - a4**2) / (1 + fu**2 + fv**2) ** 2, "K = (4 a3 a5 - a4^2) / (1 + f_u^2 + f_v^2)^2"


@memo(maxsize=64)
def surface_curvature(kind: str, a: tuple[float, ...], n: int, extent: float) -> dict:
    surf, a_used, formula = make_surface(kind, a)
    U, Vg, Z = surf.sample(n, extent)
    c = surf.curvature(U, Vg)
    o = surf.curvature(0.0, 0.0)
    Kc, k_formula = closed_form_K(kind, a_used, U, Vg)
    Kc0, _ = closed_form_K(kind, a_used, np.zeros(()), np.zeros(()))
    t = np.linspace(-extent, extent, n)
    return {
        "formula": formula,
        "coefficients": list(a_used) if a_used else None,
        "u": t,
        "v": t,
        "z": Z,
        "K": c.K,
        "H": c.H,
        "at_origin": {
            "K": float(o.K),
            "H": float(o.H),
            "k1": float(o.k1),
            "k2": float(o.k2),
            "type": surf.classify(0.0, 0.0),
        },
        "K_abs_max": float(np.max(np.abs(c.K))),
        "closed_form": {
            "formula": k_formula,
            "K": Kc,
            "K_at_origin": float(Kc0),
            "max_abs_diff": float(np.max(np.abs(Kc - c.K))),
        },
    }


# ---------------------------------------------------------------- skinning

SKIN_METHODS = ("lbs", "nlerp", "slerp", "dqs")


@memo(maxsize=4)
def skin_rig(name: str):
    """Mesh, BBW weights (0-based handles via io.load_weights_mat) and rest measurements."""
    if name not in SKIN_MESHES:
        raise KeyError(name)
    (V, F), _ = load_mesh(name)
    W, handles = io.load_weights_mat(WEIGHTS_DIR / f"BBW_{name}.mat")
    if W.shape != (len(V), len(handles)):
        raise ValueError(f"weights {W.shape} do not match mesh {name} ({len(V)} vertices)")
    _ro(W, handles)
    rest = {"volume": signed_volume(V, F)}
    if name == "cylinder":
        r = np.linalg.norm(V[:, :2], axis=1)
        ring_v = r > 1e-9  # the two cap-centre (handle) vertices sit on the axis
        z = np.round(V[:, 2], 9)
        zs = np.unique(z[ring_v])
        rings = [np.flatnonzero(ring_v & (z == zz)) for zz in zs]
        rest.update(
            ring_z=zs,
            rings=rings,
            ring_radius=np.array([r[i].mean() for i in rings]),
            mid=int(np.argmin(np.abs(zs - 0.5 * (V[:, 2].min() + V[:, 2].max())))),
        )
    return V, F, W, handles, rest


def skin_pose(name: str, V: np.ndarray, handles: np.ndarray, angle_deg: float):
    theta = np.radians(angle_deg)
    if name == "cylinder":
        # handle 0 = bottom cap centre (fixed), handle 1 = top cap centre (twisted about the tube axis)
        axes = [CYLINDER_AXIS] * len(handles)
        angles = np.zeros(len(handles))
        angles[int(np.argmax(V[handles, 2]))] = theta
        return skinning.handle_transforms(V, handles, axes=axes, angles=angles)
    return skinning.handle_transforms(V, handles, axes=CACTUS_AXES, angles=CACTUS_PROPORTIONS * theta)


def skin_pose_ratios(name: str) -> list[float]:
    """Angle of each moving handle relative to the API's ``angle`` (cylinder: the top cap; cactus: arm 1, arm 2)."""
    if name == "cylinder":
        return [1.0]
    return [float(r) for r in CACTUS_PROPORTIONS if r != 0]


@memo(maxsize=64)
def skin(name: str, method: str, angle_deg: float) -> dict:
    (V, F, W, handles, rest), _ = skin_rig(name)
    R, t = skin_pose(name, V, handles, angle_deg)
    with quiet():
        D = skinning.deform(V, W, R, t, method)
    if not np.all(np.isfinite(D)):
        raise FloatingPointError("deformation is not finite")
    out = {
        "vertices": D,
        "handles": handles,
        "volume_ratio": signed_volume(D, F) / rest["volume"],
    }
    if name == "cylinder":
        radius = np.array([np.linalg.norm(D[i, :2], axis=1).mean() for i in rest["rings"]]) / rest["ring_radius"]
        out.update(
            cross_section_ratio=float(radius.min()),
            mid_cross_section_ratio=float(radius[rest["mid"]]),
            cross_section_profile={"z": rest["ring_z"], "ratio": radius},
        )
    return out
