"""Differential geometry of surfaces (Lab 3).

For a graph patch sigma(u, v) = (u, v, f(u, v)):

    sigma_u = (1, 0, f_u),  sigma_v = (0, 1, f_v),  n = (-f_u, -f_v, 1) / W,
    W = sqrt(1 + f_u^2 + f_v^2)
    first form   E = 1 + f_u^2,  F = f_u f_v,  G = 1 + f_v^2
    second form  L = f_uu / W,   M = f_uv / W,  N = f_vv / W
    K = (LN - M^2) / (EG - F^2),  H = (EN - 2FM + GL) / (2 (EG - F^2))
    kappa_1,2 = H +- sqrt(H^2 - K)

Discrete versions for triangle meshes use the angle defect (Gaussian) and
the cotangent Laplacian of the coordinates (mean).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spectral import cotan_laplacian, mass_matrix


@dataclass
class Curvature:
    k1: np.ndarray  # larger principal curvature
    k2: np.ndarray  # smaller principal curvature
    K: np.ndarray  # Gaussian curvature
    H: np.ndarray  # mean curvature


@dataclass
class FundamentalForms:
    E: np.ndarray
    F: np.ndarray
    G: np.ndarray
    L: np.ndarray
    M: np.ndarray
    N: np.ndarray

    def curvature(self) -> Curvature:
        det = self.E * self.G - self.F**2
        K = (self.L * self.N - self.M**2) / det
        H = (self.E * self.N - 2 * self.F * self.M + self.G * self.L) / (2 * det)
        disc = np.sqrt(np.maximum(H**2 - K, 0))
        return Curvature(k1=H + disc, k2=H - disc, K=K, H=H)


def graph_forms(fu, fv, fuu, fuv, fvv) -> FundamentalForms:
    """First and second fundamental forms of z = f(u, v) from its partial derivatives."""
    fu, fv, fuu, fuv, fvv = map(np.asarray, (fu, fv, fuu, fuv, fvv))
    W = np.sqrt(1 + fu**2 + fv**2)
    return FundamentalForms(E=1 + fu**2, F=fu * fv, G=1 + fv**2, L=fuu / W, M=fuv / W, N=fvv / W)


def classify(K: float, H: float, tol: float = 1e-9) -> str:
    """Point type from Gaussian and mean curvature."""
    if K > tol:
        return "elliptic"
    if K < -tol:
        return "hyperbolic"
    if abs(H) > tol:
        return "parabolic"
    return "planar"


class PolynomialSurface:
    """Graph surface z = sum c_ij u^i v^j with exact derivatives.

    >>> s = PolynomialSurface.from_lab([0, 0, 0, 1, 0, 1])   # z = u^2 + v^2
    >>> s.classify(0, 0)
    'elliptic'
    """

    def __init__(self, terms: dict[tuple[int, int], float]):
        self.terms = {k: float(c) for k, c in terms.items() if c != 0}

    @classmethod
    def from_lab(cls, a) -> PolynomialSurface:
        """Lab 3 parameterisation f = a0 + a1 u + a2 v + a3 u^2 + a4 uv + a5 v^2."""
        exps = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
        return cls(dict(zip(exps, a)))

    @classmethod
    def monkey_saddle(cls, scale: float = 1.0) -> PolynomialSurface:
        """z = u^3 - 3 u v^2: kappa_1 = kappa_2 = 0 at the origin but not a plane."""
        return cls({(3, 0): scale, (1, 2): -3 * scale})

    def _eval(self, u, v, du: int, dv: int):
        u, v = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
        out = np.zeros(np.broadcast(u, v).shape)
        for (i, j), c in self.terms.items():
            if i < du or j < dv:
                continue
            coef = c
            for t in range(du):
                coef *= i - t
            for t in range(dv):
                coef *= j - t
            out = out + coef * u ** (i - du) * v ** (j - dv)
        return out

    def f(self, u, v):
        return self._eval(u, v, 0, 0)

    def derivatives(self, u, v):
        """(f_u, f_v, f_uu, f_uv, f_vv) at (u, v)."""
        return tuple(self._eval(u, v, a, b) for a, b in [(1, 0), (0, 1), (2, 0), (1, 1), (0, 2)])

    def forms(self, u, v) -> FundamentalForms:
        return graph_forms(*self.derivatives(u, v))

    def curvature(self, u, v) -> Curvature:
        return self.forms(u, v).curvature()

    def tangent_frame(self, u: float, v: float):
        """sigma_u, sigma_v and unit normal at (u, v)."""
        fu, fv, *_ = self.derivatives(u, v)
        su, sv = np.array([1.0, 0.0, float(fu)]), np.array([0.0, 1.0, float(fv)])
        n = np.cross(su, sv)
        return su, sv, n / np.linalg.norm(n)

    def classify(self, u: float, v: float, tol: float = 1e-9) -> str:
        c = self.curvature(u, v)
        return classify(float(c.K), float(c.H), tol)

    def sample(self, n: int = 64, extent: float = 1.0):
        """Grid (U, V, Z) over [-extent, extent]^2 for plotting."""
        t = np.linspace(-extent, extent, n)
        U, V = np.meshgrid(t, t, indexing="ij")
        return U, V, self.f(U, V)


# ---------------------------------------------------------------- meshes


def boundary_vertices(F: np.ndarray, n: int | None = None) -> np.ndarray:
    """Boolean mask of vertices on a boundary edge (edge used by only one face)."""
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    uniq, counts = np.unique(E, axis=0, return_counts=True)
    n = int(F.max()) + 1 if n is None else n
    mask = np.zeros(n, dtype=bool)
    mask[uniq[counts == 1].ravel()] = True
    return mask


def vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Area-weighted unit vertex normals (orientation follows the face winding)."""
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-300)


def mesh_curvature(V: np.ndarray, F: np.ndarray) -> Curvature:
    """Discrete curvatures per vertex.

    K: angle defect divided by the Voronoi area (pi instead of 2 pi on boundaries).
    H: half the length of the discrete Laplace-Beltrami of the coordinates,
       signed positive where the surface curves away from the normal (sphere: H = 1/r).
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    n = len(V)
    angle_sum = np.zeros(n)
    for k in range(3):
        a, b, c = F[:, k], F[:, (k + 1) % 3], F[:, (k + 2) % 3]
        e1, e2 = V[b] - V[a], V[c] - V[a]
        cosang = np.einsum("ij,ij->i", e1, e2) / (
            np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)
        )
        np.add.at(angle_sum, a, np.arccos(np.clip(cosang, -1, 1)))
    area = mass_matrix(V, F, "voronoi").diagonal()
    full = np.where(boundary_vertices(F, n), np.pi, 2 * np.pi)
    K = (full - angle_sum) / area
    lap = cotan_laplacian(V, F) @ V / area[:, None]  # ~ Laplace-Beltrami of x = -2 H n
    H = -0.5 * np.einsum("ij,ij->i", lap, vertex_normals(V, F))
    disc = np.sqrt(np.maximum(H**2 - K, 0))
    return Curvature(k1=H + disc, k2=H - disc, K=K, H=H)


def total_gaussian_curvature(V: np.ndarray, F: np.ndarray) -> float:
    """Integral of K over the mesh; equals 2 pi chi for closed meshes (Gauss-Bonnet)."""
    K = mesh_curvature(V, F).K
    return float(np.sum(K * mass_matrix(V, F, "voronoi").diagonal()))


__all__ = [
    "Curvature",
    "FundamentalForms",
    "PolynomialSurface",
    "boundary_vertices",
    "classify",
    "graph_forms",
    "mesh_curvature",
    "total_gaussian_curvature",
    "vertex_normals",
]
