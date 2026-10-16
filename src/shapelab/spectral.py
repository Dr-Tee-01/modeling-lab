"""Laplace-Beltrami spectral analysis of triangle meshes (Lab 4).

The discrete Laplace-Beltrami operator is the cotangent Laplacian L
(negative semi-definite, libigl convention) with a lumped mass matrix M.
Its eigenfunctions solve the generalised problem

    -L phi_i = lambda_i M phi_i,   0 = lambda_0 <= lambda_1 <= ...

and are M-orthonormal: phi_i^T M phi_j = delta_ij. They act as a Fourier
basis on the surface: projecting the xyz coordinate functions onto the first
k eigenfunctions gives a smooth approximation of the shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.cluster.vq import kmeans2
from scipy.sparse.linalg import eigsh


def _corner_cotangents(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """(m, 3) cotangent of the interior angle at each corner of each face."""
    cot = np.empty(F.shape)
    for k in range(3):
        a, b, c = F[:, k], F[:, (k + 1) % 3], F[:, (k + 2) % 3]
        e1, e2 = V[b] - V[a], V[c] - V[a]
        cross = np.linalg.norm(np.cross(e1, e2), axis=1)
        cot[:, k] = np.einsum("ij,ij->i", e1, e2) / np.maximum(cross, 1e-300)
    return cot


def face_areas(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    return 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)


def cotan_laplacian(V: np.ndarray, F: np.ndarray) -> sp.csr_matrix:
    """Cotangent Laplacian: L_ij = (cot alpha_ij + cot beta_ij) / 2, L_ii = -sum_j L_ij."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    n = len(V)
    cot = _corner_cotangents(V, F)
    # the angle at corner k is opposite the edge (k+1, k+2)
    i = np.concatenate([F[:, (k + 1) % 3] for k in range(3)])
    j = np.concatenate([F[:, (k + 2) % 3] for k in range(3)])
    w = 0.5 * np.concatenate([cot[:, k] for k in range(3)])
    W = sp.coo_matrix((w, (i, j)), shape=(n, n))
    W = (W + W.T).tocsr()
    return (W - sp.diags(np.asarray(W.sum(axis=1)).ravel())).tocsr()


def mass_matrix(V: np.ndarray, F: np.ndarray, kind: str = "voronoi") -> sp.dia_matrix:
    """Lumped (diagonal) mass matrix.

    kind="barycentric": each vertex gets a third of every adjacent face's area.
    kind="voronoi": mixed Voronoi areas (Meyer et al. 2003, libigl's default):
        non-obtuse faces split by the circumcentre, obtuse faces give half their
        area to the obtuse corner and a quarter to the other two.
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    n = len(V)
    area = face_areas(V, F)
    if kind == "barycentric":
        corner = np.repeat(area[:, None] / 3, 3, axis=1)
    elif kind == "voronoi":
        cot = _corner_cotangents(V, F)
        sq = np.stack(
            [np.sum((V[F[:, (k + 1) % 3]] - V[F[:, (k + 2) % 3]]) ** 2, axis=1) for k in range(3)],
            axis=1,
        )  # squared length of the edge opposite corner k
        # Voronoi area of corner k: (|e_k,k+1|^2 cot(k+2) + |e_k,k+2|^2 cot(k+1)) / 8
        w = sq * cot  # edge opposite corner j, weighted by the cotangent at j
        corner = (np.roll(w, -1, axis=1) + np.roll(w, -2, axis=1)) / 8
        obtuse = cot < 0
        any_obtuse = obtuse.any(axis=1)
        corner[any_obtuse] = np.where(
            obtuse[any_obtuse], area[any_obtuse, None] / 2, area[any_obtuse, None] / 4
        )
    else:
        raise ValueError("kind must be 'voronoi' or 'barycentric'")
    m = np.zeros(n)
    for k in range(3):
        np.add.at(m, F[:, k], corner[:, k])
    return sp.diags(m)


@dataclass
class Eigenbasis:
    """First k Laplace-Beltrami eigenpairs of a mesh."""

    values: np.ndarray  # (k,) ascending, values[0] ~ 0
    vectors: np.ndarray  # (n, k) M-orthonormal columns
    mass: np.ndarray  # (n,) lumped vertex areas

    @property
    def k(self) -> int:
        return len(self.values)

    def project(self, f: np.ndarray, k: int | None = None) -> np.ndarray:
        """Spectral coefficients phi^T M f of a scalar (n,) or vector (n, d) function."""
        Phi = self.vectors[:, : (k or self.k)]
        f = np.asarray(f, dtype=np.float64)
        return Phi.T @ (self.mass[:, None] * f if f.ndim == 2 else self.mass * f)

    def reconstruct(self, f: np.ndarray, k: int) -> np.ndarray:
        """Projection of f onto the span of the first k eigenfunctions."""
        return self.vectors[:, :k] @ self.project(f, k)


def eigenbasis(V: np.ndarray, F: np.ndarray, k: int = 100, mass: str = "voronoi") -> Eigenbasis:
    """Smallest k eigenpairs of -L phi = lambda M phi via shift-invert Lanczos.

    Eigenvector signs are fixed so the entry of largest magnitude is positive,
    which makes results reproducible across runs and platforms.
    """
    L = cotan_laplacian(V, F)
    M = mass_matrix(V, F, mass).tocsc()
    A = (-L).tocsc()
    # a tiny negative shift keeps A - sigma M positive definite (A itself is singular)
    sigma = -1e-8 * float(A.diagonal().mean() / M.diagonal().mean())
    k = min(k, len(V) - 1)
    vals, vecs = eigsh(A, k=k, M=M, sigma=sigma, which="LM")
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    flip = np.sign(vecs[np.abs(vecs).argmax(axis=0), np.arange(k)])
    vecs = vecs * flip
    return Eigenbasis(values=np.maximum(vals, 0.0), vectors=vecs, mass=M.diagonal().copy())


def reconstruction_error(V: np.ndarray, basis: Eigenbasis, k: int) -> float:
    """Relative M-weighted error of the k-term reconstruction of the coordinates."""
    m = basis.mass[:, None]
    centred = V - (m * V).sum(0) / m.sum()
    diff = basis.reconstruct(V, k) - V
    return float(np.sqrt((m * diff**2).sum() / (m * centred**2).sum()))


def reconstruction_errors(V: np.ndarray, basis: Eigenbasis) -> np.ndarray:
    """``reconstruction_error`` for every k = 1..basis.k in O(n k).

    The eigenvectors are M-orthonormal, so by Pythagoras
    ||V - P_k V||_M^2 = ||V||_M^2 - sum_{i<k} |c_i|^2 with c = Phi^T M V.
    """
    m = basis.mass[:, None]
    total = (m * V**2).sum()
    centred = (m * (V - (m * V).sum(0) / m.sum()) ** 2).sum()
    captured = np.cumsum((basis.project(V) ** 2).sum(axis=1))
    return np.sqrt(np.maximum(total - captured, 0) / centred)


def gps_embedding(basis: Eigenbasis, n: int = 20) -> np.ndarray:
    """Global Point Signature: GPS(x) = (phi_1(x)/sqrt(lambda_1), ..., phi_n(x)/sqrt(lambda_n))."""
    if n + 1 > basis.k:
        raise ValueError(f"need {n + 1} eigenpairs, basis has {basis.k}")
    return basis.vectors[:, 1 : n + 1] / np.sqrt(basis.values[1 : n + 1])


def segment(
    basis: Eigenbasis, parts: int, n: int = 20, restarts: int = 10, seed: int = 0
) -> np.ndarray:
    """Segment a mesh by k-means clustering of its GPS embedding.

    Runs k-means++ ``restarts`` times and keeps the lowest-inertia result.
    Labels are renumbered by cluster size (0 = largest) for stable colouring.
    """
    X = gps_embedding(basis, n)
    best, best_inertia = None, np.inf
    rng = np.random.default_rng(seed)
    for _ in range(restarts):
        cent, labels = kmeans2(X, parts, minit="++", seed=rng)
        inertia = np.sum((X - cent[labels]) ** 2)
        if inertia < best_inertia:
            best, best_inertia = labels, inertia
    sizes = np.bincount(best, minlength=parts)
    rank = np.empty(parts, dtype=np.int64)
    rank[np.argsort(-sizes, kind="stable")] = np.arange(parts)
    return rank[best]
