"""Procrustes shape analysis and PCA shape models (Lab 2).

Shapes are arrays of m corresponding landmarks in d dimensions, shape (m, d).
Collections are (n, m, d). "Shape" means what is left after removing
translation, scale and rotation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def normalize(shape: np.ndarray) -> np.ndarray:
    """Move the centroid to the origin and scale to unit Frobenius norm."""
    s = np.asarray(shape, dtype=np.float64)
    s = s - s.mean(axis=-2, keepdims=True)
    return s / np.linalg.norm(s, axis=(-2, -1), keepdims=True)


def optimal_rotation(ref: np.ndarray, shape: np.ndarray, allow_reflection: bool = False):
    """Orthogonal R minimising ||shape @ R - ref||_F (Kabsch / orthogonal Procrustes)."""
    U, _, Vt = np.linalg.svd(shape.T @ ref)
    if not allow_reflection and np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1
    return U @ Vt


def align(ref: np.ndarray, shapes: np.ndarray) -> np.ndarray:
    """Align a collection (n, m, d) to ``ref`` (m, d).

    Every shape is centred, scaled to unit size and rotated onto the
    normalised reference.
    """
    ref = normalize(ref)
    shapes = normalize(shapes)
    return np.stack([s @ optimal_rotation(ref, s) for s in shapes])


def procrustes_mean(
    shapes: np.ndarray, tol: float = 1e-12, max_iter: int = 200, init: int = 0
) -> np.ndarray:
    """Generalised Procrustes mean of a collection (n, m, d).

    1. start from one shape, 2. align every original shape to the current mean,
    3. average and renormalise, 4. repeat until the mean stops moving.
    """
    shapes = normalize(shapes)
    mean = shapes[init]
    for _ in range(max_iter):
        new = normalize(align(mean, shapes).mean(axis=0))
        new = new @ optimal_rotation(mean, new)  # fix the gauge so iterations can converge
        moved = np.sum((new - mean) ** 2)
        mean = new
        if moved < tol:
            break
    return mean


@dataclass
class ShapeModel:
    """Linear model x = mean + sum_i b_i * sqrt(variance_i) * mode_i."""

    mean: np.ndarray  # (m, d)
    modes: np.ndarray  # (k, m, d), orthonormal when flattened
    variances: np.ndarray  # (k,), descending
    explained: np.ndarray  # (k,), fraction of total variance per mode

    def sample(self, b: np.ndarray) -> np.ndarray:
        """Shape at mode coordinates ``b`` given in standard deviations."""
        b = np.asarray(b, dtype=np.float64)
        k = len(b)
        coeff = b * np.sqrt(self.variances[:k])
        return self.mean + np.tensordot(coeff, self.modes[:k], axes=1)

    def project(self, shape: np.ndarray, k: int | None = None) -> np.ndarray:
        """Mode coordinates (in standard deviations) of a shape after aligning it to the mean."""
        k = len(self.variances) if k is None else k
        s = align(self.mean, shape[None])[0] - self.mean
        coeff = self.modes[:k].reshape(k, -1) @ s.ravel()
        return coeff / np.sqrt(self.variances[:k])


def fit_shape_model(shapes: np.ndarray, n_modes: int | None = None) -> ShapeModel:
    """Procrustes-align a collection and run PCA on the aligned coordinates."""
    shapes = np.asarray(shapes, dtype=np.float64)
    n, m, d = shapes.shape
    mean = procrustes_mean(shapes)
    X = (align(mean, shapes) - mean).reshape(n, m * d)
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    variances = S**2 / (n - 1)
    k = len(variances) if n_modes is None else n_modes
    return ShapeModel(
        mean=mean,
        modes=Vt[:k].reshape(k, m, d),
        variances=variances[:k],
        explained=variances[:k] / variances.sum(),
    )
