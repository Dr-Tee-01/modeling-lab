"""PCA by SVD, the same decomposition ShapeLab's shape models use."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PCA:
    mean: np.ndarray          # (d,)
    modes: np.ndarray         # (k, d), unit rows
    sd: np.ndarray            # (k,), standard deviation along each mode
    var_ratio: np.ndarray     # (k,), share of total variance

    def scores(self, X: np.ndarray) -> np.ndarray:
        """Coordinates of rows of X in standard deviations along each mode."""
        with np.errstate(all="ignore"):     # Accelerate BLAS raises spurious matmul warnings
            return (np.asarray(X) - self.mean) @ self.modes.T / self.sd

    def sample(self, b) -> np.ndarray:
        """mean + sum_k b_k sd_k mode_k for b in standard deviations."""
        b = np.asarray(b, dtype=float)
        with np.errstate(all="ignore"):
            return self.mean + (b * self.sd[: len(b)]) @ self.modes[: len(b)]

    def rebuild(self, X: np.ndarray, k: int) -> np.ndarray:
        if k == 0:
            return np.tile(self.mean, (len(X), 1))
        with np.errstate(all="ignore"):
            return self.mean + (self.scores(X)[:, :k] * self.sd[:k]) @ self.modes[:k]


def fit(X: np.ndarray, k: int | None = None, tol: float = 1e-10) -> PCA:
    """PCA of the rows of X. Modes with (numerically) zero variance are dropped."""
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    with np.errstate(all="ignore"):     # Accelerate BLAS raises spurious matmul warnings
        _, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    total = (S**2).sum()
    keep = S > tol * max(S[0], 1e-300)
    if k is not None:
        keep[k:] = False
    S, Vt = S[keep], Vt[keep]
    return PCA(mean=mean, modes=Vt, sd=S / np.sqrt(max(len(X) - 1, 1)), var_ratio=S**2 / total)
