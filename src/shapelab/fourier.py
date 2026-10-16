"""Fourier descriptors of closed 2D curves (Lab 1).

A closed boundary sampled at N points is treated as the complex sequence
c[n] = x[n] + i y[n]. Its DFT coefficients p[k] describe the curve. Keeping
only the lowest frequencies |f| gives a smooth approximation, and by
Parseval's theorem the error of dropping coefficients is

    ||b - b_r||_F^2 = (1/N) * sum over dropped k of |p[k]|^2.
"""

from __future__ import annotations

import numpy as np


def descriptors(b: np.ndarray) -> np.ndarray:
    """DFT coefficients p[k] of the boundary ``b`` (N, 2), unnormalised (numpy convention)."""
    b = np.asarray(b, dtype=np.float64)
    return np.fft.fft(b[:, 0] + 1j * b[:, 1])


def frequency_order(N: int) -> np.ndarray:
    """Coefficient indices from lowest to highest |frequency|: 0, 1, N-1, 2, N-2, ..."""
    order = [0]
    f = 1
    while len(order) < N:
        order.append(f)
        if len(order) < N:
            order.append(N - f)
        f += 1
    return np.array(order)


def coefficient_count(keep: float | int, N: int) -> int:
    if isinstance(keep, (int, np.integer)) and not isinstance(keep, bool):
        K = int(keep)
    elif 0 < keep <= 1:
        K = int(round(keep * N))
    else:
        raise ValueError("keep must be a fraction in (0, 1] or an integer count")
    return min(max(K, 1), N)


def reconstruct(b: np.ndarray, keep: float | int) -> tuple[np.ndarray, float]:
    """Rebuild a boundary from its lowest-frequency Fourier coefficients.

    Parameters
    ----------
    b : (N, 2) boundary points
    keep : fraction of coefficients in (0, 1], or an integer number of coefficients

    Returns
    -------
    br : (N, 2) reconstructed boundary
    err : reconstruction error ||b - br||_F
    """
    b = np.asarray(b, dtype=np.float64)
    N = len(b)
    p = descriptors(b)
    mask = np.zeros(N, dtype=bool)
    mask[frequency_order(N)[: coefficient_count(keep, N)]] = True
    c = np.fft.ifft(np.where(mask, p, 0))
    br = np.stack([c.real, c.imag], axis=1)
    return br, float(np.linalg.norm(b - br))


def error_curve(b: np.ndarray) -> np.ndarray:
    """Relative error ||b - b_K||_F / ||b||_F for every K = 0..N (via Parseval, O(N log N))."""
    b = np.asarray(b, dtype=np.float64)
    N = len(b)
    power = np.abs(descriptors(b)[frequency_order(N)]) ** 2 / N
    tail = np.concatenate([np.cumsum(power[::-1])[::-1], [0.0]])
    return np.sqrt(np.maximum(tail, 0)) / np.linalg.norm(b)


def coefficients_for_error(b: np.ndarray, rel_tol: float = 0.1) -> int:
    """Smallest number of coefficients with relative error <= ``rel_tol``."""
    curve = error_curve(b)
    return int(np.argmax(curve <= rel_tol))
