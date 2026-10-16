"""Classical fourth-order Runge-Kutta with a fixed step.

``rk4(f, y0, t_end, h)`` integrates y' = f(t, y) from t = 0 and returns the
time grid and the solution, one row per step. A fixed step keeps the solver
deterministic and easy to test against closed forms (global error O(h^4)).
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def rk4(f: Callable[[float, np.ndarray], np.ndarray], y0, t_end: float, h: float,
        events: Callable[[float, np.ndarray], np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Integrate y' = f(t, y) on [0, t_end] with step h.

    ``events(t, y)``, if given, is applied after each step and may return a
    modified state (used for instantaneous jumps such as a drug dose).
    """
    if h <= 0 or t_end <= 0:
        raise ValueError("t_end and h must be positive")
    n = int(round(t_end / h))
    y = np.asarray(y0, dtype=float).copy()
    out = np.empty((n + 1, y.size))
    out[0] = y
    t = 0.0
    for i in range(1, n + 1):
        k1 = f(t, y)
        k2 = f(t + h / 2, y + h / 2 * k1)
        k3 = f(t + h / 2, y + h / 2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t = i * h
        if events is not None:
            y = np.asarray(events(t, y), dtype=float)
        out[i] = y
    return np.linspace(0, n * h, n + 1), out


def sample(t: np.ndarray, Y: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Values of the solution at times ``at`` (nearest grid point)."""
    idx = np.clip(np.rint(np.asarray(at) / (t[1] - t[0])).astype(int), 0, len(t) - 1)
    return Y[idx]
