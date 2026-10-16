"""Least-squares fitting with bounds, reporting fit quality alongside the parameters."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy import optimize


def least_squares(residuals: Callable[[np.ndarray], np.ndarray], x0: Sequence[float],
                  lo: Sequence[float], hi: Sequence[float]) -> dict:
    """Bounded nonlinear least squares (trust region reflective).

    Returns the parameters, RMSE of the residuals and the evaluation count.
    """
    res = optimize.least_squares(residuals, np.asarray(x0, float), bounds=(lo, hi), method="trf", x_scale="jac")
    r = res.fun
    return {"x": res.x, "rmse": float(np.sqrt(np.mean(r**2))), "evaluations": int(res.nfev), "success": bool(res.success)}


def r2(y: np.ndarray, f: np.ndarray) -> float:
    y, f = np.asarray(y, float), np.asarray(f, float)
    return float(1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum())
