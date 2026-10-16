"""Gaussian GARCH(1,1) with the variance recursion run as a linear IIR filter.

    s2[t] = w + a e[t-1]^2 + b s2[t-1]

is ``s2 - b s2[t-1] = x[t]`` with ``x[t] = w + a e[t-1]^2``: a first-order
recursive filter, so ``scipy.signal.lfilter([1], [1, -b], x, zi=[b s2[0]])``
evaluates it in C. The likelihood, constraints, starting point and optimiser
(Nelder-Mead with stocklab's tolerances) are the same as
``stocklab.models.garch_fit``, so the fitted parameters agree with the Python
loop version to optimiser precision while each likelihood call is ~20x faster.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

from ..server.cache import memo
from .common import panel

LOG_2PI = np.log(2 * np.pi)


def variance_path(e2: np.ndarray, w: float, a: float, b: float, v0: float) -> np.ndarray:
    """s2[0] = v0, s2[t] = w + a e2[t-1] + b s2[t-1]; returns len(e2)+1 values
    (the last one is the forecast for the day after the sample)."""
    x = w + a * e2
    tail = lfilter([1.0], [1.0, -b], x, zi=[b * v0])[0]
    return np.r_[v0, tail]


def nll(theta, r: np.ndarray, v0: float) -> float:
    mu, w, a, b = theta
    if w <= 0 or a < 0 or b < 0 or a + b >= 0.9999:
        return 1e12
    e2 = (r - mu) ** 2
    s2 = variance_path(e2[:-1], w, a, b, v0)
    return 0.5 * float(np.sum(LOG_2PI + np.log(s2) + e2 / s2))


def garch_fit(r) -> np.ndarray:
    """Drop-in for stocklab.models.garch_fit (r in %): returns [mu, w, a, b]."""
    r = np.asarray(r, dtype=float)
    v0 = np.var(r)
    x0 = [r.mean(), v0 * 0.05, 0.08, 0.9]
    res = minimize(nll, x0, args=(r, v0), method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-4})
    return res.x


def garch_filter(r, theta, v0) -> np.ndarray:
    """Drop-in for stocklab.models.garch_filter: forecast for t+1 made at t."""
    mu, w, a, b = theta
    e2 = (np.asarray(r, dtype=float) - mu) ** 2
    return variance_path(e2, w, a, b, v0)[1:]


@memo(maxsize=256)
def garch_year(ticker: str, year: int, window_years: int = 12):
    """Fit on [year-window, year) and filter through the end of `year` (as in
    stocklab.models.garch_forecasts). Returns (Series of variance forecasts in
    return units^2, params [mu, w, a, b]) or None when the year has no data."""
    ret = panel(ticker, with_vix=False)["ret"]
    r = ret * 100
    lo, start, end = (pd.Timestamp(f"{y}-01-01") for y in (year - window_years, year, year + 1))
    idx = r.index
    if ((idx >= start) & (idx < end)).sum() == 0:
        return None
    tr = (idx >= lo) & (idx < start)
    th = garch_fit(r[tr].values)
    span = (idx >= lo) & (idx < end)
    nxt = garch_filter(r[span].values, th, np.var(r[tr].values))
    return pd.Series(nxt / 1e4, index=idx[span]), [float(v) for v in th]
