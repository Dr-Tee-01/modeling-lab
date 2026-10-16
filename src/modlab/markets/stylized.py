"""Stylised facts of daily returns: no linear autocorrelation, persistent
volatility (|r| autocorrelation), fat tails versus the normal distribution."""
from __future__ import annotations

import numpy as np
from scipy import stats

from ..server.cache import memo
from .common import iso, panel


def acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation at lags 1..max_lag (FFT, biased estimator as statsmodels)."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 ** int(np.ceil(np.log2(2 * n))))
    c = np.fft.irfft(f * np.conj(f))[: max_lag + 1]
    return c[1:] / c[0]


@memo(maxsize=64)
def stylized(ticker: str, max_lag: int, bins: int = 81, z_max: float = 8.0) -> dict:
    r = panel(ticker, with_vix=False)["ret"].dropna()
    x = r.values
    n = len(x)
    z = (x - x.mean()) / x.std()
    edges = np.linspace(-z_max, z_max, bins + 1)
    counts, _ = np.histogram(z, edges)
    expected = n * np.diff(stats.norm.cdf(edges))
    return {
        "ticker": ticker,
        "start": iso(r.index[:1])[0], "end": iso(r.index[-1:])[0], "n": n,
        "lags": list(range(1, max_lag + 1)),
        "acf_r": acf(x, max_lag),
        "acf_abs": acf(np.abs(x), max_lag),
        "acf_sq": acf(x ** 2, max_lag),
        "band": 1.96 / np.sqrt(n),
        "kurtosis": float(stats.kurtosis(x)),          # excess kurtosis (normal = 0)
        "skew": float(stats.skew(x)),
        "mean_pct": float(x.mean() * 100), "std_pct": float(x.std() * 100),
        "tail_counts": {"beyond_4sd": int(np.sum(np.abs(z) > 4)),
                        "normal_expected_beyond_4sd": float(n * 2 * stats.norm.sf(4))},
        "hist": {"edges": edges, "counts": counts, "outside": int(np.sum(np.abs(z) >= z_max)),
                 "units": "standard deviations"},
        "normal_expected": expected,
    }
