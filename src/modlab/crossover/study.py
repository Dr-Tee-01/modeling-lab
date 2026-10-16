"""Crossover study: market data seen through the MC312 and ShapeLab toolkits.

A. random_walk   Is a stock the i.i.d. random walk of Lab 10? Variance ratios
                 (permutation band + Lo-MacKinlay robust z), tails, memory, and a
                 Monte Carlo distribution of maximum drawdown (Lab 9).
B. vix_*         Does fear clear like a drug dose (Lab 2)? Event study of VIX
                 spikes with one- and two-compartment fits, and a "doses +
                 first-order clearance" reconstruction of the whole VIX series.
C. crash_shapes  Do crashes have a shape? Drawdowns resampled to landmarks and
                 decomposed with PCA, the recipe ShapeLab uses for jawlines.

All functions take their parameters explicitly and compute from the raw CSVs.
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd
from scipy import optimize, signal, stats

from .. import DATA
from ..core import montecarlo, pca

TICKERS = ["SPY", "AAPL", "MSFT", "NVDA", "JPM", "XOM"]


@functools.lru_cache(maxsize=16)
def load(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"prices/{ticker}.csv", index_col="Date", parse_dates=True)
    return df.sort_index()


def log_returns(ticker: str) -> pd.Series:
    return np.log(load(ticker)["AdjClose"]).diff().dropna()


# --------------------------------------------------------------------------- A
def variance_ratio(r: np.ndarray, k: int) -> float:
    s = np.convolve(r, np.ones(k), "valid")
    return float(s.var(ddof=1) / (k * r.var(ddof=1)))


def lo_mackinlay(x: np.ndarray, q: int) -> tuple[float, float]:
    """Variance ratio and its heteroskedasticity-robust z (Lo and MacKinlay 1988)."""
    e = x - x.mean()
    s2 = (e**2).sum() ** 2
    theta = sum((2 * (q - j) / q) ** 2 * (e[j:] ** 2 * e[:-j] ** 2).sum() / s2 for j in range(1, q))
    vr = variance_ratio(x, q)
    return vr, float((vr - 1) / np.sqrt(theta))


def acf(v: np.ndarray, lags: int) -> np.ndarray:
    v = v - v.mean()
    d = (v * v).sum()
    return np.array([(v[:-L] * v[L:]).sum() / d for L in range(1, lags + 1)])


def random_walk(ticker: str = "SPY", sims: int = 2000, perms: int = 200, seed: int = 312) -> dict:
    g = montecarlo.rng(seed)
    r = log_returns(ticker)
    x = r.to_numpy()
    n, mu, sd = len(x), float(x.mean()), float(x.std(ddof=1))

    ks = [1, 2, 3, 5, 10, 15, 21, 42, 63]
    vr = [variance_ratio(x, k) for k in ks]
    shuffled = np.array([[variance_ratio(g.permutation(x), k) for k in ks] for _ in range(perms)])
    lo, hi = np.percentile(shuffled, [2.5, 97.5], axis=0)

    ewma = pd.Series(x**2).ewm(alpha=0.06, adjust=False).mean().shift(1).to_numpy()
    xs = (x / np.sqrt(ewma))[63:]                     # each day / the volatility known the evening before
    robust = []
    for k in [2, 5, 10, 21, 63]:
        v1, z1 = lo_mackinlay(x, k)
        v2, z2 = lo_mackinlay(xs, k)
        robust.append({"k": k, "vr": v1, "z": z1, "vr_scaled": v2, "z_scaled": z2})
    decades = []
    for a, b in [("1993", "2002"), ("2003", "2012"), ("2013", "2026")]:
        m = (r.index >= a) & (r.index <= f"{b}-12-31")
        v1, z1 = lo_mackinlay(x[m], 10)
        decades.append({"span": f"{a}-{b}", "vr10": v1, "z10": z1})

    z = (x - mu) / sd
    zs = (xs - xs.mean()) / xs.std(ddof=1)
    p4 = 2 * stats.norm.sf(4)
    edges = np.linspace(-8, 8, 65)
    kurt_h = []
    for k in [1, 5, 10, 21, 63]:
        m = (n // k) * k
        s = x[:m].reshape(-1, k).sum(axis=1)
        kurt_h.append({"k": k, "kurtosis": float(stats.kurtosis(s, fisher=False)), "count": len(s)})

    # Monte Carlo maximum drawdown over the same horizon, in batches to bound memory.
    real = float(montecarlo.max_drawdown(np.log(load(ticker)["AdjClose"].to_numpy())))
    mdd_n, mdd_b = [], []
    for start in range(0, sims, 250):
        b = min(250, sims - start)
        mdd_n.append(montecarlo.max_drawdown(montecarlo.walks(g, b, n, mu, sd)))
        mdd_b.append(montecarlo.max_drawdown(montecarlo.bootstrap_walks(g, x, b, n)))
    mdd_n, mdd_b = np.concatenate(mdd_n), np.concatenate(mdd_b)
    bins = np.linspace(0, 1, 51)

    return {
        "ticker": ticker, "n": n, "mu": mu, "sigma": sd,
        "start": str(r.index[0].date()), "end": str(r.index[-1].date()),
        "variance_ratio": [{"k": k, "vr": v, "lo": a, "hi": b} for k, v, a, b in zip(ks, vr, lo, hi)],
        "robust": robust, "decades": decades,
        "acf_r": acf(x, 100), "acf_abs": acf(np.abs(x), 100), "acf_band": 1.96 / np.sqrt(n),
        "tails": {
            "beyond4": int((np.abs(z) > 4).sum()), "expected4": n * p4,
            "beyond4_scaled": int((np.abs(zs) > 4).sum()), "expected4_scaled": len(zs) * p4,
            "kurtosis": float(stats.kurtosis(z, fisher=False)), "kurtosis_scaled": float(stats.kurtosis(zs, fisher=False)),
            "worst_day": str(r.idxmin().date()), "worst_return": float(np.expm1(x.min())), "worst_z": float(z.min()),
        },
        "hist": {"edges": edges, "raw": np.histogram(z, edges)[0], "scaled": np.histogram(zs, edges)[0],
                 "n_raw": len(z), "n_scaled": len(zs)},
        "kurtosis_by_horizon": kurt_h,
        "drawdown": {
            "real": real, "sims": sims,
            "share_worse_normal": float((mdd_n >= real).mean()), "share_worse_bootstrap": float((mdd_b >= real).mean()),
            "median_normal": float(np.median(mdd_n)), "median_bootstrap": float(np.median(mdd_b)),
            "bins": bins, "hist_normal": np.histogram(mdd_n, bins)[0], "hist_bootstrap": np.histogram(mdd_b, bins)[0],
        },
    }


def walk_paths(ticker: str = "SPY", n_paths: int = 20, seed: int | None = None) -> dict:
    """Weekly log price of the ticker next to random walks with the same weekly drift and spread."""
    wk = np.log(load(ticker)["AdjClose"]).resample("W-FRI").last().dropna()
    steps = wk.diff().dropna().to_numpy()
    paths = montecarlo.walks(montecarlo.rng(seed), n_paths, len(steps), float(steps.mean()), float(steps.std(ddof=1)))
    ends = np.exp(paths[:, -1])
    return {"ticker": ticker, "dates": [d.strftime("%Y-%m-%d") for d in wk.index], "real": wk.to_numpy() - wk.iloc[0],
            "walks": paths, "weekly_mu": float(steps.mean()), "weekly_sigma": float(steps.std(ddof=1)),
            "real_multiple": float(np.exp(wk.iloc[-1] - wk.iloc[0])), "walk_multiples": np.sort(ends)}


# --------------------------------------------------------------------------- B
def vix() -> pd.Series:
    v = load("VIX")["Close"]
    return v[v.index >= "1993-01-01"]


def find_spikes(v: pd.Series, ratio: float, gap: int, horizon: int, base_win: int = 63) -> list[tuple]:
    """Peaks where the VIX is at least `ratio` times its prior median, `gap` days apart."""
    base = v.rolling(base_win).median().shift(1)
    rel = v / base
    peaks: list = []
    for t in rel[rel >= ratio].index:
        i = v.index.get_loc(t)
        top = v.iloc[i:i + 21].idxmax()                    # walk to the local top within a month
        if peaks and v.index.get_loc(top) - v.index.get_loc(peaks[-1]) < gap:
            if v[top] > v[peaks[-1]]:
                peaks[-1] = top
            continue
        peaks.append(top)
    out = []
    for p in sorted(set(peaks)):
        i = v.index.get_loc(p)
        b = base.iloc[i]
        if i + horizon + 1 <= len(v) and np.isfinite(b) and v.iloc[i] > b:
            out.append((p, float(b)))
    return out


def vix_events(ratio: float = 1.6, gap: int = 63, horizon: int = 60) -> dict:
    v = vix()
    spikes = find_spikes(v, ratio, gap, horizon)
    if len(spikes) < 3:
        raise ValueError(f"Only {len(spikes)} spikes match; lower the ratio or the gap.")
    curves, events = [], []
    for p, b in spikes:
        i = v.index.get_loc(p)
        seg = v.iloc[i:i + horizon + 1].to_numpy()
        frac = (seg - b) / (seg[0] - b)                   # share of the "dose" still in the blood
        curves.append(frac)
        below = np.nonzero(frac <= 0.5)[0]
        events.append({"date": str(p.date()), "peak": float(seg[0]), "base": b,
                       "days_to_half": int(below[0]) if len(below) else None})
    C = np.array(curves)
    h = np.arange(horizon + 1)
    med = np.median(C, axis=0)
    q25, q75 = np.percentile(C, [25, 75], axis=0)

    one = lambda t, k: np.exp(-k * t)
    two = lambda t, a, kf, ks: a * np.exp(-kf * t) + (1 - a) * np.exp(-ks * t)
    (k1,), _ = optimize.curve_fit(one, h, med, p0=[0.05], bounds=([1e-4], [5]))
    (a2, kf, ks), _ = optimize.curve_fit(two, h, med, p0=[0.5, 0.2, 0.02], bounds=([0, 0, 0], [1, 5, 1]))
    if kf < ks:
        a2, kf, ks = 1 - a2, ks, kf
    sst = float(((med - med.mean()) ** 2).sum())
    r2 = lambda f: 1 - float(((med - f) ** 2).sum()) / sst
    vv = v.to_numpy()
    phi, c = np.polyfit(vv[:-1], vv[1:], 1)
    return {
        "params": {"ratio": ratio, "gap": gap, "horizon": horizon},
        "events": events, "h": h, "median": med, "q25": q25, "q75": q75,
        "one": {"k": k1, "half_life": np.log(2) / k1, "r2": r2(one(h, k1)), "curve": one(h, k1)},
        "two": {"a": a2, "k_fast": kf, "k_slow": ks, "hl_fast": np.log(2) / kf,
                "hl_slow": np.log(2) / ks if ks > 0 else None, "r2": r2(two(h, a2, kf, ks)), "curve": two(h, a2, kf, ks)},
        "ar1": {"phi": phi, "half_life": np.log(2) / -np.log(phi), "base": c / (1 - phi)},
    }


def _clearance(doses: np.ndarray, decay: float, e0: float) -> np.ndarray:
    """e[t] = decay * e[t-1] + doses[t], e[0] = e0: a first-order IIR filter."""
    zi = signal.lfiltic([1.0], [1.0, -decay], y=[e0])
    return signal.lfilter([1.0], [1.0, -decay], doses, zi=zi)[0]


def vix_dose_model(half_life: float | None = 12.5, threshold: float = 3.4, compartments: int = 1,
                   fit: bool = False, a: float = 0.77, hl_fast: float = 5.9, hl_slow: float = 44.0) -> dict:
    """VIX = baseline + what is left of past doses. A dose is a daily jump above
    `threshold`; between doses the excess clears at first order (one compartment)
    or split between a fast and a slow compartment. With fit=True the clearance
    parameters are chosen by least squares for the given threshold."""
    v = vix()
    x = v.to_numpy()
    base = float(np.median(x))
    jumps = np.r_[0.0, np.diff(x)]
    doses = np.where(jumps > threshold, jumps, 0.0)
    e0 = x[0] - base
    sst = ((x - x.mean()) ** 2).sum()

    def model(params):
        if compartments == 1:
            return base + _clearance(doses, np.exp(-np.log(2) / params[0]), e0)
        aa, hf, hs = params
        return base + _clearance(aa * doses, np.exp(-np.log(2) / hf), 0.0) + _clearance((1 - aa) * doses, np.exp(-np.log(2) / hs), e0)

    if compartments == 1:
        params = [half_life]
        if fit:
            res = optimize.minimize_scalar(lambda hl: ((x - model([hl])) ** 2).sum(), bounds=(0.5, 120), method="bounded")
            params = [float(res.x)]
    else:
        params = [a, hl_fast, hl_slow]
        if fit:
            def loss(q):
                aa, hf, hs = q
                if not (0 <= aa <= 1 and 0.2 < hf < hs < 400):
                    return 1e18
                return ((x - model(q)) ** 2).sum()
            res = optimize.minimize(loss, [0.6, 3.0, 40.0], method="Nelder-Mead",
                                    options={"maxiter": 600, "xatol": 1e-4, "fatol": 1e-3})
            params = [float(p) for p in res.x]
    m = model(params)
    out = {"compartments": compartments, "threshold": threshold, "baseline": base,
           "n_doses": int((doses > 0).sum()), "n_days": len(x),
           "r2": float(1 - ((x - m) ** 2).sum() / sst), "fitted": fit}
    if compartments == 1:
        out["half_life"] = params[0]
    else:
        out.update(a=params[0], hl_fast=params[1], hl_slow=params[2])
    out["windows"] = {}
    for name, (lo, hi) in {"2008": ("2007-06-01", "2010-01-01"), "2020": ("2019-10-01", "2021-01-01"),
                           "2025": ("2024-10-01", "2026-12-31")}.items():
        w = (v.index >= lo) & (v.index < hi)
        out["windows"][name] = float(1 - ((x[w] - m[w]) ** 2).sum() / ((x[w] - x[w].mean()) ** 2).sum())
    out["_series"] = (v.index, x, m)
    return out


# --------------------------------------------------------------------------- C
def drawdown_episodes(p: pd.Series, min_depth: float, min_days: int) -> list[tuple]:
    """Completed peak -> trough -> recovery episodes of a price series."""
    vals, idx = p.to_numpy(), p.index
    eps, i, n, peak_i = [], 0, len(vals), 0
    while i < n:
        if vals[i] >= vals[peak_i]:
            peak_i, i = i, i + 1
            continue
        j = i
        while j < n and vals[j] < vals[peak_i]:
            j += 1
        if j >= n:
            break                                     # still under water: not a completed shape
        seg = vals[peak_i:j + 1]
        depth = 1 - seg.min() / vals[peak_i]
        if depth >= min_depth and (j - peak_i) >= min_days:
            eps.append((peak_i, peak_i + int(seg.argmin()), j, depth))
        peak_i, i = j, j + 1
    return [(idx[a], idx[b], idx[c], d) for a, b, c, d in eps]


def crash_shapes(spy_depth: float = 0.10, stock_depth: float = 0.20, landmarks: int = 48,
                 min_days: int = 15, tickers: tuple = tuple(TICKERS)) -> dict:
    s = np.linspace(0, 1, landmarks)
    rows, X = [], []
    for tk in tickers:
        p = load(tk)["AdjClose"]
        for t0, tm, t1, d in drawdown_episodes(p, spy_depth if tk == "SPY" else stock_depth, min_days):
            seg = p.loc[t0:t1].to_numpy()
            y = (seg / seg[0] - 1) / d                 # remove level and depth: peak 0, trough -1
            X.append(np.interp(s, np.linspace(0, 1, len(seg)), y))
            rows.append({"ticker": tk, "peak": str(t0.date()), "trough": str(tm.date()), "recovered": str(t1.date()),
                         "depth": float(d), "days": int(len(seg) - 1),
                         "trough_at": float((p.index.get_loc(tm) - p.index.get_loc(t0)) / (len(seg) - 1))})
    if len(X) < 6:
        raise ValueError(f"Only {len(X)} completed crashes match; lower the depth thresholds.")
    X = np.array(X)
    model = pca.fit(X, k=8)
    trough_at = np.array([r["trough_at"] for r in rows])
    # Orient the modes: +mode 1 = higher curve (a sharp V), +mode 2 = later trough.
    if model.modes[0].sum() < 0:
        model.modes[0] *= -1
    if len(model.modes) > 1 and np.corrcoef(model.scores(X)[:, 1], trough_at)[0, 1] < 0:
        model.modes[1] *= -1
    scores = model.scores(X)
    for r, sc, x in zip(rows, scores, X):
        r["scores"], r["curve"] = sc[:4], x
    depth = np.array([r["depth"] for r in rows])
    return {
        "params": {"spy_depth": spy_depth, "stock_depth": stock_depth, "landmarks": landmarks, "min_days": min_days},
        "s": s, "mean": model.mean, "modes": model.modes[:4], "sd": model.sd[:4],
        "variance": model.var_ratio[:8], "cumulative": np.cumsum(model.var_ratio)[:8],
        "rebuild_rmse": [float(np.sqrt(((X - model.rebuild(X, k)) ** 2).mean())) for k in range(0, len(model.sd) + 1)],
        "episodes": rows,
        "correlations": {
            "mode1_depth": float(np.corrcoef(scores[:, 0], depth)[0, 1]),
            "mode1_trough_at": float(np.corrcoef(scores[:, 0], trough_at)[0, 1]),
            "mode2_trough_at": float(np.corrcoef(scores[:, 1], trough_at)[0, 1]),
        },
        "_model": model,
    }
