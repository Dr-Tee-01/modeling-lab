"""Crossover study: the market data seen through the MC312 and ShapeLab toolkits.

Three questions, each answered with a model from one of the other two projects:

A. Random walk (MC312 Labs 9-10). Are daily SPY returns the i.i.d. random walk
   of Lab 10? Variance ratios, tails, autocorrelation, and a Monte Carlo
   distribution of maximum drawdown under the random-walk null.
B. Dosing (MC312 Lab 2). Does fear (the VIX) clear like a drug dose? Event study
   of VIX spikes, one- and two-compartment elimination fits, and a
   "doses + first-order clearance" reconstruction of the whole VIX series.
C. Shape modelling (ShapeLab). Do market crashes have a shape? Drawdown episodes
   resampled to landmarks, aligned, and decomposed with PCA, as ShapeLab does
   for jawlines.

Run:  python crossover/crossover.py      (writes results/crossover.json)
"""
import json
import pathlib

import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
TICKERS = ["SPY", "AAPL", "MSFT", "NVDA", "JPM", "XOM"]
RNG = np.random.default_rng(312)


def load(ticker):
    df = pd.read_csv(ROOT / f"data/prices/{ticker}.csv", index_col="Date", parse_dates=True)
    return df.sort_index()


def r4(x):
    """Round floats for compact JSON."""
    if isinstance(x, (list, tuple, np.ndarray)):
        return [r4(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else float(f"{x:.5g}")
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


# --------------------------------------------------------------------------- A
def max_drawdown(logp):
    """Max drawdown of each row of a (paths, days) array of log prices."""
    peak = np.maximum.accumulate(logp, axis=-1)
    return 1 - np.exp((logp - peak).min(axis=-1))


def variance_ratio(r, k):
    s = np.convolve(r, np.ones(k), "valid")
    return s.var(ddof=1) / (k * r.var(ddof=1))


def lo_mackinlay(x, q):
    """Variance ratio and its heteroskedasticity-robust z (Lo and MacKinlay 1988)."""
    e = x - x.mean()
    s2 = (e**2).sum() ** 2
    theta = sum((2 * (q - j) / q) ** 2 * (e[j:] ** 2 * e[:-j] ** 2).sum() / s2 for j in range(1, q))
    vr = variance_ratio(x, q)
    return vr, (vr - 1) / np.sqrt(theta)


def study_random_walk():
    spy = load("SPY")["AdjClose"]
    r = np.log(spy).diff().dropna()
    x = r.to_numpy()
    n, mu, sd = len(x), x.mean(), x.std(ddof=1)

    # Variance ratios. A random walk gives 1 at every horizon; the band comes from
    # shuffling the real returns, which keeps their distribution but kills any
    # time dependence (a permutation Monte Carlo, as in Lab 9).
    ks = [1, 2, 3, 5, 10, 15, 21, 42, 63]
    vr = [variance_ratio(x, k) for k in ks]
    sims = np.array([[variance_ratio(RNG.permutation(x), k) for k in ks] for _ in range(300)])
    vr_lo, vr_hi = np.percentile(sims, [2.5, 97.5], axis=0)

    # The permutation band assumes i.i.d. steps. Volatility clustering widens the
    # true band, so also report the heteroskedasticity-robust z, on raw returns,
    # on EWMA-scaled returns, and by decade.
    ewma = pd.Series(x**2).ewm(alpha=0.06, adjust=False).mean().shift(1).to_numpy()
    xs = (x / np.sqrt(ewma))[63:]
    robust = []
    for k in [2, 5, 10, 21, 63]:
        vr_k, z_k = lo_mackinlay(x, k)
        vs_k, zs_k = lo_mackinlay(xs, k)
        robust.append({"k": k, "vr": vr_k, "z": z_k, "vr_scaled": vs_k, "z_scaled": zs_k})
    decades = []
    for a, b in [("1993", "2002"), ("2003", "2012"), ("2013", "2026")]:
        m = (r.index >= a) & (r.index <= f"{b}-12-31")
        vr_k, z_k = lo_mackinlay(x[m], 10)
        decades.append({"span": f"{a}-{b}", "vr10": vr_k, "z10": z_k})

    # Autocorrelations of r and |r|.
    def acf(v, lags):
        v = v - v.mean()
        d = (v * v).sum()
        return [float((v[:-L] * v[L:]).sum() / d) for L in lags]

    lags = list(range(1, 101))
    acf_r, acf_abs = acf(x, lags), acf(np.abs(x), lags)
    band = 1.96 / np.sqrt(n)

    # Tails. Standardise once with the full-sample sd (the Lab 10 assumption of
    # a fixed step size), and once with an EWMA vol known the day before.
    z = (x - mu) / sd
    zs = (xs - xs.mean()) / xs.std(ddof=1)
    p4 = 2 * stats.norm.sf(4)
    tails = {
        "n": n,
        "obs4": int((np.abs(z) > 4).sum()),
        "exp4": n * p4,
        "obs4_scaled": int((np.abs(zs) > 4).sum()),
        "exp4_scaled": len(zs) * p4,
        "kurt": float(stats.kurtosis(z, fisher=False)),
        "kurt_scaled": float(stats.kurtosis(zs, fisher=False)),
        "worst_day": str(r.idxmin().date()),
        "worst_z": float(z.min()),
        "worst_ret": float(np.expm1(x.min())),
    }
    edges = np.linspace(-8, 8, 65)
    hist = {
        "edges": r4(edges),
        "raw": np.histogram(z, edges)[0].tolist(),
        "scaled": np.histogram(zs, edges)[0].tolist(),
        "n_raw": len(z),
        "n_scaled": len(zs),
    }

    # Kurtosis by horizon (non-overlapping sums). The CLT of Lab 10 says it
    # should fall to 3 as steps add up.
    kurt_h = []
    for k in [1, 5, 10, 21, 63]:
        m = (n // k) * k
        s = x[:m].reshape(-1, k).sum(axis=1)
        kurt_h.append({"k": k, "kurt": float(stats.kurtosis(s, fisher=False)), "count": len(s)})

    # Monte Carlo of maximum drawdown over the same 33 years.
    P = 2000
    normal = mu + sd * RNG.standard_normal((P, n))
    boot = RNG.choice(x, size=(P, n), replace=True)
    mdd_n = max_drawdown(np.cumsum(normal, axis=1))
    mdd_b = max_drawdown(np.cumsum(boot, axis=1))
    real = float(max_drawdown(np.log(spy.to_numpy())))
    bins = np.linspace(0, 0.9, 46)
    mdd = {
        "real": real,
        "pct_normal": float((mdd_n >= real).mean()),
        "pct_boot": float((mdd_b >= real).mean()),
        "median_normal": float(np.median(mdd_n)),
        "median_boot": float(np.median(mdd_b)),
        "bins": r4(bins),
        "hist_normal": np.histogram(mdd_n, bins)[0].tolist(),
        "hist_boot": np.histogram(mdd_b, bins)[0].tolist(),
    }

    # Weekly SPY log price for the browser, where the visitor draws random walks
    # with the same mu and sigma next to it.
    wk = np.log(spy).resample("W-FRI").last().dropna()
    return {
        "mu": mu, "sigma": sd, "n": n,
        "start": str(r.index[0].date()), "end": str(r.index[-1].date()),
        "vr": [{"k": k, "vr": v, "lo": lo, "hi": hi} for k, v, lo, hi in zip(ks, vr, vr_lo, vr_hi)],
        "vr_robust": robust, "vr_decades": decades,
        "acf_r": r4(acf_r), "acf_abs": r4(acf_abs), "acf_band": band,
        "acf_abs_sum100": float(np.sum(acf_abs)),
        "tails": tails, "hist": hist, "kurt_h": kurt_h, "mdd": mdd,
        "spy_weekly": {"t0": str(wk.index[0].date()), "logp": r4(wk.to_numpy() - wk.iloc[0])},
    }


# --------------------------------------------------------------------------- B
def find_spikes(vix, ratio=1.6, gap=63, base_win=63):
    """Peaks where the VIX is at least `ratio` times its prior 63-day median."""
    base = vix.rolling(base_win).median().shift(1)
    rel = vix / base
    cand = rel[rel >= ratio].index
    peaks = []
    for t in cand:
        i = vix.index.get_loc(t)
        w = vix.iloc[i:i + 21]                     # walk to the local top within a month
        top = w.idxmax()
        if peaks and vix.index.get_loc(top) - vix.index.get_loc(peaks[-1]) < gap:
            if vix[top] > vix[peaks[-1]]:
                peaks[-1] = top
            continue
        peaks.append(top)
    out = []
    for p in sorted(set(peaks)):
        i = vix.index.get_loc(p)
        b = base.iloc[vix.index.get_loc(p)]
        if i + 61 > len(vix) or not np.isfinite(b):
            continue
        out.append((p, float(b)))
    return out


def study_dosing():
    vix = load("VIX")["Close"]
    vix = vix[vix.index >= "1993-01-01"]

    spikes = find_spikes(vix)
    H = 60
    curves, events = [], []
    for p, b in spikes:
        i = vix.index.get_loc(p)
        seg = vix.iloc[i:i + H + 1].to_numpy()
        frac = (seg - b) / (seg[0] - b)            # fraction of the "dose" still in the blood
        curves.append(frac)
        below = np.where(frac <= 0.5)[0]
        events.append({
            "date": str(p.date()), "peak": float(seg[0]), "base": b,
            "half_life": int(below[0]) if len(below) else None,
        })
    C = np.array(curves)
    h = np.arange(H + 1)
    med = np.median(C, axis=0)
    q25, q75 = np.percentile(C, [25, 75], axis=0)

    one = lambda t, k: np.exp(-k * t)                                  # one compartment
    two = lambda t, a, ka, kb: a * np.exp(-ka * t) + (1 - a) * np.exp(-kb * t)  # two compartments
    (k1,), _ = optimize.curve_fit(one, h, med, p0=[0.05])
    (a2, ka2, kb2), _ = optimize.curve_fit(two, h, med, p0=[0.5, 0.2, 0.02],
                                           bounds=([0, 0, 0], [1, 5, 1]))
    if ka2 < kb2:
        a2, ka2, kb2 = 1 - a2, kb2, ka2
    sse = lambda f: float(((med - f) ** 2).sum())
    sst = float(((med - med.mean()) ** 2).sum())

    # AR(1) on the whole VIX series = one compartment with a constant infusion.
    v = vix.to_numpy()
    phi, c = np.polyfit(v[:-1], v[1:], 1)
    base_ar = c / (1 - phi)

    # Dose reconstruction: only large daily jumps are "doses", everything else
    # is first-order clearance toward the baseline. Same recursion the browser runs.
    resid = v[1:] - (c + phi * v[:-1])
    thr = 2.0 * resid.std()

    def simulate(k, thr, base):
        decay = np.exp(-k)
        e = np.zeros_like(v)
        e[0] = v[0] - base
        for t in range(1, len(v)):
            jump = v[t] - v[t - 1]
            dose = jump if jump > thr else 0.0
            e[t] = e[t - 1] * decay + dose
        return base + e

    base_med = float(np.median(v))
    def r2(k):
        m = simulate(k, thr, base_med)
        return 1 - ((v - m) ** 2).sum() / ((v - v.mean()) ** 2).sum()
    best = optimize.minimize_scalar(lambda k: -r2(k), bounds=(0.005, 0.3), method="bounded")
    k_best = float(best.x)
    n_doses = int((np.diff(v) > thr).sum())

    # Same doses, but each splits between a fast and a slow compartment (the
    # shape the event study found). Dose rule and baseline stay fixed.
    def simulate2(a, kf, ks):
        df, dsl = np.exp(-kf), np.exp(-ks)
        ef, es = 0.0, v[0] - base_med
        out = np.empty_like(v)
        out[0] = v[0]
        for t in range(1, len(v)):
            jump = v[t] - v[t - 1]
            dose = jump if jump > thr else 0.0
            ef, es = ef * df + a * dose, es * dsl + (1 - a) * dose
            out[t] = base_med + ef + es
        return out

    def r2_two(q):
        a, kf, ks = q
        if not (0 <= a <= 1 and 0 < ks < kf < 3):
            return -1e9
        m = simulate2(a, kf, ks)
        return 1 - ((v - m) ** 2).sum() / ((v - v.mean()) ** 2).sum()
    fit2 = optimize.minimize(lambda q: -r2_two(q), [0.5, 0.3, 0.03], method="Nelder-Mead",
                             options={"maxiter": 400, "xatol": 1e-4, "fatol": 1e-6})
    a_r, kf_r, ks_r = fit2.x

    return {
        "dates_t0": str(vix.index[0].date()),
        "dates": [d.strftime("%Y-%m-%d") for d in vix.index[::5]],   # every 5th day, for axis labels
        "vix": r4(v),
        "events": events,
        "curve_median": r4(med), "curve_q25": r4(q25), "curve_q75": r4(q75),
        "one": {"k": float(k1), "half_life": float(np.log(2) / k1), "r2": 1 - sse(one(h, k1)) / sst},
        "two": {"a": float(a2), "k_fast": float(ka2), "k_slow": float(kb2),
                "hl_fast": float(np.log(2) / ka2), "hl_slow": float(np.log(2) / kb2),
                "r2": 1 - sse(two(h, a2, ka2, kb2)) / sst},
        "ar1": {"phi": float(phi), "half_life": float(np.log(2) / -np.log(phi)), "base": float(base_ar)},
        "recon": {"k": k_best, "half_life": float(np.log(2) / k_best), "threshold": float(thr),
                  "base": base_med, "r2": float(r2(k_best)), "n_doses": n_doses, "n_days": len(v)},
        "recon2": {"a": float(a_r), "k_fast": float(kf_r), "k_slow": float(ks_r),
                   "hl_fast": float(np.log(2) / kf_r), "hl_slow": float(np.log(2) / ks_r),
                   "r2": float(-fit2.fun)},
    }


# --------------------------------------------------------------------------- C
def drawdown_episodes(p, min_depth, min_days=15):
    """Completed peak -> trough -> recovery episodes of a price series."""
    vals, idx = p.to_numpy(), p.index
    eps, i, n = [], 0, len(vals)
    peak_i = 0
    while i < n:
        if vals[i] >= vals[peak_i]:
            peak_i = i
            i += 1
            continue
        j = i
        while j < n and vals[j] < vals[peak_i]:
            j += 1
        if j >= n:
            break                                   # still under water: not a completed shape
        seg = vals[peak_i:j + 1]
        depth = 1 - seg.min() / vals[peak_i]
        if depth >= min_depth and (j - peak_i) >= min_days:
            eps.append((peak_i, peak_i + int(seg.argmin()), j, depth))
        peak_i, i = j, j + 1
    return [(idx[a], idx[b], idx[c], d) for a, b, c, d in eps]


def study_shapes(L=48):
    depth_rule = {"SPY": 0.10}
    rows, X = [], []
    s = np.linspace(0, 1, L)
    for tk in TICKERS:
        p = load(tk)["AdjClose"]
        for t0, tm, t1, d in drawdown_episodes(p, depth_rule.get(tk, 0.20)):
            seg = p.loc[t0:t1].to_numpy()
            tt = np.linspace(0, 1, len(seg))
            y = (seg / seg[0] - 1) / d               # remove level and depth: peak 0, trough -1
            X.append(np.interp(s, tt, y))             # resample to L landmarks in time
            rows.append({
                "ticker": tk, "peak": str(t0.date()), "trough": str(tm.date()), "recovered": str(t1.date()),
                "depth": float(d), "days": int(len(seg) - 1),
                "trough_at": float((p.index.get_loc(tm) - p.index.get_loc(t0)) / (len(seg) - 1)),
            })
    X = np.array(X)
    mean = X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    var = S**2 / (S**2).sum()
    K = 8                                            # endpoints are pinned at 0, so the tail modes are empty
    S, Vt = S[:K], Vt[:K]
    sd = S / np.sqrt(len(X) - 1)
    trough_at = np.array([r["trough_at"] for r in rows])
    # Orient the modes: +mode 1 = higher curve (a sharp V), +mode 2 = later trough.
    if Vt[0].sum() < 0:
        Vt[0] *= -1
    if np.corrcoef((X - mean) @ Vt[1], trough_at)[0, 1] < 0:
        Vt[1] *= -1
    scores = (X - mean) @ Vt.T / sd                  # in standard deviations, as in ShapeLab's sample()
    for r, sc, x in zip(rows, scores, X):
        r["scores"] = r4(sc[:4])
        r["curve"] = r4(x)
    # How well k modes rebuild a curve, averaged over episodes.
    rebuild = []
    for k in range(0, K + 1):
        Xk = mean + (scores[:, :k] * sd[:k]) @ Vt[:k] if k else np.tile(mean, (len(X), 1))
        rebuild.append(float(np.sqrt(((X - Xk) ** 2).mean())))
    corr = {
        "m1_vs_trough_at": float(np.corrcoef(scores[:, 0], [r["trough_at"] for r in rows])[0, 1]),
        "m1_vs_log_days": float(np.corrcoef(scores[:, 0], np.log([r["days"] for r in rows]))[0, 1]),
        "m1_vs_depth": float(np.corrcoef(scores[:, 0], [r["depth"] for r in rows])[0, 1]),
        "m2_vs_depth": float(np.corrcoef(scores[:, 1], [r["depth"] for r in rows])[0, 1]),
        "m2_vs_trough_at": float(np.corrcoef(scores[:, 1], trough_at)[0, 1]),
    }
    return {
        "L": L, "s": r4(s), "mean": r4(mean),
        "modes": r4(Vt[:4]), "sd": r4(sd[:4]),
        "var": r4(var[:8]), "cumvar": r4(np.cumsum(var)[:8]),
        "rebuild_rmse": r4(rebuild),
        "episodes": rows, "corr": corr,
    }


if __name__ == "__main__":
    out = {"random_walk": study_random_walk(), "dosing": study_dosing(), "shapes": study_shapes()}
    path = ROOT / "results/crossover.json"
    path.write_text(json.dumps(out, default=r4, separators=(",", ":")))
    rw, ds, sh = out["random_walk"], out["dosing"], out["shapes"]
    print("A  n", rw["n"], "mu", rw["mu"], "sd", rw["sigma"])
    print("   VR", [(v["k"], round(v["vr"], 3), round(v["lo"], 3), round(v["hi"], 3)) for v in rw["vr"]])
    print("   tails", rw["tails"])
    print("   kurt by horizon", rw["kurt_h"])
    print("   acf r lag1", rw["acf_r"][0], "acf |r| lag1,10,50,100", [rw["acf_abs"][i] for i in (0, 9, 49, 99)], "band", rw["acf_band"])
    print("   robust", rw["vr_robust"], "\n   decades", rw["vr_decades"])
    print("   mdd", {k: v for k, v in rw["mdd"].items() if not isinstance(v, list)})
    print("B  events", len(ds["events"]), [(e["date"], round(e["peak"], 1), e["half_life"]) for e in ds["events"]])
    print("   one", ds["one"], "\n   two", ds["two"], "\n   ar1", ds["ar1"], "\n   recon", ds["recon"], "\n   recon2", ds["recon2"])
    print("C  episodes", len(sh["episodes"]), "var", sh["var"][:4], "cum", sh["cumvar"][:4])
    print("   rebuild", sh["rebuild_rmse"])
    print("   corr", sh["corr"])
    for e in sh["episodes"]:
        print("  ", e["ticker"], e["peak"], e["trough"], e["recovered"], round(e["depth"], 2), e["days"], round(e["trough_at"], 2), e["scores"][:2])
    print("wrote", path, path.stat().st_size, "bytes")
