"""Explainers for the Crossover section: market data through the MC312 and ShapeLab toolkits.

Every figure reuses modlab.crossover.study (and the Lab 10 walk in modlab.spread.models),
so the numbers on these pages are the numbers the live views show. Facts quoted in the
prose are computed at import from cheap calls; the expensive Monte Carlo run is only
done inside figures, once, and memoised.
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd
from scipy import stats

from ..core import montecarlo
from ..crossover import study as S
from ..spread import models as M
from .registry import explainer, figure


def _pct(v: float, d: int = 0) -> str:
    return f"{100 * v:.{d}f}%"


def _signed(v: float, d: int = 1) -> str:
    return f"{v:+.{d}f}".replace("-", "−")


def _n(v: float, d: int = 2) -> str:
    """A number with a true minus sign."""
    return f"{v:.{d}f}".replace("-", "−")


def _years(dates) -> np.ndarray:
    d = pd.DatetimeIndex(dates)
    return np.asarray(d.year + (d.dayofyear - 1) / 365.25, dtype=float)


def _every(n: int, most: int) -> np.ndarray:
    """At most `most` evenly spaced indices into a length-n array, always keeping the last."""
    step = max(1, -(-n // most))
    return np.unique(np.r_[np.arange(0, n, step), n - 1])


# ======================================================================== A. random walk
RW_TICKER, RW_SEED, RW_SIMS, RW_PERMS = "SPY", 312, 2000, 200        # the live view's defaults
VR_K = 10                                                            # the horizon of the live reversion test
KMAX = 63


@functools.lru_cache(maxsize=8)
def _x(tk: str = RW_TICKER):
    r = S.log_returns(tk)
    return r.to_numpy(), r.index


@functools.lru_cache(maxsize=2)
def _rw_full():
    """The live view's full analysis at its defaults (the slowest call on this page, ~0.4 s once)."""
    return S.random_walk(RW_TICKER, RW_SIMS, RW_PERMS, RW_SEED)


def _vr_all(v: np.ndarray, ks) -> np.ndarray:
    """Variance ratio at each horizon in ks, from one cumulative sum (same as study.variance_ratio)."""
    c = np.r_[0.0, np.cumsum(v)]
    v1 = v.var(ddof=1)
    return np.array([(c[k:] - c[:-k]).var(ddof=1) / (k * v1) for k in ks])


@functools.lru_cache(maxsize=8)
def _vr_curve(tk: str = RW_TICKER, shuffles: int = 100):
    x, _ = _x(tk)
    ks = np.arange(1, KMAX + 1)
    g = montecarlo.rng(RW_SEED)
    sh = np.array([_vr_all(g.permutation(x), ks) for _ in range(shuffles)])
    lo, med, hi = np.percentile(sh, [2.5, 50, 97.5], axis=0)
    return ks, _vr_all(x, ks), lo, med, hi, shuffles


@functools.lru_cache(maxsize=2)
def _vr10_shuffles(n: int = 1000):
    x, _ = _x()
    g = montecarlo.rng(RW_SEED + 1)
    return np.array([_vr_all(g.permutation(x), [VR_K])[0] for _ in range(n)])


def _scaled(x: np.ndarray) -> np.ndarray:
    """Each day divided by the volatility known the evening before (as in study.random_walk)."""
    ewma = pd.Series(x**2).ewm(alpha=0.06, adjust=False).mean().shift(1).to_numpy()
    return (x / np.sqrt(ewma))[63:]


def _runs(signs: np.ndarray) -> np.ndarray:
    """Lengths of the runs of equal signs."""
    cut = np.nonzero(np.diff(signs))[0] + 1
    return np.diff(np.r_[0, cut, len(signs)])


@functools.lru_cache(maxsize=1)
def _rw_facts() -> dict:
    x, idx = _x()
    n = len(x)
    z = (x - x.mean()) / x.std(ddof=1)
    vr10, z10 = S.lo_mackinlay(x, VR_K)
    return {"n": n, "start": idx[0], "end": idx[-1], "sigma": float(x.std(ddof=1)), "mu": float(x.mean()),
            "vr10": vr10, "z10": z10, "beyond4": int((np.abs(z) > 4).sum()), "expected4": n * 2 * stats.norm.sf(4),
            "kurtosis": float(stats.kurtosis(z, fisher=False)), "acf1": float(S.acf(x, 1)[0]),
            "acf_abs1": float(S.acf(np.abs(x), 1)[0]), "band": 1.96 / np.sqrt(n)}


TICKER_CTL = {"k": "ticker", "label": "Stock", "options": [[t, t] for t in S.TICKERS], "value": RW_TICKER}


@functools.lru_cache(maxsize=64)
def _coin(p: float, steps: int):
    return M.walk(p=p, steps=steps, walkers=2000, seed=10)


def _bg(name: str, color: str, y) -> dict:
    """A background series: drawn thin, kept out of the legend and the tooltip."""
    return {"name": name, "color": color, "y": y, "thin": True, "legend": False, "hover": False}


@figure("rw-coin")
def _fig_coin(p):
    """Lab 10's ±1 walk: forty walkers and the theory's ±2 sd envelope."""
    steps, pr = p.num("steps", 200, 50, 1000, integer=True), p.num("p", 0.5, 0.3, 0.7)
    w = _coin(round(pr, 4), steps)
    n = np.asarray(w["n"], float)
    keep = _every(len(n), 250)
    series = [_bg(f"Walker {i + 1}", "neutral", row[keep]) for i, row in enumerate(w["paths"])]
    series += [{"name": "Theory: mean + 2 sd", "color": "accent", "y": np.asarray(w["band_hi"])[keep], "dash": True},
               {"name": "Theory: mean − 2 sd", "color": "accent", "y": np.asarray(w["band_lo"])[keep], "dash": True},
               {"name": "Theory: mean n(2p − 1)", "color": "s2", "y": np.asarray(w["mean_theory"])[keep]}]
    s = w["summary"]
    vals, counts = np.asarray(w["final_values"]), np.asarray(w["final_counts"])
    inside = counts[np.abs(vals - s["mean_theory"]) <= 2 * s["sd_theory"]].sum() / counts.sum()
    return {
        "kind": "lines", "x": n[keep], "series": series, "xlabel": "Step n", "ylabel": "Position", "height": 300,
        "legend": [{"name": f"Walkers ({len(w['paths'])} of 2000)", "color": "neutral"},
                   {"name": "Theory: mean ± 2 sd", "color": "accent", "dash": True},
                   {"name": "Theory: mean n(2p − 1)", "color": "s2"}],
        "notes": [f"After {steps} steps the 2000 walkers' positions have variance {s['var']:.0f}; "
                  f"theory says 4np(1 − p) = {s['var_theory']:.0f}. Double the steps and the variance doubles.",
                  f"{_pct(inside)} of walkers end within ±2 sd of {_n(s['mean_theory'], 0)} "
                  f"(from {_n(s['range95'][0], 0)} to {_n(s['range95'][1], 0)}): the typical distance grows like √n, not n."],
    }


@functools.lru_cache(maxsize=64)
def _wp(tk: str, n: int):
    return S.walk_paths(tk, n, 1)


@figure("rw-paths")
def _fig_paths(p):
    """A stock's weekly log growth next to normal walks with its own weekly drift and spread."""
    tk, n = p.choice("ticker", RW_TICKER, S.TICKERS), p.num("n", 20, 5, 60, integer=True)
    w = _wp(tk, n)
    yrs = _years(w["dates"])
    keep = _every(len(yrs), 450)
    t = np.arange(len(yrs), dtype=float)
    mu, sd = w["weekly_mu"], w["weekly_sigma"]
    hi, lo = mu * t + 2 * sd * np.sqrt(t), mu * t - 2 * sd * np.sqrt(t)
    real = np.asarray(w["real"])
    series = [_bg(f"Walk {i + 1}", "neutral", row[keep]) for i, row in enumerate(np.asarray(w["walks"]))]
    series += [{"name": "Drift + 2σ√n", "color": "accent", "y": hi[keep], "dash": True},
               {"name": "Drift − 2σ√n", "color": "accent", "y": lo[keep], "dash": True},
               {"name": f"{tk} (log of growth)", "color": "s1", "y": real[keep]}]
    inside = float(((real <= hi) & (real >= lo)).mean())
    above = float((np.asarray(w["walk_multiples"]) > w["real_multiple"]).mean())
    return {
        "kind": "lines", "x": yrs[keep], "xdate": True, "series": series, "xlabel": "Year", "ylabel": "log(price / first price)",
        "height": 300,
        "legend": [{"name": f"{tk} (log of growth)", "color": "s1"}, {"name": "Random walks", "color": "neutral"},
                   {"name": "Drift ± 2σ√n", "color": "accent", "dash": True}],
        "notes": [f"$1 in {tk} became ${w['real_multiple']:.1f}; the median of these {n} walks ends at "
                  f"${np.median(w['walk_multiples']):.1f}, and {_pct(above)} of them end higher than {tk}.",
                  f"{tk} spends {_pct(inside)} of its {len(real)} weeks inside the ±2σ√n envelope"
                  + (". By eye, nothing separates it from the walks." if inside >= 0.95 else ", a little less than the 95% a walk would manage.")],
    }


@figure("rw-growth")
def _fig_growth(p):
    """How the variance of k-day returns grows with k, against the walk's straight line."""
    ks, vr, lo, med, hi, nsh = _vr_curve(RW_TICKER)
    return {
        "kind": "lines", "x": ks, "xlabel": "Horizon k (trading days)", "ylabel": "Var(k-day return) / Var(1-day return)", "ymin": 0,
        "series": [{"name": "Random walk: exactly k", "color": "neutral", "y": ks.astype(float), "dash": True},
                   {"name": f"Shuffled {RW_TICKER} days (median of {nsh})", "color": "s3", "y": ks * med},
                   {"name": f"{RW_TICKER}", "color": "s1", "y": ks * vr, "area": True}],
        "notes": [f"Quarterly ({KMAX}-day) returns have {KMAX * vr[-1]:.0f} times the variance of daily returns, "
                  f"not the {KMAX} times a random walk gives.",
                  f"Shuffling the same days into a random order gives {KMAX * med[-1]:.0f}: the shortfall comes from the order of the days, not their sizes."],
    }


@figure("rw-vr")
def _fig_vr(p):
    tk, k = p.choice("ticker", RW_TICKER, S.TICKERS), p.num("k", VR_K, 2, KMAX, integer=True)
    ks, vr, lo, med, hi, nsh = _vr_curve(tk)
    x, _ = _x(tk)
    v, z = S.lo_mackinlay(x, k)
    i = k - 1
    side = "less" if v < 1 else "more"
    return {
        "kind": "lines", "x": ks, "xlabel": "Horizon k (trading days)", "ylabel": "Variance ratio VR(k)",
        "series": [{"name": f"Shuffled days, 97.5th percentile ({nsh} shuffles)", "color": "s3", "y": hi, "dash": True, "thin": True},
                   {"name": "Shuffled days, 2.5th percentile", "color": "s3", "y": lo, "dash": True, "thin": True, "legend": False},
                   {"name": tk, "color": "s1", "y": vr}],
        "hlines": [{"y": 1, "label": "random walk", "color": "neutral"}],
        "markers": [{"x": k, "label": f"k = {k}"}],
        "points": [{"x": k, "y": float(vr[i]), "color": "accent"}],
        "notes": [f"{tk}: VR({k}) = {v:.2f}, so {k}-day returns vary {_pct(abs(1 - v))} {side} than {k} independent days would.",
                  f"Shuffled days give {lo[i]:.2f} to {hi[i]:.2f}; the Lo–MacKinlay robust z is {_signed(z)}, "
                  + ("outside ±1.96, so the walk is rejected at this horizon." if abs(z) > 1.96 else "inside ±1.96, so this horizon alone does not reject the walk.")],
    }


@figure("rw-perm")
def _fig_perm(p):
    n = p.num("shuffles", 200, 20, 1000, integer=True)
    sh = _vr10_shuffles()[:n]
    x, _ = _x()
    real = S.variance_ratio(x, VR_K)
    edges = np.linspace(min(sh.min(), real) - 0.02, sh.max() + 0.02, 41)
    counts = np.histogram(sh, edges)[0]
    below = int((sh <= real).sum())
    pv = (1 + below) / (1 + n)
    _, z = S.lo_mackinlay(x, VR_K)
    return {
        "kind": "lines", "x": (edges[1:] + edges[:-1]) / 2, "xlabel": f"VR({VR_K}) of a shuffled history", "ylabel": "Shuffles", "ymin": 0,
        "series": [{"name": f"{n} shuffles of the real days", "color": "s3", "y": counts, "area": True}],
        "markers": [{"x": real, "label": f"real {real:.2f}"}],
        "notes": [f"{below} of {n} shuffles give a VR({VR_K}) as low as the real {real:.2f}, so the permutation p-value is "
                  f"(1 + {below}) / (1 + {n}) = {pv:.3g}.",
                  (f"With {n} shuffles no p-value below {1 / (1 + n):.3g} is possible; more shuffles sharpen the bound. "
                   if below == 0 else "") + f"The robust z of {_signed(z)} gives a normal-theory p of {2 * stats.norm.sf(abs(z)):.2g}."],
    }


@figure("rw-acf")
def _fig_acf(p):
    tk = p.choice("ticker", RW_TICKER, S.TICKERS)
    x, _ = _x(tk)
    L = 100
    a, b = S.acf(x, L), S.acf(np.abs(x), L)
    band = 1.96 / np.sqrt(len(x))
    lags = np.arange(1, L + 1)
    out_r, out_abs = int((np.abs(a) > band).sum()), int((b > band).sum())
    return {
        "kind": "lines", "x": lags, "xlabel": "Lag ℓ (trading days)", "ylabel": "Autocorrelation ρ(ℓ)",
        "series": [{"name": f"{tk} returns r", "color": "s1", "y": a},
                   {"name": f"{tk} size of returns |r|", "color": "s2", "y": b, "area": True}],
        "hlines": [{"y": band, "label": "+1.96/√n", "color": "neutral"}, {"y": -band, "label": "−1.96/√n", "color": "neutral"}],
        "notes": [f"{tk}'s returns themselves remember little: ρ(1) = {_n(a[0], 3)}, and {out_r} of {L} lags fall outside the "
                  f"±{band:.3f} band (about {0.05 * L:.0f} would by chance).",
                  ("Their size remembers far longer" if out_abs > out_r else "Their size remembers no longer") + f": |r| has ρ(1) = {_n(b[0])}, {out_abs} of {L} lags are above the band, and ρ({L}) is {_n(b[-1])}."],
    }


@functools.lru_cache(maxsize=1)
def _runs_data(shuffles: int = 50):
    x, _ = _x()
    s = np.sign(x[x != 0])
    top = 10
    hist = lambda lens: np.bincount(np.minimum(lens, top), minlength=top + 1)[1:]
    obs = _runs(s)
    g = montecarlo.rng(RW_SEED + 2)
    sh = np.mean([hist(_runs(g.permutation(s))) for _ in range(shuffles)], axis=0)
    n1, n2 = int((s > 0).sum()), int((s < 0).sum())
    N = n1 + n2
    e = 2 * n1 * n2 / N + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - N) / (N**2 * (N - 1))
    return hist(obs), sh, len(obs), e, (len(obs) - e) / np.sqrt(var), n1, n2, shuffles, top


@figure("rw-runs")
def _fig_runs(p):
    obs, sh, R, e, z, n1, n2, nsh, top = _runs_data()
    labels = [str(k) for k in range(1, top)] + [f"{top}+"]
    more = "more" if R > e else "fewer"
    return {
        "kind": "bars", "labels": labels, "ylabel": "Number of runs", "xlabel": "Run length (days in a row with the same sign)",
        "series": [{"name": f"{RW_TICKER}", "color": "s1", "values": obs},
                   {"name": f"Shuffled days (mean of {nsh})", "color": "s3", "values": sh}],
        "notes": [f"{n1} up days and {n2} down days form {R} runs; independent signs would give {e:.0f} on average "
                  f"(Wald–Wolfowitz z = {_signed(z)}).",
                  f"{more.capitalize()} runs than chance means " + ("a down day is slightly more often followed by an up day, and the reverse: short reversals."
                                                                   if R > e else "days tend to follow the previous day's sign: short trends.")],
    }


@figure("rw-mdd")
def _fig_mdd(p):
    d = _rw_full()["drawdown"]
    bins = np.asarray(d["bins"])
    c = (bins[1:] + bins[:-1]) / 2
    return {
        "kind": "lines", "x": c, "xlabel": "Maximum drawdown over the whole history", "ylabel": "Simulated histories", "ymin": 0,
        "series": [{"name": "Normal walks", "color": "neutral", "y": d["hist_normal"], "area": True},
                   {"name": "Resampled real days", "color": "s3", "y": d["hist_bootstrap"]}],
        "markers": [{"x": d["real"], "label": f"{RW_TICKER} {_pct(d['real'])}"}],
        "notes": [f"{_pct(d['share_worse_normal'], 1)} of {d['sims']} normal walks of the same length fall further than "
                  f"{RW_TICKER}'s real {_pct(d['real'])}; with resampled real days it is {_pct(d['share_worse_bootstrap'], 1)}.",
                  f"The median simulated history loses {_pct(d['median_normal'])} at its worst. A deep crash on its own is what a long random walk produces."],
    }


@functools.lru_cache(maxsize=1)
def _ticker_z():
    out = []
    for tk in S.TICKERS:
        x, _ = _x(tk)
        out.append((tk, S.lo_mackinlay(x, VR_K)[1], S.lo_mackinlay(_scaled(x), VR_K)[1]))
    return out


@figure("rw-tickers")
def _fig_tickers(p):
    rows = _ticker_z()
    rej = [t for t, z, _ in rows if abs(z) > 1.96]
    rej_s = [t for t, _, z in rows if abs(z) > 1.96]
    worst = min(rows, key=lambda r: r[1])
    return {
        "kind": "bars", "labels": [r[0] for r in rows], "ylabel": f"Robust z of VR({VR_K})",
        "hlines": [{"y": 1.96, "label": "+1.96", "color": "neutral"}, {"y": -1.96, "label": "−1.96", "color": "neutral"}],
        "series": [{"name": "Raw daily returns", "color": "s1", "values": [r[1] for r in rows]},
                   {"name": "Returns ÷ yesterday's volatility", "color": "s2", "values": [r[2] for r in rows]}],
        "notes": [f"{len(rej)} of {len(rows)} tickers have |z| above 1.96 on raw returns"
                  + (f" ({', '.join(rej)})" if rej else "") + f"; the most negative is {worst[0]} at {_signed(worst[1])}.",
                  f"After dividing each day by the volatility known the evening before, {len(rej_s)} of {len(rows)} do"
                  + (f" ({', '.join(rej_s)})." if rej_s else ".")],
    }


_TEST_NAMES = {"normal": "Normal steps", "reversion": f"No mean reversion (VR at {VR_K} days)",
               "memory": "No memory in |r|", "crash": "Crash size"}


@figure("rw-tests")
def _fig_tests(p):
    d = _rw_full()
    T = {t["id"]: t for t in d["tests"]}
    tone = lambda i: "bad" if T[i]["fail"] else "good"
    verdict = lambda i: "rejects the walk" if T[i]["fail"] else "consistent with the walk"
    items = [
        {"label": _TEST_NAMES["normal"], "value": f"{T['normal']['beyond4']} days beyond 4σ",
         "note": f"normal steps expect {T['normal']['expected4']:.2f} · {verdict('normal')}", "tone": tone("normal")},
        {"label": _TEST_NAMES["reversion"], "value": f"{T['reversion']['vr']:.2f}",
         "note": f"robust z {_signed(T['reversion']['z'])} · {verdict('reversion')}", "tone": tone("reversion")},
        {"label": _TEST_NAMES["memory"], "value": f"ρ(1) = {T['memory']['acf_abs_1']:.2f}",
         "note": f"band ±{T['memory']['band']:.3f} · {verdict('memory')}", "tone": tone("memory")},
        {"label": _TEST_NAMES["crash"], "value": f"{_pct(T['crash']['share_worse'], 1)} fall further",
         "note": f"of {d['drawdown']['sims']} normal walks · {verdict('crash')}", "tone": tone("crash")},
    ]
    fails = sum(t["fail"] for t in d["tests"])
    dec = d["decades"]
    notes = [f"{fails} of {len(d['tests'])} tests reject the random walk for {RW_TICKER} "
             f"({d['n']} trading days, {d['start'][:4]}–{d['end'][:4]})."]
    notes.append(f"The {VR_K}-day z is {_signed(dec[0]['z10'])} for {dec[0]['span'].replace('-', '–')} and {_signed(dec[-1]['z10'])} for {dec[-1]['span'].replace('-', '–')}"
                 + (": the mean reversion has weakened." if d["decades_weaker"] else "."))
    return {"kind": "stats", "items": items, "notes": notes}


# ------------------------------------------------------------------------ A: text
_RW = _rw_facts()
_P0 = round(1 / (1 + RW_PERMS), 4)
_VOL = max(((tk, float(_x(tk)[0].std(ddof=1))) for tk in S.TICKERS), key=lambda r: r[1])

explainer("crossover/random-walk", {
    "title": "Is a stock a coin-flip walk? Testing the random-walk hypothesis",
    "kicker": "Time Series Lab · MC312 Labs 9–10 × markets",
    "summary": f"Lab 10's walker takes independent steps, so its spread grows like √n. The random-walk hypothesis says "
               f"log prices do the same. This page tests that claim on {_RW['n']} trading days of {RW_TICKER} with variance "
               "ratios, runs, autocorrelation and Monte Carlo, and explains what passing or failing each test means.",
    "sections": [
        {"id": "walker", "title": "Lab 10's walker", "blocks": [
            {"type": "p", "text": "In Lab 10 a walker tosses a coin at every step: heads one step right, tails one step left. "
                                  "Each toss ignores every earlier one. That single property, <b>independent and identically "
                                  "distributed steps</b>, fixes everything about the walk's spread."},
            {"type": "figure", "figure": "rw-coin", "title": "Two thousand coin-flip walkers",
             "caption": "Forty of the walkers, with the theory's mean and ±2 sd envelope. Change the number of steps or bias the coin.",
             "controls": [{"k": "steps", "label": "Steps", "min": 50, "max": 1000, "step": 50, "value": 200},
                          {"k": "p", "label": "P(step right)", "min": 0.3, "max": 0.7, "step": 0.05, "value": 0.5}]},
            {"type": "eq", "text": "Xₙ = ε₁ + ε₂ + … + εₙ,   εᵢ = +1 with probability p, −1 otherwise\nE[Xₙ] = n (2p − 1)\nVar(Xₙ) = 4 n p (1 − p)",
             "where": [["Xₙ", "position after n steps"], ["εᵢ", "the i-th step, independent of all the others"],
                       ["p", "chance of a step to the right"]]},
            {"type": "callout", "tone": "info", "title": "Variances add, distances do not",
             "text": "Because the steps are independent, the variance of their sum is the sum of their variances: n steps give n times "
                     "the variance of one. The typical distance is the square root of that, so it grows like √n. A walker who goes 100 "
                     "steps is typically only 10 steps from home."},
        ]},
        {"id": "hypothesis", "title": "Prices as a walk", "blocks": [
            {"type": "p", "text": "The <b>random-walk hypothesis</b> says the same of a stock's log price: each day adds a return "
                                  "drawn independently from one distribution. Tomorrow's return cannot be predicted from today's. "
                                  "Log prices are used because returns compound: adding log returns is the same as multiplying growth factors."},
            {"type": "eq", "text": "log Pₜ = log Pₜ₋₁ + rₜ,   rₜ = μ + σ εₜ,   εₜ independent, mean 0, variance 1",
             "where": [["Pₜ", "price on day t, including dividends"], ["rₜ", "log return on day t"],
                       ["μ", f"drift per day ({RW_TICKER}: {_RW['mu']:.5f})"],
                       ["σ", f"volatility per day ({RW_TICKER}: {_RW['sigma']:.4f})"]]},
            {"type": "figure", "figure": "rw-paths", "title": f"{RW_TICKER} next to walks with its own drift and volatility",
             "caption": "Weekly log growth since the first week of data. Each walk draws independent normal weekly steps with "
                        "the stock's own mean and standard deviation. Pick another stock.",
             "controls": [TICKER_CTL, {"k": "n", "label": "Walks shown", "min": 5, "max": 60, "step": 5, "value": 20}]},
            {"type": "p", "text": "By eye the real path is just another walk. The tests below look for what the eye misses: whether the "
                                  "variance really grows like n, whether the order of the days matters, and whether the steps "
                                  "have the shape and independence the walk assumes."},
        ]},
        {"id": "vr", "title": "The variance-ratio test", "blocks": [
            {"type": "p", "text": "If daily returns are independent, a k-day return is a sum of k independent days, so its variance is "
                                  "k times the daily variance. Measure both and divide: the ratio should be 1 at every horizon."},
            {"type": "eq", "text": "VR(k) = Var(rₜ + rₜ₋₁ + … + rₜ₋ₖ₊₁) / (k · Var(rₜ))\nVR(k) ≈ 1 + 2 Σⱼ₌₁ᵏ⁻¹ (1 − j/k) ρ(j)",
             "where": [["k", "horizon in trading days"], ["ρ(j)", "correlation between returns j days apart"],
                       ["VR < 1", "mean reversion: moves partly undo themselves"], ["VR > 1", "momentum: moves tend to continue"]]},
            {"type": "figure", "figure": "rw-growth", "title": "Variance against horizon",
             "caption": "A walk's variance grows along the dashed line. The real days grow more slowly; the same days shuffled do not."},
            {"type": "figure", "figure": "rw-vr", "title": "Variance ratio at each horizon",
             "caption": "The dashed band holds 95% of the ratios from shuffled histories. Move the horizon to read off the ratio and its robust z, "
                        "or pick another stock.",
             "controls": [TICKER_CTL, {"k": "k", "label": "Horizon k", "min": 2, "max": KMAX, "step": 1, "value": VR_K, "unit": "days"}]},
            {"type": "quiz", "q": f"{RW_TICKER}'s {VR_K}-day variance ratio is {_RW['vr10']:.2f}. What does that say about {VR_K}-day returns?",
             "options": ["They vary more than independent days would: momentum",
                         "They vary less than independent days would: some of each move is undone",
                         "Nothing: the variance ratio of any stock is 1"],
             "answer": 1, "why": f"A ratio of {_RW['vr10']:.2f} means the {VR_K}-day variance is {_pct(1 - _RW['vr10'])} short of {VR_K} "
                                 "times the daily one. By the second formula the correlations ρ(j) are on balance negative."},
        ]},
        {"id": "pvalues", "title": "Is the gap real? Shuffles and p-values", "blocks": [
            {"type": "p", "text": "A ratio below 1 could be luck. A <b>permutation test</b> asks how often luck alone produces it. "
                                  "Shuffling the days keeps every return, so the distribution is untouched, but destroys their order, "
                                  "which is exactly what the hypothesis says does not matter."},
            {"type": "steps", "title": "The permutation test, as in Lab 9", "items": [
                f"Compute the statistic on the real data: VR({VR_K}).",
                "Shuffle the days into a random order and compute it again.",
                "Repeat many times. The shuffled values show what the statistic looks like when order carries no information.",
                "The p-value is the share of shuffles at least as extreme as the real value, counting the real data as one of them.",
            ]},
            {"type": "eq", "text": "p = (1 + #{shuffles with VR ≤ VR_real}) / (1 + N)",
             "where": [["N", "number of shuffles"], ["+1", "the real ordering counts as one possible shuffle, so p is never 0"]]},
            {"type": "figure", "figure": "rw-perm", "title": f"Where the real VR({VR_K}) falls among shuffles",
             "caption": "Each shuffle is one Monte Carlo draw from the world where the walk is true. Add shuffles to sharpen the p-value.",
             "controls": [{"k": "shuffles", "label": "Shuffles", "min": 20, "max": 1000, "step": 20, "value": 200}]},
            {"type": "callout", "tone": "tip", "title": "The robust z",
             "text": "Shuffling also scrambles volatility clustering, so a calm week and a wild week become interchangeable. "
                     "Lo and MacKinlay's z (1988) measures the ratio against a standard error that allows volatility to change "
                     "over time. It is the number the live model uses for its verdict."},
            {"type": "quiz", "q": f"You run {RW_PERMS} shuffles and none gives a ratio as low as the real one. What p-value can you report?",
             "options": ["Exactly 0", f"About {_P0}, the smallest {RW_PERMS} shuffles can show", "0.05"],
             "answer": 1, "why": f"With the real ordering counted as one of the {RW_PERMS + 1} possibilities, the smallest p-value is "
                                 f"1/{RW_PERMS + 1} ≈ {_P0}. To claim less you need more shuffles."},
        ]},
        {"id": "memory", "title": "Runs and memory", "blocks": [
            {"type": "p", "text": "Two simpler tests look at the order of the days directly. The <b>runs test</b> counts streaks of up "
                                  "days and down days: independent signs produce a predictable number of streaks. "
                                  "<b>Autocorrelation</b> measures how much a day's return resembles the return ℓ days earlier."},
            {"type": "figure", "figure": "rw-runs", "title": "Streaks of up and down days",
             "caption": "Runs of each length in the real data against the average over shuffled histories."},
            {"type": "eq", "text": "ρ(ℓ) = Σ (rₜ − r̄)(rₜ₋ℓ − r̄) / Σ (rₜ − r̄)²\nunder independence ρ(ℓ) ≈ N(0, 1/n), so the 95% band is ±1.96 / √n",
             "where": [["ℓ", "lag in trading days"], ["n", f"number of days ({_RW['n']})"]]},
            {"type": "figure", "figure": "rw-acf", "title": "Memory in returns and in their size",
             "caption": "Returns (unshaded line) and their size |r| (shaded), with the 95% band for no memory. Pick another stock.",
             "controls": [TICKER_CTL]},
            {"type": "callout", "tone": "info", "title": "Unpredictable is not the same as independent",
             "text": "The direction of tomorrow's move is close to a coin flip, but its size is not: big days follow big days. "
                     "The walk needs both to be independent, so this <b>volatility clustering</b> alone rejects it."},
        ]},
        {"id": "tails", "title": "Fat tails and crash size: Monte Carlo", "blocks": [
            {"type": "p", "text": f"A normal walk almost never moves 4 standard deviations in a day: over {_RW['n']} days it expects "
                                  f"{_RW['expected4']:.2f} such days. {RW_TICKER} had {_RW['beyond4']}, and its daily kurtosis is "
                                  f"{_RW['kurtosis']:.1f} where a normal distribution has 3. The steps have <b>fat tails</b>."},
            {"type": "p", "text": "Do fat tails make crashes deeper than a walk allows? There is no formula for the worst drawdown of a "
                                  "long walk, so the server simulates thousands of histories of the same length and looks at the distribution."},
            {"type": "compare", "items": [
                {"title": "Normal walks", "points": ["Each day drawn from N(μ, σ²)", "Thin tails, no clustering", "The textbook hypothesis"]},
                {"title": "Resampled real days", "points": ["Each day drawn from the real returns with replacement",
                                                            "Keeps the fat tails", "Loses the clustering, because order is random"]},
            ]},
            {"type": "figure", "figure": "rw-mdd", "title": "Worst crash in a simulated history",
             "caption": f"The deepest peak-to-trough fall in each of {RW_SIMS} simulated histories, against {RW_TICKER}'s real one."},
            {"type": "try", "label": f"Run the four tests on {_VOL[0]}", "params": {"ticker": _VOL[0]},
             "note": f"The most volatile of the {len(S.TICKERS)} tickers: {_VOL[1]:.2%} a day against {RW_TICKER}'s {_RW['sigma']:.2%}."},
        ]},
        {"id": "six", "title": "Six stocks, one test", "blocks": [
            {"type": "p", "text": f"The same {VR_K}-day test on every stock in the data. The second bar divides each day by the volatility "
                                  "known the evening before, which removes most of the clustering and asks whether mean reversion "
                                  "survives once calm and wild periods are put on one scale."},
            {"type": "figure", "figure": "rw-tickers", "title": f"Robust z of the {VR_K}-day variance ratio",
             "caption": "Bars below −1.96 reject the walk in favour of mean reversion; above +1.96 in favour of momentum."},
            {"type": "try", "label": "Compare XOM with more Monte Carlo histories", "params": {"ticker": "XOM", "sims": 5000}},
        ]},
        {"id": "verdict", "title": "What passing and failing mean", "blocks": [
            {"type": "figure", "figure": "rw-tests", "title": f"The four tests for {RW_TICKER}",
             "caption": "The same verdicts the live model shows, from the same code and the same seed."},
            {"type": "compare", "items": [
                {"title": "A stock that passes", "points": ["The data cannot tell it from a walk at this sample size",
                                                            "Not proof that it is one: a weak pattern needs more data",
                                                            "Past moves carry no usable signal for this statistic"]},
                {"title": "A stock that fails", "points": ["Some assumption is wrong: shape, independence, or both",
                                                           "Says which one, not how to profit from it",
                                                           "The size of the effect matters as much as its z"]},
            ]},
            {"type": "callout", "tone": "warn", "title": "Limits of these tests",
             "text": "Running many tests on several stocks and horizons makes some rejections likely by chance. A small mean "
                     "reversion can be statistically clear and still far smaller than trading costs. The tests assume the "
                     "process stayed the same over the whole sample; the decade-by-decade numbers under the verdicts show whether it did. "
                     "None of this is investment advice."},
        ]},
    ],
    "takeaways": [
        "A random walk adds independent steps, so its variance grows like n and its typical distance like √n.",
        f"The variance ratio VR(k) compares k-day variance with k times daily variance; {RW_TICKER}'s is {_RW['vr10']:.2f} at {VR_K} days, "
        f"with a robust z of {_signed(_RW['z10'])}.",
        "Shuffling the days builds the null distribution directly: p = (1 + #extreme) / (1 + N), never zero.",
        f"Returns barely remember (ρ(1) = {_n(_RW['acf1'])}), but their size does (ρ(1) of |r| = {_RW['acf_abs1']:.2f}).",
        f"Fat tails are plain: {_RW['beyond4']} days beyond 4σ where a normal walk expects {_RW['expected4']:.2f}. A deep crash alone does not reject the walk.",
    ],
    "glossary": [
        ["Random-walk hypothesis", "Log prices add independent, identically distributed returns, so past returns do not predict future ones."],
        ["Variance ratio", "Variance of k-day returns divided by k times the variance of daily returns; 1 for a random walk."],
        ["Permutation test", "Comparing a statistic with its values on shuffled data, where any pattern in the order has been destroyed."],
        ["p-value", "The chance, if the hypothesis were true, of a result at least as extreme as the one seen."],
        ["Runs test", "Counts streaks of same-sign days and compares the count with what independent signs would give."],
        ["Autocorrelation", "Correlation of a series with itself ℓ steps earlier."],
        ["Volatility clustering", "Large moves follow large moves and calm follows calm, even when the direction is unpredictable."],
        ["Maximum drawdown", "The largest fall from a running peak to a later low, as a share of the peak."],
    ],
}, order=1)


# ======================================================================== B. dose
SPIKE_RATIO = 1.6                                                    # the live view's default spike threshold


@functools.lru_cache(maxsize=64)
def _events(ratio: float = SPIKE_RATIO):
    return S.vix_events(ratio)


@functools.lru_cache(maxsize=256)
def _dose(half_life: float, threshold: float):
    return S.vix_dose_model(half_life=half_life, threshold=threshold)


@functools.lru_cache(maxsize=64)
def _dose_fit(threshold: float, compartments: int = 1):
    return S.vix_dose_model(threshold=threshold, compartments=compartments, fit=True)


def _hl_slow(two: dict) -> str:
    hs = two["hl_slow"]
    return "it barely clears in the window" if hs is None or hs > 1000 else f"{hs:.0f} days"


@figure("dose-spikes")
def _fig_spikes(p):
    ratio = p.num("ratio", SPIKE_RATIO, 1.3, 2.2)
    E = _events(round(ratio, 3))
    h = np.asarray(E["h"], float)
    C = np.array([(np.asarray(S.vix().loc[ev["date"]:].iloc[:len(h)]) - ev["base"]) / (ev["peak"] - ev["base"]) for ev in E["events"]])
    series = [_bg(ev["date"], "neutral", row) for ev, row in zip(E["events"], C)]
    series += [{"name": "Median spike", "color": "s1", "y": E["median"]},
               {"name": "Middle 50%, upper edge", "color": "s1", "y": E["q75"], "dash": True, "thin": True, "legend": False},
               {"name": "Middle 50%, lower edge", "color": "s1", "y": E["q25"], "dash": True, "thin": True, "legend": False}]
    half = int(np.argmax(np.asarray(E["median"]) <= 0.5))
    never = sum(ev["days_to_half"] is None for ev in E["events"])
    quick = sum(ev["days_to_half"] is not None and ev["days_to_half"] <= 5 for ev in E["events"])
    return {
        "kind": "lines", "x": h, "series": series, "xlabel": "Trading days after the peak", "ylabel": "Share of the excess left",
        "height": 300, "hlines": [{"y": 0.5, "label": "half the excess", "color": "accent"}, {"y": 0, "label": "back at the median", "color": "neutral"}],
        "legend": [{"name": "Median spike", "color": "s1"}, {"name": "Middle 50% of spikes", "color": "s1", "dash": True},
                   {"name": f"Each of the {E['n_events']} spikes", "color": "neutral"}],
        "notes": [f"{E['n_events']} spikes reach at least ×{ratio:.2f} their prior 3-month median. "
                  f"The median spike has lost half its excess after {half} trading days.",
                  f"{quick} of {E['n_events']} spikes halve within 5 days; "
                  + (f"{never} never halve within {E['params']['horizon']} days." if never else f"every one halves within {E['params']['horizon']} days.")],
    }


@figure("dose-halflife")
def _fig_halflife(p):
    hl = p.num("half_life", 10, 1, 30)
    E = _events()
    h, med = np.asarray(E["h"], float), np.asarray(E["median"])
    curve = np.exp(-np.log(2) / hl * h)
    r2 = 1 - ((med - curve) ** 2).sum() / ((med - med.mean()) ** 2).sum()
    best = E["one"]
    return {
        "kind": "lines", "x": h, "xlabel": "Trading days after the peak", "ylabel": "Share of the excess left",
        "series": [{"name": "Median spike", "color": "s1", "y": med},
                   {"name": "Middle 50%, upper edge", "color": "neutral", "y": E["q75"], "dash": True, "thin": True},
                   {"name": "Middle 50%, lower edge", "color": "neutral", "y": E["q25"], "dash": True, "thin": True, "legend": False},
                   {"name": f"e^(−kt), t½ = {hl:.1f} days", "color": "s2", "y": curve}],
        "markers": [{"x": hl, "label": "t½"}],
        "hlines": [{"y": 0.5, "label": "half", "color": "accent"}],
        "points": [{"x": hl, "y": 0.5, "color": "s2"}],
        "notes": [(f"With a half-life of {hl:.1f} days one exponential explains R² = {r2:.2f} of the median spike's shape." if r2 >= 0 else
                   f"With a half-life of {hl:.1f} days one exponential fits worse than a flat line (R² = {_n(r2)})."),
                  f"Least squares chooses t½ = {best['half_life']:.1f} days, and even that reaches only R² = {best['r2']:.2f}: "
                  + ("no single rate fits both the fast first days and the slow tail." if E["two_better"] else "a single rate fits well.")],
    }


@figure("dose-flow")
def _fig_flow(p):
    E = _events()
    one, two = E["one"], E["two"]
    hs = _hl_slow(two)
    w = lambda share: 1.5 + 9 * share
    return {
        "kind": "flow", "height": 330,
        "nodes": [
            {"id": "d1", "x": 20, "y": 40, "w": 140, "h": 70, "title": "Dose", "sub": "the spike's excess"},
            {"id": "c1", "x": 230, "y": 40, "w": 140, "h": 70, "title": "One tank", "value": f"t½ {one['half_life']:.1f} d",
             "sub": f"R² {one['r2']:.2f}", "fill": 0.5, "fill_color": "s2", "hot": not E["two_better"]},
            {"id": "o1", "x": 440, "y": 40, "w": 140, "h": 70, "title": "Cleared", "sub": "back to the median"},
            {"id": "d2", "x": 20, "y": 205, "w": 140, "h": 70, "title": "Dose", "sub": "split on arrival"},
            {"id": "fast", "x": 230, "y": 170, "w": 140, "h": 62, "title": "Fast tank", "value": f"t½ {two['hl_fast']:.1f} d",
             "fill": float(two["a"]), "fill_color": "s1", "hot": E["two_better"]},
            {"id": "slow", "x": 230, "y": 255, "w": 140, "h": 62, "title": "Slow tank",
             "value": f"t½ {two['hl_slow']:.0f} d" if two["hl_slow"] and two["hl_slow"] < 1000 else "t½ very long",
             "fill": float(two["a_slow"]), "fill_color": "s3"},
            {"id": "o2", "x": 440, "y": 205, "w": 140, "h": 70, "title": "Cleared", "sub": f"R² {two['r2']:.2f}"},
        ],
        "edges": [
            {"from": "d1", "to": "c1", "label": "100%", "width": w(1), "flow": True, "color": "s2"},
            {"from": "c1", "to": "o1", "label": f"k = {one['k']:.3f}/day", "width": 3, "flow": True, "color": "s2"},
            {"from": "d2", "to": "fast", "label": _pct(two["a"]), "width": w(two["a"]), "flow": True, "color": "s1"},
            {"from": "d2", "to": "slow", "label": _pct(two["a_slow"]), "width": w(two["a_slow"]), "flow": True, "color": "s3"},
            {"from": "fast", "to": "o2", "label": f"k = {two['k_fast']:.3f}", "width": 3, "flow": True, "color": "s1"},
            {"from": "slow", "to": "o2", "label": f"k = {two['k_slow']:.3f}", "width": 1.5, "flow": True, "color": "s3"},
        ],
        "texts": [{"x": 300, "y": 22, "text": "One compartment", "cls": "n-title"},
                  {"x": 300, "y": 152, "text": "Two compartments in parallel", "cls": "n-title"}],
        "notes": [f"One tank: the whole excess clears at k = {one['k']:.3f} per day, a half-life of {one['half_life']:.1f} days (R² {one['r2']:.2f}).",
                  f"Two tanks: {_pct(two['a'])} clears with a {two['hl_fast']:.1f}-day half-life and {_pct(two['a_slow'])} over {hs} (R² {two['r2']:.2f})."],
    }


@figure("dose-log")
def _fig_log(p):
    E = _events()
    h, med = np.asarray(E["h"], float), np.asarray(E["median"])
    lo = float(med.min()) * 0.7
    early = np.log(2) / ((np.log(med[0]) - np.log(med[5])) / 5)
    late = np.log(2) / ((np.log(med[20]) - np.log(med[-1])) / (h[-1] - 20))
    return {
        "kind": "lines", "x": h, "xlabel": "Trading days after the peak", "ylabel": "Share of the excess left (log scale)",
        "ylog": True, "ymin": lo, "ymax": 1.1,
        "series": [{"name": "Median spike", "color": "s1", "y": med},
                   {"name": "One compartment", "color": "s2", "y": E["one"]["curve"], "dash": True},
                   {"name": "Two compartments", "color": "s3", "y": E["two"]["curve"]}],
        "notes": [f"On a log axis one exponential is a straight line. The median spike bends: over days 0–5 it halves every "
                  f"{early:.1f} days, over days 20–{int(h[-1])} every {late:.0f} days.",
                  f"Two straight pieces fit that bend: R² {E['two']['r2']:.2f} against {E['one']['r2']:.2f} for one."],
    }


@figure("dose-ar1")
def _fig_ar1(p):
    v = S.vix().to_numpy()
    E = _events()
    phi, base = E["ar1"]["phi"], E["ar1"]["base"]
    keep = np.arange(0, len(v) - 1, 9)
    xs = np.array([v.min(), v.max()])
    c = base * (1 - phi)
    dates = S.vix().index
    return {
        "kind": "scatter", "xlabel": "VIX yesterday", "ylabel": "VIX today", "height": 320,
        "points": [{"x": float(v[i]), "y": float(v[i + 1]), "color": "neutral", "r": 2, "name": f"{dates[i + 1]:%d %b %Y}"} for i in keep],
        "lines": [{"x": xs, "y": xs, "color": "neutral", "dash": True, "name": "No change"},
                  {"x": xs, "y": c + phi * xs, "color": "s2", "name": f"AR(1) fit, φ = {phi:.3f}"}],
        "legend": [{"name": f"Every ninth day ({len(keep)} pairs)", "color": "neutral"},
                   {"name": "No change (today = yesterday)", "color": "neutral", "dash": True},
                   {"name": f"AR(1) fit, φ = {phi:.3f}", "color": "s2"}],
        "notes": [f"Least squares over all {len(v) - 1} day pairs gives φ = {phi:.4f}: each day keeps {_pct(phi, 1)} of yesterday's "
                  f"distance from the long-run level of {base:.1f}.",
                  f"That is a half-life of ln 2 / (−ln φ) = {E['ar1']['half_life']:.0f} trading days, much slower than the median "
                  f"spike's {E['one']['half_life']:.1f}, because calm days dominate the fit."],
    }


@figure("dose-vix")
def _fig_vix(p):
    hl, th = p.num("half_life", 12.5, 1, 60), p.num("threshold", 3.4, 1, 8)
    m = _dose(round(hl, 3), round(th, 3))
    idx, x, model = m["_series"]
    keep = _every(len(x), 1400)
    yrs = _years(idx)
    end = float(yrs[-1])
    bands = [{"x0": float(_years([lo])[0]), "x1": min(end, float(_years([hi])[0])), "label": k} for k, (lo, hi) in S.VIX_WINDOWS.items()]
    f = _dose_fit(round(th, 3))
    win = ", ".join(f"{k}: {_n(v)}" for k, v in m["windows"].items())
    return {
        "kind": "lines", "x": yrs[keep], "xdate": True, "xlabel": "Year", "ylabel": "VIX", "ymin": 0, "height": 320,
        "series": [{"name": "VIX", "color": "s1", "y": x[keep], "thin": True},
                   {"name": "Median + what is left of past doses", "color": "s2", "y": model[keep]}],
        "bands": bands,
        "notes": [f"{m['n_doses']} of {m['n_days']} days ({_pct(m['dose_share'], 1)}) have a jump above +{th:.1f} points. With a "
                  f"{hl:.1f}-day half-life the rebuilt series has R² = {_n(m['r2'])} on all days (crisis windows: {win}).",
                  f"For this threshold least squares would choose t½ = {f['half_life']:.1f} days (R² {f['r2']:.2f})."],
    }


@functools.lru_cache(maxsize=1)
def _threshold_sweep():
    ths = np.round(np.arange(1, 8.001, 0.25), 2)
    fits = [_dose_fit(float(t)) for t in ths]
    return ths, np.array([f["r2"] for f in fits]), np.array([f["dose_share"] for f in fits]), np.array([f["half_life"] for f in fits])


@figure("dose-threshold")
def _fig_threshold(p):
    ths, r2, share, hls = _threshold_sweep()
    i = int(np.argmax(r2))
    return {
        "kind": "lines", "x": ths, "xlabel": "Dose threshold (VIX points in one day)", "ylabel": "Share", "ymin": 0, "ymax": 1,
        "series": [{"name": "R² of the rebuilt VIX (half-life fitted)", "color": "s1", "y": r2},
                   {"name": "Share of days that are doses", "color": "s3", "y": share, "area": True}],
        "markers": [{"x": float(ths[i]), "label": "best R²"}],
        "notes": [f"R² peaks at {r2[i]:.2f} with a threshold of {ths[i]:.2f} points, where {_pct(share[i], 1)} of days are doses "
                  f"and the fitted half-life is {hls[i]:.1f} days.",
                  f"Raising the threshold to {ths[-1]:.0f} points keeps only {_pct(share[-1], 1)} of days as doses; R² falls to {r2[-1]:.2f} "
                  f"and the half-life stretches to {hls[-1]:.0f} days to make up for the missing doses."],
    }


# ------------------------------------------------------------------------ B: text
_E = _events()
_F1, _F2 = _dose_fit(3.4), _dose_fit(3.4, 2)
_clip = lambda v, lo, hi: min(hi, max(lo, v))
_WIN = _dose(12.5, 3.4)["windows"]
_ALL = _dose(12.5, 3.4)["r2"]
_worst_win = min(_WIN, key=_WIN.get)
_FAIL = (f" In the {_worst_win} window R² is only {_n(_WIN[_worst_win])}: there the model fails." if _WIN[_worst_win] < 0.2 else "")
_YRS = int(S.vix().index[-1].year - S.vix().index[0].year)

explainer("crossover/dose", {
    "title": "Fear as a drug dose: how fast VIX spikes clear",
    "kicker": "Time Series Lab · MC312 Lab 2 × markets",
    "summary": f"Lab 2 clears a drug from the blood at a fixed fractional rate. Treat each jump in the VIX as a dose of fear and "
               f"the same maths applies: across {_E['n_events']} spikes the median loses half its excess in about "
               f"{_E['one']['half_life']:.0f} trading days, a slow second phase lingers, and the whole idea turns out to be an AR(1) "
               "process with a threshold.",
    "sections": [
        {"id": "pill", "title": "From a pill to a panic", "blocks": [
            {"type": "p", "text": "The VIX measures how much volatility options traders expect from the S&P 500 over the next month. "
                                  "It sits near a typical level, jumps when markets are frightened and then drifts back down. "
                                  "That is the shape of a drug in the blood after a single dose."},
            {"type": "cards", "items": [
                {"title": "The dose", "text": "The jump above the VIX's usual level: how much fear arrived at once."},
                {"title": "The baseline", "text": "The level it returns to. For a spike: the median of the three months before it."},
                {"title": "Clearance", "text": "The fraction of the remaining excess that disappears each day."},
            ]},
            {"type": "figure", "figure": "dose-spikes", "title": "Every VIX spike, scaled to one dose",
             "caption": "Each thin curve is one spike's excess over its prior 3-month median, divided by the excess at the peak, so every "
                        "spike starts at 1. Raise the threshold to keep only bigger spikes.",
             "controls": [{"k": "ratio", "label": "Spike: at least", "min": 1.3, "max": 2.2, "step": 0.05, "value": SPIKE_RATIO,
                           "unit": "× median"}]},
        ]},
        {"id": "clearance", "title": "First-order clearance and the half-life", "blocks": [
            {"type": "p", "text": "In Lab 2's one-compartment model the body removes a fixed <i>fraction</i> of the drug per hour, "
                                  "not a fixed amount. More drug means faster removal, so the level falls quickly at first and "
                                  "slowly later, and the time to halve is always the same."},
            {"type": "eq", "text": "dE/dt = −k E   ⇒   E(t) = E₀ e^(−k t)\nt½ = ln 2 / k",
             "where": [["E(t)", "excess over the baseline, as a share of the excess at the peak"],
                       ["k", "clearance rate per trading day"], ["t½", "half-life: days for the excess to halve"]]},
            {"type": "figure", "figure": "dose-halflife", "title": "Fit a half-life by hand",
             "caption": "Drag the half-life until the curve follows the median spike. The dashed lines hold the middle half of all spikes.",
             "controls": [{"k": "half_life", "label": "Half-life", "min": 1, "max": 30, "step": 0.5, "value": 10, "unit": "days"}]},
            {"type": "quiz", "q": "A spike's excess halves in 4 days. What share is left after 12 days?",
             "options": ["A quarter", "An eighth", "None: it is gone after 8 days"], "answer": 1,
             "why": "Twelve days is three half-lives, so ½ × ½ × ½ = ⅛ is left. First-order clearance never reaches zero."},
        ]},
        {"id": "compartments", "title": "One tank or two?", "blocks": [
            {"type": "p", "text": "Drugs often show two phases: a fast fall as the drug spreads into tissue, then a slow fall as it is "
                                  "eliminated. A two-compartment model captures this with two exponentials. Here each dose splits on "
                                  "arrival between a fast tank and a slow tank, and each tank clears at its own rate."},
            {"type": "eq", "text": "E(t) = a e^(−k_f t) + (1 − a) e^(−k_s t)",
             "where": [["a", "share of the dose in the fast tank"], ["k_f, k_s", "fast and slow clearance rates, k_f > k_s"]]},
            {"type": "figure", "figure": "dose-flow", "title": "One compartment against two",
             "caption": "Both models fitted by least squares to the median spike. Arrow widths follow the shares of the dose."},
            {"type": "compare", "items": [
                {"title": "Lab 2's two compartments", "points": ["Gut and blood, in series", "The drug must pass through the gut first",
                                                                 "Produces a delayed rise, then a fall"]},
                {"title": "The VIX's two compartments", "points": ["Fast and slow tanks, in parallel", "Fear arrives in both at once",
                                                                    "Produces a fast fall, then a long tail"]},
            ]},
            {"type": "figure", "figure": "dose-log", "title": "The bend that one exponential cannot follow",
             "caption": "The same curves on a logarithmic axis, where a single exponential is a straight line."},
        ]},
        {"id": "ar1", "title": "It is really an AR(1)", "blocks": [
            {"type": "p", "text": "Step the clearance equation forward one day at a time and it becomes a familiar statistical model. "
                                  "Each day keeps a fixed share φ of yesterday's excess. That is an <b>autoregressive process of order "
                                  "one</b>, AR(1), and φ is e^(−k)."},
            {"type": "eq", "text": "xₜ − m = φ (xₜ₋₁ − m) + εₜ,   φ = e^(−k)\nt½ = ln 2 / (−ln φ)",
             "where": [["xₜ", "the VIX on day t"], ["m", "the long-run level"], ["φ", "share of yesterday's excess kept today"],
                       ["εₜ", "the day's new shock"]]},
            {"type": "figure", "figure": "dose-ar1", "title": "Today against yesterday",
             "caption": "Every ninth day of the VIX. Points lie close to a line slightly flatter than the diagonal: that slope is φ."},
            {"type": "quiz", "q": f"An AR(1) fitted to the whole VIX gives φ = {_E['ar1']['phi']:.3f}. Roughly what half-life is that?",
             "options": ["About 1 day", f"About {_E['ar1']['half_life']:.0f} days", f"About {1 / (1 - _E['ar1']['phi']):.0f} days"],
             "answer": 1, "why": f"t½ = ln 2 / (−ln {_E['ar1']['phi']:.3f}) ≈ {_E['ar1']['half_life']:.0f} trading days. "
                                 "The whole-series fit is slower than the spikes because most days are calm, and calm days hardly move."},
            {"type": "p", "text": "The dose model is an AR(1) with one change: the shock εₜ only enters on days when the VIX jumps by "
                                  "more than a threshold, and then it is the jump itself. Every other day the VIX just clears. "
                                  "It is a <b>thresholded AR(1)</b>."},
            {"type": "eq", "text": "eₜ = φ eₜ₋₁ + dₜ,   dₜ = Δxₜ if Δxₜ > θ, otherwise 0\nmodel VIX = m + eₜ",
             "where": [["eₜ", "excess fear carried on day t"], ["dₜ", "the day's dose"], ["θ", "the dose threshold in VIX points"],
                       ["m", "the median of the whole series"]]},
        ]},
        {"id": "rebuild", "title": f"Rebuilding {_YRS} years of the VIX", "blocks": [
            {"type": "p", "text": "Feed every real jump above the threshold into the clearance model and see how much of the VIX it "
                                  "reproduces. The shaded crisis windows are scored separately."},
            {"type": "figure", "figure": "dose-vix", "title": "The VIX rebuilt from doses",
             "caption": "The VIX (thin) and the median plus whatever is left of past doses. Shaded: the crisis windows. Change the half-life and the threshold.",
             "controls": [{"k": "half_life", "label": "Half-life", "min": 1, "max": 60, "step": 0.5, "value": 12.5, "unit": "days"},
                          {"k": "threshold", "label": "Dose threshold", "min": 1, "max": 8, "step": 0.1, "value": 3.4, "unit": "pts"}]},
            {"type": "try", "label": "Open the two-compartment fit in the live model",
             "params": {"compartments": "2", "a": round(_clip(_F2["a"], 0.05, 0.95), 2),
                        "hl_fast": round(_clip(_F2["hl_fast"], 0.5, 20), 1), "hl_slow": int(round(_clip(_F2["hl_slow"], 5, 200)))},
             "note": f"Least squares over the whole series: {_pct(_F2['a'])} fast at {_F2['hl_fast']:.1f} days, "
                     f"the rest at {_F2['hl_slow']:.0f} days (R² {_F2['r2']:.2f})."},
            {"type": "try", "label": "Zoom on the 2020 crisis", "params": {"window": "2020", "compartments": "1",
                                                                         "half_life": round(_clip(_F1["half_life"], 1, 60), 1)}},
            {"type": "figure", "figure": "dose-threshold", "title": "How much counts as a dose?",
             "caption": "For each threshold the half-life is refitted by least squares. Lower thresholds count more days as doses."},
        ]},
        {"id": "limits", "title": "What the model leaves out", "blocks": [
            {"type": "p", "text": f"With the live defaults R² is {_ALL:.2f} across all days and "
                                  f"{', '.join(f'{_n(v)} in the {k} window' for k, v in _WIN.items())}. Fear that builds "
                                  "slowly over weeks, without a single big jump, never counts as a dose, so the model misses it."},
            {"type": "callout", "tone": "warn", "title": "Limits",
             "text": "The doses are the VIX's own jumps, so the model is told when and how much fear arrived and only has to explain "
                     "the decay; it cannot forecast a spike. The baseline is one fixed median for the whole series. The crisis windows "
                     f"were chosen after the fact.{_FAIL} This is a "
                     "description of the past, not a trading signal or investment advice."},
            {"type": "list", "items": [
                "Spike half-lives depend on the spike threshold: try the slider in the first figure.",
                "Two compartments describe the median spike better, but each individual spike is noisier than the median.",
                "The AR(1) view says the same thing with fewer assumptions, and is the model a statistician would start from.",
            ]},
        ]},
    ],
    "takeaways": [
        "First-order clearance removes a fixed fraction per day: E(t) = e^(−kt), with half-life ln 2 / k.",
        f"The median VIX spike halves in about {_E['one']['half_life']:.1f} trading days with one compartment (R² {_E['one']['r2']:.2f}).",
        f"Two compartments fit far better (R² {_E['two']['r2']:.2f}): {_pct(_E['two']['a'])} clears in {_E['two']['hl_fast']:.1f} days, "
        f"the rest over {_hl_slow(_E['two'])}." if _E["two_better"] else
        f"Two compartments add little (R² {_E['two']['r2']:.2f} against {_E['one']['r2']:.2f}).",
        f"Daily clearance is an AR(1) with φ = e^(−k); the whole-series φ of {_E['ar1']['phi']:.3f} means a {_E['ar1']['half_life']:.0f}-day half-life.",
        f"Doses on {_pct(_F1['dose_share'], 1)} of days plus clearance rebuild R² {_F1['r2']:.2f} of the VIX: a thresholded AR(1).",
    ],
    "glossary": [
        ["VIX", "An index of the S&P 500 volatility implied by option prices over the next month."],
        ["Dose", "Here, a one-day jump in the VIX larger than the threshold."],
        ["First-order clearance", "Removal at a rate proportional to the amount present, giving exponential decay."],
        ["Half-life", "Time for the excess to fall by half; ln 2 / k for first-order clearance."],
        ["Compartment", "A well-mixed tank with its own clearance rate."],
        ["AR(1)", "A process where each value is a fixed share φ of the last one plus a new shock."],
        ["Thresholded AR(1)", "An AR(1) whose shocks only enter when they exceed a threshold."],
    ],
}, order=2)


# ======================================================================== C. crash shapes
LM = 48                                                              # the live view's default landmark count


@functools.lru_cache(maxsize=32)
def _shapes(landmarks: int = LM):
    return S.crash_shapes(landmarks=landmarks)


@functools.lru_cache(maxsize=1)
def _by_depth() -> list[int]:
    eps = _shapes()["episodes"]
    return sorted(range(len(eps)), key=lambda i: -eps[i]["depth"])


def _name(e: dict) -> str:
    return e["label"] or f"{e['ticker']} {e['peak'][:4]}"


def _ep(rank: int, landmarks: int = LM):
    """The crash of a given depth rank (1 = deepest) and its index."""
    i = _by_depth()[rank - 1]
    return i, _shapes(landmarks)["episodes"][i]


def _below_half(y: np.ndarray) -> float:
    return float((np.asarray(y) < -0.5).mean())


def _dips(y: np.ndarray) -> int:
    """Number of separate stretches below half the depth."""
    b = np.r_[False, np.asarray(y) < -0.5, False]
    return int((np.diff(b.astype(int)) == 1).sum())


@figure("crash-normalise")
def _fig_normalise(p):
    C0 = _shapes()
    rank = p.num("rank", _by_depth().index(C0["main_spy"]) + 1, 1, len(C0["episodes"]), integer=True)
    L = p.num("landmarks", LM, 12, 128, integer=True)
    i, e = _ep(rank, L)
    price = S.load(e["ticker"])["AdjClose"].loc[e["peak"]:e["recovered"]].to_numpy()
    rel = price / price[0] - 1
    keep = _every(len(rel), 400)
    t = np.linspace(0, 1, len(rel))
    s = np.asarray(_shapes(L)["s"])
    return {
        "kind": "lines", "x": t[keep], "xlabel": "Share of the way from peak to recovery", "ylabel": "Change from the peak",
        "series": [{"name": f"{_name(e)}: price ÷ peak − 1", "color": "neutral", "y": rel[keep], "dash": True},
                   {"name": "Divided by the depth", "color": "s1", "y": rel[keep] / e["depth"]}],
        "points": [{"x": float(a), "y": float(b), "color": "s2", "r": 3} for a, b in zip(s, e["curve"])],
        "markers": [{"x": e["trough_at"], "label": "bottom"}],
        "notes": [f"{_name(e)}: peak {e['peak']}, bottom {e['trough']} ({_pct(e['trough_at'])} of the way), back at the peak "
                  f"{e['recovered']}: {e['days']} trading days and a {_pct(e['depth'])} fall (rank {rank} of {len(C0['episodes'])} by depth).",
                  f"Squeezing the {e['days']} days into 0–1 and dividing by the depth puts every crash in the same box; "
                  f"{L} landmarks (dots) turn this one into a list of {L} numbers."],
    }


@figure("crash-all")
def _fig_all(p):
    C = _shapes()
    s = np.asarray(C["s"])
    eps = sorted(C["episodes"], key=lambda e: e["ticker"] == "SPY")
    series = [_bg(_name(e), "s1" if e["ticker"] == "SPY" else "s3", e["curve"]) for e in eps]
    series += [{"name": "Middle 50%, upper edge", "color": "neutral", "y": C["q75"], "dash": True},
               {"name": "Middle 50%, lower edge", "color": "neutral", "y": C["q25"], "dash": True},
               {"name": "Mean shape", "color": "accent", "y": C["mean"]}]
    ta = np.array([e["trough_at"] for e in C["episodes"]])
    q1, q3 = np.quantile(ta, [0.25, 0.75])
    mean = np.asarray(C["mean"])
    return {
        "kind": "lines", "x": s, "series": series, "xlabel": "Share of the way from peak to recovery",
        "ylabel": "Change ÷ depth", "height": 300, "hlines": [{"y": -1, "label": "the bottom", "color": "neutral"}],
        "legend": [{"name": "Mean shape", "color": "accent"}, {"name": "Middle 50%", "color": "neutral", "dash": True},
                   {"name": "SPY crashes", "color": "s1"}, {"name": "Single-stock crashes", "color": "s3"}],
        "notes": [f"{C['counts']['total']} completed crashes: {C['counts']['spy']} in SPY (at least {_pct(C['params']['spy_depth'])} deep) "
                  f"and {C['counts']['stocks']} in single stocks (at least {_pct(C['params']['stock_depth'])}).",
                  f"The mean shape only reaches {_n(mean.min())}, not −1, because the bottoms come at different times: "
                  f"between {_pct(q1)} and {_pct(q3)} of the way for the middle half."],
    }


@figure("crash-variance")
def _fig_variance(p):
    from ..shape_service import jaw_model
    C = _shapes()
    with np.errstate(all="ignore"):     # Accelerate BLAS raises spurious matmul warnings (see core/pca.py)
        jm, _ = jaw_model()
    jaw3 = float(np.cumsum(jm.model.explained)[:3][-1])
    v, c = np.asarray(C["variance"]), np.asarray(C["cumulative"])
    return {
        "kind": "bars", "labels": [f"Mode {k + 1}" for k in range(len(v))], "ylabel": "Share of variance",
        "series": [{"name": "This mode", "color": "s1", "values": v}, {"name": "Cumulative", "color": "s2", "values": c}],
        "notes": [f"Mode 1 explains {_pct(v[0])}, mode 2 {_pct(v[1])} and mode 3 {_pct(v[2])}: {_pct(c[2])} together.",
                  f"ShapeLab's jawlines reach {_pct(jaw3)} with three modes. Crashes vary in more independent ways than jaws do."],
    }


@figure("crash-modes")
def _fig_modes(p):
    k, b = p.num("mode", 1, 1, 3, integer=True), p.num("b", 2, 0.5, 3)
    C = _shapes()
    model = C["_model"]
    s = np.asarray(C["s"])
    vec = np.zeros(k)
    vec[k - 1] = b
    plus, minus = model.sample(vec), model.sample(-vec)
    tp, tm = s[int(np.argmin(plus))], s[int(np.argmin(minus))]
    label = {1: "U ↔ V", 2: "early ↔ late bottom", 3: "double dip ↔ single bottom"}[k]
    return {
        "kind": "lines", "x": s, "xlabel": "Share of the way from peak to recovery", "ylabel": "Change ÷ depth",
        "series": [{"name": "Mean shape", "color": "neutral", "y": C["mean"], "dash": True},
                   {"name": f"−{b:g} sd along mode {k}", "color": "s1", "y": minus},
                   {"name": f"+{b:g} sd along mode {k}", "color": "s2", "y": plus}],
        "notes": [f"Mode {k} ({label}) carries {_pct(C['variance'][k - 1])} of the variation. At +{b:g} sd the bottom is at "
                  f"{_pct(tp)} of the way; at −{b:g} sd at {_pct(tm)}.",
                  f"Time below half the depth: {_pct(_below_half(plus))} at +{b:g} sd in {_dips(plus)} stretch{'es' if _dips(plus) != 1 else ''}, "
                  f"{_pct(_below_half(minus))} at −{b:g} sd in {_dips(minus)} stretch{'es' if _dips(minus) != 1 else ''}. "
                  "More time near the bottom means a flatter U; two stretches mean two dips."],
    }


@figure("crash-space")
def _fig_space(p):
    C = _shapes()
    pts = []
    for e in sorted(C["episodes"], key=lambda e: e["ticker"] == "SPY"):
        pts.append({"x": float(e["scores"][0]), "y": float(e["scores"][1]), "color": "s1" if e["ticker"] == "SPY" else "s3",
                    "r": 5 if e["label"] else 3.5, "name": f"{_name(e)} (−{_pct(e['depth'])})",
                    **({"label": e["label"]} if e["label"] else {})})
    named = [e for e in C["episodes"] if e["label"]]
    desc = "; ".join(f"{e['label']} at ({_signed(e['scores'][0])}, {_signed(e['scores'][1])})" for e in named)
    r = C["correlations"]["mode1_depth"]
    return {
        "kind": "scatter", "points": pts, "xlabel": "Mode 1 score (sd): U ← → V", "ylabel": "Mode 2 score (sd): early ← → late bottom",
        "height": 320, "legend": [{"name": f"SPY ({C['counts']['spy']})", "color": "s1"},
                                  {"name": f"Single stocks ({C['counts']['stocks']})", "color": "s3"}],
        "notes": [f"Named SPY crashes: {desc}.",
                  f"Mode 1 against depth gives r = {_n(r)}: "
                  + ({"U": "deeper crashes lean towards the long, flat U.", "V": "deeper crashes lean towards the sharp V."}.get(C["depth_lean"], "depth barely sets the shape."))],
    }


@figure("crash-timing")
def _fig_timing(p):
    C = _shapes()
    ta = np.array([e["trough_at"] for e in C["episodes"]])
    m2 = np.array([e["scores"][1] for e in C["episodes"]])
    slope, icpt = np.polyfit(ta, m2, 1)
    xs = np.array([ta.min(), ta.max()])
    return {
        "kind": "scatter", "xlabel": "Where the bottom falls (share of the way)", "ylabel": "Mode 2 score (sd)", "height": 300,
        "points": [{"x": float(a), "y": float(b), "color": "s1" if e["ticker"] == "SPY" else "s3", "r": 3.5, "name": _name(e)}
                   for a, b, e in zip(ta, m2, C["episodes"])],
        "lines": [{"x": xs, "y": icpt + slope * xs, "color": "s2", "name": "Least-squares line"}],
        "legend": [{"name": "SPY", "color": "s1"}, {"name": "Single stocks", "color": "s3"}, {"name": "Least-squares line", "color": "s2"}],
        "notes": [f"Mode 2 tracks the timing of the bottom: r = {_n(C['correlations']['mode2_trough_at'])} over {len(ta)} crashes.",
                  f"Mode 1 is nearly unrelated to timing (r = {_n(C['correlations']['mode1_trough_at'])}): PCA has separated "
                  "how the fall is shaped from when it bottoms."],
    }


@figure("crash-rebuild")
def _fig_rebuild(p):
    C = _shapes()
    model = C["_model"]
    kmax = len(model.sd)
    k = p.num("k", 3, 0, kmax, integer=True)
    rank = p.num("rank", _by_depth().index(C["main_spy"]) + 1, 1, len(C["episodes"]), integer=True)
    i, e = _ep(rank)
    x = np.asarray(e["curve"])[None, :]
    reb = model.rebuild(x, k)[0]
    rmse = float(np.sqrt(((x[0] - reb) ** 2).mean()))
    rmse0 = float(np.sqrt(((x[0] - model.mean) ** 2).mean()))
    return {
        "kind": "lines", "x": C["s"], "xlabel": "Share of the way from peak to recovery", "ylabel": "Change ÷ depth",
        "series": [{"name": "Mean shape", "color": "neutral", "y": model.mean, "dash": True},
                   {"name": _name(e), "color": "s1", "y": x[0]},
                   {"name": f"Rebuilt from {k} mode{'s' if k != 1 else ''}", "color": "s2", "y": reb}],
        "notes": [(f"{_name(e)} is described by {k} number{'s' if k != 1 else ''} instead of {len(x[0])}: "
                   f"typical error {rmse:.3f} of the depth, against {rmse0:.3f} for the mean shape alone." if k else
                   f"With no modes every crash is rebuilt as the mean shape; for {_name(e)} the typical error is {rmse0:.3f} of the depth."),
                  f"Over all {C['counts']['total']} crashes the typical error with {k} modes is {C['rebuild_rmse'][k]:.3f}."],
    }


@functools.lru_cache(maxsize=1)
def _open_drawdowns():
    out = []
    for tk in S.TICKERS:
        p = S.load(tk)["AdjClose"]
        out.append((tk, float(1 - p.iloc[-1] / p.max()), str(p.idxmax().date()), str(p.index[-1].date())))
    return out


@figure("crash-censored")
def _fig_censored(p):
    rows = _open_drawdowns()
    under = [r for r in rows if r[1] > 0]
    deep = max(rows, key=lambda r: r[1])
    par = _shapes()["params"]
    big = [r for r in rows if r[1] >= (par["spy_depth"] if r[0] == "SPY" else par["stock_depth"])]
    return {
        "kind": "bars", "labels": [r[0] for r in rows], "ylabel": "Below the all-time high at the end of the data (%)",
        "series": [{"name": "Open drawdown", "color": "warn", "values": [100 * r[1] for r in rows]}],
        "notes": [f"On {rows[0][3]}, {len(under)} of {len(rows)} tickers are below an earlier peak; the deepest is {deep[0]}, "
                  f"{_pct(deep[1])} below its high of {deep[2]}.",
                  (f"{len(big)} of them ({', '.join(r[0] for r in big)}) are deep enough to count as crashes, but none can enter the sample "
                   "until the price is back at its peak." if big else
                   f"None is deep enough to count yet ({_pct(par['spy_depth'])} for SPY, {_pct(par['stock_depth'])} for stocks); "
                   "even a deep one could not enter the sample until the price is back at its peak.")],
    }


# ------------------------------------------------------------------------ C: text
_C = _shapes()
_main = _C["episodes"][_C["main_spy"]]
_cr = _C["correlations"]

explainer("crossover/crash-shapes", {
    "title": "Do crashes have a shape? PCA of drawdowns",
    "kicker": "Time Series Lab · ShapeLab × markets",
    "summary": f"ShapeLab builds a jawline model by lining up outlines and finding their main ways of varying. The same recipe "
               f"works for market crashes: stretch each of {_C['counts']['total']} completed drawdowns to the same length and depth, "
               "sample it at landmarks, and run PCA. Three modes say whether a crash is a U or a V, when it bottoms, and whether it dips twice.",
    "sections": [
        {"id": "sizes", "title": "Every crash is a different size", "blocks": [
            {"type": "p", "text": f"{_name(_main)} lasted {_main['days']} trading days from peak to recovery and fell {_pct(_main['depth'])}. "
                                  f"The shortest in the sample took {min(e['days'] for e in _C['episodes'])} days and the shallowest fell "
                                  f"{_pct(min(e['depth'] for e in _C['episodes']))}. To compare their <i>shapes</i>, "
                                  "size has to be taken out first, just as ShapeLab removes position, scale and rotation before "
                                  "comparing jaws."},
            {"type": "cards", "items": [
                {"title": "A completed crash", "text": "From a peak, down to the lowest close, and back to the peak's price."},
                {"title": "Shape", "text": "What is left after the length and the depth are divided out."},
                {"title": "Landmarks", "text": "The same number of evenly spaced points on every crash, so shapes become vectors."},
            ]},
        ]},
        {"id": "normalise", "title": "Turning a drawdown into a shape", "blocks": [
            {"type": "steps", "items": [
                "Find a peak and the first later day the price is back at it. Everything between is one episode.",
                f"Keep it if it fell far enough: {_pct(_C['params']['spy_depth'])} for SPY, {_pct(_C['params']['stock_depth'])} for single stocks, which swing more.",
                "Divide prices by the peak and subtract 1, so the episode starts and ends at 0.",
                "Divide by the depth, so the lowest point is exactly −1.",
                "Rescale time to run from 0 to 1 and read the curve at evenly spaced landmarks.",
            ]},
            {"type": "eq", "text": "y(s) = (P(t₀ + s T) / P(t₀) − 1) / depth,   s ∈ [0, 1]\nx = (y(s₁), y(s₂), …, y(s_N)),   s_j = (j − 1) / (N − 1)",
             "where": [["t₀, T", "start and length of the episode in trading days"], ["depth", "1 − lowest price / peak price"],
                       ["N", f"number of landmarks ({LM} in the live model)"], ["x", "the crash as a vector of N numbers"]]},
            {"type": "figure", "figure": "crash-normalise", "title": "One crash, before and after",
             "caption": "Dashed: the raw fall from the peak. Solid: divided by the depth. Dots: the landmarks. Pick any crash by depth rank.",
             "controls": [{"k": "rank", "label": "Crash by depth (1 = deepest)", "min": 1, "max": _C["counts"]["total"], "step": 1,
                           "value": _by_depth().index(_C["main_spy"]) + 1},
                          {"k": "landmarks", "label": "Landmarks", "min": 12, "max": 128, "step": 4, "value": LM}]},
            {"type": "callout", "tone": "info", "title": "Why no rotation",
             "text": "Procrustes alignment of jaws also removes rotation, because a jaw tilted on the page is still the same jaw. "
                     "Here the axes are time and price, which cannot be swapped, so only level, depth and length are divided out."},
        ]},
        {"id": "all", "title": "All the crashes on one chart", "blocks": [
            {"type": "figure", "figure": "crash-all", "title": "Every crash, normalised",
             "caption": "Each thin line is one completed crash. The thick line is the mean; the dashed lines hold the middle half at each landmark."},
            {"type": "p", "text": "The curves share a start, an end and a bottom at −1, but they are not one shape. Some drop and recover "
                                  "in a sharp V. Others fall fast and crawl back; others sag slowly into a wide U. PCA finds the few "
                                  "directions that account for most of these differences."},
        ]},
        {"id": "pca", "title": "PCA, exactly as for jawlines", "blocks": [
            {"type": "p", "text": "Stack the crash vectors as rows of a matrix, subtract the mean shape and take the singular value "
                                  "decomposition. The first <b>mode</b> is the direction in which crashes differ most; the second is "
                                  "the largest remaining difference at right angles to the first, and so on. ShapeLab's jaw model "
                                  "runs the same decomposition on aligned outlines."},
            {"type": "eq", "text": "X − x̄ = U Σ Vᵀ\nx ≈ x̄ + Σₖ bₖ σₖ φₖ",
             "where": [["x̄", "the mean crash shape"], ["φₖ", "mode k: row k of Vᵀ, a shape of unit length"],
                       ["σₖ", "standard deviation of the crashes along mode k"], ["bₖ", "a crash's score on mode k, in standard deviations"]]},
            {"type": "figure", "figure": "crash-variance", "title": "How much each mode explains",
             "caption": "First bar: each mode's share of the total variation. Second bar: the running total."},
        ]},
        {"id": "modes", "title": "What the modes mean", "blocks": [
            {"type": "p", "text": "A mode is easiest to read by walking along it: add and subtract a few standard deviations of the mode "
                                  "to the mean shape and look at what changes."},
            {"type": "figure", "figure": "crash-modes", "title": "Walk along a mode",
             "caption": "The mean shape plus and minus b standard deviations of one mode. Modes are oriented so + is the sharper V on "
                        "mode 1 and the later bottom on mode 2.",
             "controls": [{"k": "mode", "label": "Mode", "min": 1, "max": 3, "step": 1, "value": 1},
                          {"k": "b", "label": "Distance", "min": 0.5, "max": 3, "step": 0.5, "value": 2, "unit": "sd"}]},
            {"type": "cards", "items": [
                {"title": "Mode 1: U or V", "text": "How long the crash spends near the bottom. A V touches it and turns; a U stays there."},
                {"title": "Mode 2: early or late", "text": "Whether the bottom comes early, with a long recovery, or late, after a long slide."},
                {"title": "Mode 3: one dip or two", "text": "Whether the fall pauses and rallies before a second low."},
            ]},
            {"type": "figure", "figure": "crash-space", "title": "Crashes in shape space",
             "caption": "Each dot is a crash at its scores on modes 1 and 2. Labelled dots are the named SPY crashes; hover for the others."},
            {"type": "figure", "figure": "crash-timing", "title": "Mode 2 is the timing of the bottom",
             "caption": "Where each crash bottoms, against its mode-2 score."},
            {"type": "quiz", "q": "A crash scores +2 on mode 2 and about 0 on the others. What does it look like?",
             "options": ["A sharp V with an average bottom", "An average crash whose bottom comes late, after a long slide",
                         "Two separate dips"], "answer": 1,
             "why": f"Mode 2 moves the bottom: positive scores mean later troughs (r = {_n(_cr['mode2_trough_at'])} with trough timing). "
                    "Scores near 0 on the other modes keep the rest of the shape average."},
            {"type": "try", "label": "Build a sharp V in the live model", "params": {"b1": 2, "b2": 0, "b3": 0}},
        ]},
        {"id": "rebuild", "title": "A crash in three numbers", "blocks": [
            {"type": "p", "text": "Because the modes are ordered by importance, the first few scores rebuild most of a crash. This is the "
                                  "compression ShapeLab uses to describe a jaw with a handful of numbers instead of every outline point."},
            {"type": "figure", "figure": "crash-rebuild", "title": "Rebuild a crash from k modes",
             "caption": "The crash, the mean shape (dashed) and the rebuild from the first k scores.",
             "controls": [{"k": "k", "label": "Modes used", "min": 0, "max": len(_C["_model"].sd), "step": 1, "value": 3},
                          {"k": "rank", "label": "Crash by depth (1 = deepest)", "min": 1, "max": _C["counts"]["total"], "step": 1,
                           "value": _by_depth().index(_C["main_spy"]) + 1}]},
            {"type": "try", "label": "Keep only the deep crashes", "params": {"spy_depth": 0.2, "stock_depth": 0.35, "landmarks": LM},
             "note": "Fewer, larger crashes: see whether the modes keep their meaning."},
        ]},
        {"id": "limits", "title": "Survivors and unfinished falls", "blocks": [
            {"type": "p", "text": "Only <i>completed</i> crashes have a shape, because the shape needs an end point. A fall that has not "
                                  "recovered by the last day of data is left out, however deep it is."},
            {"type": "figure", "figure": "crash-censored", "title": "Falls the sample cannot see yet",
             "caption": "How far each ticker is below its highest close on the last day of data."},
            {"type": "callout", "tone": "warn", "title": "Survivorship and censoring",
             "text": f"The {len(S.TICKERS)} tickers were chosen today, and every one still trades, so firms that crashed and never "
                     "came back are missing by construction. Crashes that never recovered, or have not yet, cannot enter the sample "
                     "either. Both biases leave out the slowest recoveries. The modes describe how past completed crashes "
                     "looked; they do not predict how a crash under way will end, and they are not investment advice."},
            {"type": "quiz", "q": "Why does the sample leave out a stock that is 30% below its peak on the last day of data?",
             "options": ["It is not deep enough", "It has not recovered, so the episode has no end point to scale time to",
                         "Single stocks are excluded"], "answer": 1,
             "why": "Time is scaled from the peak to the recovery. Without a recovery there is no T, so the fall cannot be turned into a shape."},
        ]},
    ],
    "takeaways": [
        "A crash becomes a shape by dividing out its level, depth and length, then sampling it at evenly spaced landmarks.",
        "PCA of the landmark vectors is the same decomposition ShapeLab uses for jawlines, without the rotation step.",
        f"Three modes explain {_pct(_C['cumulative'][2])} of the variation between {_C['counts']['total']} crashes: U against V, "
        "early against late bottom, one dip against two.",
        f"Mode 2 tracks where the bottom falls (r = {_n(_cr['mode2_trough_at'])}); mode 1 against depth gives r = {_n(_cr['mode1_depth'])}.",
        "Only completed crashes in surviving stocks are in the sample, which leaves out the slowest recoveries of all.",
    ],
    "glossary": [
        ["Drawdown", "A fall from a running peak, as a share of that peak."],
        ["Completed crash", "A drawdown whose price has returned to the earlier peak."],
        ["Landmark", "One of N evenly spaced points at which every normalised crash is sampled."],
        ["PCA", "Principal component analysis: the orthogonal directions of greatest variation in a set of vectors."],
        ["Mode", "One such direction, read as a change of shape added to the mean."],
        ["Score", "How far a crash lies along a mode, in standard deviations."],
        ["Survivorship bias", "Distortion from studying only the cases that lasted to the present."],
        ["Censoring", "Leaving out cases whose outcome is not known yet."],
    ],
}, order=3)
