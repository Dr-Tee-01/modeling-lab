"""Explainers for the Markets section: volatility, returns and neural networks.

Every figure is computed from the daily price CSVs with the same code the live views
use (modlab.markets and stocklab). Nothing slow runs here: the volatility walk-forward
is the live view's own memoised call (about a tenth of a second), and the return and
neural pages use direct computations on prices instead of the slow walk-forward fits
and PyTorch training, which stay in the live views.
"""
from __future__ import annotations

import functools
import inspect
import math

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy

from stocklab import evaluate as E
from stocklab import models as SM

from ..markets import common as C
from ..markets import garch as G
from ..markets import neural as NN
from ..markets import returns as RET
from ..markets import stylized as STY
from ..markets import volatility as VOL
from .registry import explainer, figure

TICKER = "SPY"
START = C.TEST_YEARS[0]                       # the live views' default first test year
DAYS = E.DAYS
NAME = {k: v[0] for k, v in C.MODEL_NAMES.items()}


def _default(fn, arg):
    """A function's default argument value, so the prose quotes the code's own settings."""
    return inspect.signature(fn).parameters[arg].default


LAMBDA = _default(VOL.ewma_var, "lam")                   # RiskMetrics decay
GARCH_YEARS = _default(G.garch_year, "window_years")      # GARCH estimation window
RIDGE_ALPHAS = _default(SM.fit_ridge, "alphas")
RNN_HIDDEN = _default(SM.fit_rnn, "hidden")
RNN_PATIENCE = _default(SM.fit_rnn, "patience")
TRAP_HIDDEN = 50                                          # LSTM(50) in modlab.markets.neural.lstm_trap
TRAP_WINDOW = 60                                          # the tutorial's default window (api_markets._parse_trap)
COST = E.COST_BPS
TICKER_CTRL = {"k": "ticker", "label": "Ticker", "options": [[t, t] for t in C.TICKERS], "value": TICKER}
CONF = {"99": 0.01, "95": 0.05}                           # VaR confidence -> breach rate, as the live view's options
CONF_CTRL = {"k": "conf", "label": "Confidence", "options": [[k, f"{k}%"] for k in CONF], "value": "99"}


def _ticker(p) -> str:
    return p.choice("ticker", TICKER, C.TICKERS)


def _years(idx) -> np.ndarray:
    """Dates as fractional years, for a numeric x axis."""
    idx = pd.DatetimeIndex(idx)
    return np.asarray(idx.year + (idx.dayofyear - 1) / 365.25, dtype=float)


def _ann(logvar) -> np.ndarray:
    """Log daily variance (return units²) to annualised volatility in %."""
    return 100 * np.sqrt(DAYS * np.exp(np.asarray(logvar, dtype=float)))


def _half_life(p: float) -> float:
    return math.log(0.5) / math.log(p) if 0 < p < 1 else math.inf


def _r2(a, p) -> float:
    a, p = np.asarray(a, float), np.asarray(p, float)
    return float(1 - np.sum((a - p) ** 2) / np.sum((a - a.mean()) ** 2))


@functools.lru_cache(maxsize=1)
def _end_year() -> int:
    """The last test year the live views use (as the markets catalog)."""
    return min(int(C.last_date().year), C.YEAR_MAX)


# ===================================================================== volatility: data
@functools.lru_cache(maxsize=16)
def _vol(ticker: str = TICKER, alpha: float = 0.01) -> dict:
    """The live volatility view's default walk-forward (memoised in modlab.markets)."""
    return VOL.volatility(ticker, tuple(VOL.VOL_MODELS), START, _end_year(), float(alpha), "har_x")[0]


@functools.lru_cache(maxsize=16)
def _vol_dates(ticker: str = TICKER, alpha: float = 0.01) -> np.ndarray:
    return _years(_vol(ticker, alpha)["series"]["dates"])


@functools.lru_cache(maxsize=8)
def _abs_returns(ticker: str = TICKER):
    r = C.panel(ticker, with_vix=False)["ret"].dropna()
    return r.index, np.abs(r.values) * 100


# ===================================================================== volatility: figures
@figure("vol-cluster")
def _vol_cluster(p):
    t = _ticker(p)
    idx, a = _abs_returns(t)
    n = len(a)
    roll = pd.Series(a).rolling(21).mean().values
    k = C.stride(n, 1400)
    g = np.arange(n) // k
    peak = pd.Series(a).groupby(g).max().values
    level = pd.Series(roll).groupby(g).mean().values
    x = _years(idx[::k])[: len(peak)]
    # clustering: the largest 1% of days, and how many sit within a month of another one
    q = np.quantile(a, 0.99)
    pos = np.where(a >= q)[0]

    def near(ps):
        gaps = np.diff(np.sort(ps))
        return float(np.mean(np.minimum(np.r_[np.inf, gaps], np.r_[gaps, np.inf]) <= 21))

    rng = np.random.default_rng(0)
    shuffled = np.mean([near(rng.choice(n, len(pos), replace=False)) for _ in range(50)])
    return {
        "kind": "lines", "x": x, "xdate": True, "xlabel": "Year", "ylabel": "Size of the daily move, %", "ymin": 0,
        "series": [{"name": f"Largest |return| in each {k}-day block", "color": "s1", "y": peak, "thin": True},
                   {"name": "21-day average |return|", "color": "s2", "y": np.nan_to_num(level, nan=float(np.nanmean(a)))}],
        "notes": [f"The largest 1% of {t}'s daily moves ({len(pos)} days, each above {q:.1f}%) are not spread evenly: "
                  f"{100 * near(pos):.0f}% of them fall within 21 trading days of another one.",
                  f"If those days were scattered at random through the {n:,} days, only about {100 * shuffled:.0f}% would."],
    }


@figure("vol-acf")
def _vol_acf(p):
    t = _ticker(p)
    s = STY.stylized(t, 100)[0]
    lags, ar, aa, band = np.asarray(s["lags"]), np.asarray(s["acf_r"]), np.asarray(s["acf_abs"]), float(s["band"])
    outside = int(np.sum(np.abs(ar) > band))
    return {
        "kind": "lines", "x": lags, "xlabel": "Lag (trading days)", "ylabel": "Autocorrelation",
        "series": [{"name": "Size of the move |r|", "color": "s2", "y": aa, "area": True},
                   {"name": "Return r", "color": "s1", "y": ar}],
        "hlines": [{"y": band, "label": "95% noise band", "color": "neutral"}, {"y": -band, "color": "neutral"}],
        "notes": [f"For {t}, today's |r| is still correlated with |r| {lags[-1]} days later (ρ = {aa[-1]:.2f}, "
                  f"{aa[-1] / band:.0f}× the noise band of ±{band:.3f}). Calm and stormy spells last for months.",
                  f"The return's own autocorrelations are small: the largest is {ar[np.argmax(np.abs(ar))]:+.2f} at lag "
                  f"{lags[np.argmax(np.abs(ar))]}, and {outside} of {len(lags)} lags leave the band. Real but tiny patterns, "
                  "unlike the steady memory of |r|."],
    }


@figure("vol-ewma")
def _vol_ewma(p):
    lam = p.num("lam", LAMBDA, 0.8, 0.99)
    k = np.arange(1, 81)
    w = (1 - lam) * lam ** (k - 1)
    flat = np.where(k <= 21, 1 / 21, 0.0)
    return {
        "kind": "lines", "x": k, "xlabel": "Days ago", "ylabel": "Weight on that day's squared return", "ymin": 0,
        "series": [{"name": f"EWMA, λ = {lam:.2f}", "color": "s2", "y": w, "area": True},
                   {"name": "21-day average", "color": "neutral", "y": flat, "dash": True}],
        "notes": [f"With λ = {lam:.2f} a day's weight halves every {_half_life(lam):.1f} days; "
                  f"the last 21 days carry {100 * (1 - lam ** 21):.0f}% of the total weight.",
                  f"Yesterday counts {w[0] / (1 / 21):.1f}× as much as in the 21-day average, and nothing ever drops out suddenly."],
    }


@figure("vol-garch")
def _vol_garch(p):
    a, b = p.num("alpha", 0.10, 0.02, 0.25), p.num("beta", 0.85, 0.6, 0.97)
    end = _end_year()
    _, w0, a0, b0 = _vol()["params"]["garch"][end]
    lr = w0 / (1 - a0 - b0)                          # fitted long-run daily variance (%²)
    shock = 4.0                                       # a move of four long-run standard deviations
    h = np.arange(0, 121)

    def path(al, be):
        pers = al + be
        w = lr * (1 - pers) if pers < 1 else w0
        s1 = w + al * shock ** 2 * lr + be * lr        # the day after the shock
        v = np.empty(len(h))
        v[0] = lr
        for i in range(1, len(h)):
            v[i] = s1 if i == 1 else w + pers * v[i - 1]
        return np.sqrt(DAYS * v)

    mine, fitted = path(a, b), path(a0, b0)
    cap = 1.15 * max(mine[1], fitted[1])
    pers = a + b
    return {
        "kind": "lines", "x": h, "xlabel": "Days after the shock", "ylabel": "Expected volatility, % a year", "ymin": 0,
        "series": [{"name": f"Your GARCH: α = {a:.2f}, β = {b:.2f}", "color": "s2", "y": np.minimum(mine, cap), "area": True},
                   {"name": f"Fitted to {TICKER} for {end}: α = {a0:.2f}, β = {b0:.2f}", "color": "s1", "y": fitted, "dash": True}],
        "hlines": [{"y": math.sqrt(DAYS * lr), "label": "long-run level", "color": "neutral"}],
        "notes": [(f"α + β = {pers:.2f}: half of the extra variance is gone after {_half_life(pers):.0f} days, "
                   f"and volatility returns to its long-run {math.sqrt(DAYS * lr):.0f}% a year.") if pers < 1 else
                  f"α + β = {pers:.2f} ≥ 1: variance never returns to a long-run level, so a shock is remembered for ever "
                  "(the chart is cut at the top). Fitted models stay below 1.",
                  f"The dashed curve is the model fitted on the {GARCH_YEARS} years before {end}: α + β = {a0 + b0:.3f}, "
                  f"a half-life of {_half_life(a0 + b0):.0f} days."],
    }


@figure("vol-garch-years")
def _vol_garch_years(p):
    g = _vol()["params"]["garch"]
    ys = sorted(g)
    a = np.array([g[y][2] for y in ys])
    b = np.array([g[y][3] for y in ys])
    return {
        "kind": "lines", "x": ys, "xlabel": "Test year (fitted on the years before it)", "ylabel": "Parameter", "ymin": 0, "ymax": 1,
        "series": [{"name": "α + β (persistence)", "color": "accent", "y": a + b},
                   {"name": "β (memory of past variance)", "color": "s1", "y": b},
                   {"name": "α (reaction to yesterday's shock)", "color": "s2", "y": a, "area": True}],
        "notes": [f"Refitted every year, persistence stays between {np.min(a + b):.3f} and {np.max(a + b):.3f}.",
                  f"α goes from {a[0]:.2f} to {a[-1]:.2f} over the test years and β from {b[0]:.2f} to {b[-1]:.2f}"
                  + (": recent fits react faster to news and forget faster." if a[-1] > a[0] and b[-1] < b[0] else ".")],
    }


@functools.lru_cache(maxsize=1)
def _har_now():
    V, _ = VOL.vol_frame(TICKER)
    end = _end_year()
    coef = _vol()["params"]["har"]["har_x"][end]
    cols = VOL.HAR_COLS["har_x"]
    x = V[cols].iloc[-1].values
    return V.index[-1], end, np.asarray(coef), x, float(coef[0] + x @ coef[1:])


@figure("vol-har-flow")
def _vol_har_flow(p):
    last, end, coef, x, f = _har_now()
    labels = [("Yesterday", "log RV, 1 day"), ("Last week", "log RV, 5-day mean"), ("Last month", "log RV, 22-day mean"),
              ("VIX²", "log implied variance"), ("Yesterday's fall", "squared down-move")]
    big = max(abs(c) for c in coef[1:5])
    nodes, edges = [], []
    for i, ((t, sub), c, v) in enumerate(zip(labels, coef[1:], x)):
        val = f"{v:.2f}" if i == 4 else f"σ ≈ {_ann(v):.0f}% a year"
        nodes.append({"id": f"in{i}", "x": 20, "y": 14 + 58 * i, "w": 190, "h": 46, "title": t, "value": val,
                      "fill_color": ["s1", "s2", "s3", "s4", "s4"][i]})
        edges.append({"from": f"in{i}", "to": "out", "label": f"{c:+.2f}",
                      "width": 1.5 + 9 * min(1.0, abs(c) / big), "flow": i < 4, "color": ["s1", "s2", "s3", "s4", "s4"][i]})
    nodes.append({"id": "out", "x": 400, "y": 105, "w": 180, "h": 96, "title": "Tomorrow", "value": f"σ ≈ {_ann(f):.0f}% a year",
                  "sub": "log RV forecast", "hot": True})
    hsum = coef[1] + coef[2] + coef[3]
    return {
        "kind": "flow", "height": 330, "nodes": nodes, "edges": edges,
        "texts": [{"x": 300, "y": 322, "text": f"HAR-X fitted for test year {end}; inputs as at the close on {last:%d %B %Y}", "cls": "n-sub"}],
        "notes": [(f"The VIX carries the largest weight ({coef[4]:.2f}); the day, week and month terms share {hsum:.2f} between them. "
                   "Once the options market's own forecast is known, past realised variance adds less.") if coef[4] > max(coef[1:4]) else
                  f"The day, week and month terms carry {coef[1]:.2f}, {coef[2]:.2f} and {coef[3]:.2f}; the VIX adds {coef[4]:.2f}.",
                  f"With {TICKER}'s inputs on {last:%d %B %Y} the model forecasts realised volatility of about {_ann(f):.0f}% a year for the next day "
                  "(before the small bias correction the live model adds)."],
    }


@figure("vol-har-parts")
def _vol_har_parts(p):
    year = p.num("year", 2020, C.YEAR_MIN, C.YEAR_MAX, integer=True)
    V, _ = VOL.vol_frame(TICKER)
    sub = V[V.index.year == year]
    if len(sub) < 30:
        sub = V[V.index.year == V.index.year.max()]
        year = int(sub.index.year[0])
    cols = [("lrv_d", "Yesterday (1 day)", "s1", True), ("lrv_w", "Last week (5 days)", "s2", False),
            ("lrv_m", "Last month (22 days)", "s3", False), ("lvix2", "VIX implied", "s4", False)]
    ok = sub["y"].notna()
    corr = {c: float(np.corrcoef(sub.loc[ok, c], sub.loc[ok, "y"])[0, 1]) for c, *_ in cols}
    sd = {c: float(sub[c].std()) for c, *_ in cols}
    return {
        "kind": "lines", "x": _years(sub.index), "xdate": True, "xlabel": "Date", "ylabel": "Volatility, % a year (log scale)", "ylog": True,
        "series": [{"name": n, "color": col, "y": _ann(sub[c].values), "thin": thin} for c, n, col, thin in cols],
        "notes": [f"In {year} the one-day measure swings {sd['lrv_d'] / sd['lrv_m']:.1f}× as much as the monthly average "
                  f"(standard deviation of log variance {sd['lrv_d']:.2f} against {sd['lrv_m']:.2f}).",
                  f"Correlation with the next day's log variance in {year}: daily {corr['lrv_d']:.2f}, weekly {corr['lrv_w']:.2f}, "
                  f"monthly {corr['lrv_m']:.2f}, VIX {corr['lvix2']:.2f}. HAR lets a regression weigh them."],
    }


@figure("vol-r2")
def _vol_r2(p):
    t = _ticker(p)
    v = _vol(t)
    ms = VOL.VOL_MODELS
    r2 = [100 * v["models"][m]["r2_log"] for m in ms]
    best, base = v["summary"]["best"], v["summary"]["baseline"]
    return {
        "kind": "bars", "labels": [C.MODEL_NAMES[m][1] for m in ms], "ylabel": "Out-of-sample R², %",
        "series": [{"name": "R² of log variance", "color": "s2", "values": r2}],
        "notes": [f"Over {v['n_test']:,} test days ({v['years'][0]}–{v['years'][-1]}), {NAME[best]} explains "
                  f"{100 * v['models'][best]['r2_log']:.0f}% of the day-to-day variation in {t}'s log variance, "
                  f"{100 * v['summary']['gain_over_baseline']:.0f} points more than the {NAME[base].lower()}.",
                  (f"Even the simplest, the {NAME['hist21'].lower()}, reaches {100 * v['models']['hist21']['r2_log']:.0f}%. "
                   "Volatility is forecastable in a way returns are not.") if min(r2) > 0 else
                  f"The weakest model scores {min(r2):.0f}%: worse than forecasting the test-period average."],
    }


@figure("vol-var")
def _vol_var(p):
    year = p.num("year", 2020, START, C.YEAR_MAX, integer=True)
    alpha = CONF[p.choice("conf", "99", list(CONF))]
    v, x = _vol(TICKER, alpha), _vol_dates(TICKER, alpha)
    s = v["series"]
    years = np.floor(x).astype(int)
    if not np.any(years == year):
        year = int(years.max())
    m = years == year
    rn, vn, vf = (np.asarray(s[k])[m] * 100 for k in ("next_return", "var_normal", "var_fhs"))
    bn, bf = rn < vn, rn < vf
    lvl = 100 * (1 - v["var"]["alpha"])
    return {
        "kind": "lines", "x": x[m], "xdate": True, "xlabel": "Forecast made on", "ylabel": "Next day's return, %",
        "series": [{"name": "Next day's return", "color": "neutral", "y": rn, "thin": True},
                   {"name": f"{lvl:.0f}% VaR, normal", "color": "warn", "y": vn},
                   {"name": f"{lvl:.0f}% VaR, filtered historical", "color": "s2", "y": vf, "dash": True}],
        "points": [{"x": float(a), "y": float(b), "color": "bad"} for a, b in zip(x[m][bn], rn[bn])],
        "notes": [f"In {year} the next-day loss went past the normal VaR on {int(bn.sum())} of {int(m.sum())} days "
                  f"and past the filtered-historical VaR on {int(bf.sum())}; the target is {100 - lvl:.0f}%, about {m.sum() * (100 - lvl) / 100:.1f} days.",
                  f"Over all test years: {v['var']['normal']['exceed']} breaches for the normal VaR and {v['var']['fhs']['exceed']} "
                  f"for the filtered one, out of {v['var']['normal']['n']:,} days."],
    }


def _kupiec_p(x, n: int, a: float) -> np.ndarray:
    """Kupiec's p-value for x breaches in n days (vectorised; as stocklab.evaluate.kupiec)."""
    x = np.asarray(x, dtype=float)
    ph = x / n
    lr = -2 * ((n - x) * np.log(1 - a) + x * np.log(a) - xlogy(n - x, 1 - ph) - xlogy(x, ph))
    return stats.chi2.sf(np.maximum(lr, 0), 1)


@figure("vol-kupiec")
def _vol_kupiec(p):
    alpha = CONF[p.choice("conf", "99", list(CONF))]
    v, x = _vol(TICKER, alpha), _vol_dates(TICKER, alpha)
    ys = v["years"]
    k = p.num("years", len(C.TEST_YEARS), 1, len(C.TEST_YEARS), integer=True)
    k = min(k, len(ys))
    s = v["series"]
    m = np.floor(x) < ys[0] + k
    rn = np.asarray(s["next_return"])[m]
    a = v["var"]["alpha"]
    n = int(m.sum())
    rate = np.linspace(0, 4 * a, 161)
    pv = _kupiec_p(rate * n, n, a)
    pts, verdict = [], []
    for name, key, colr in (("normal", "var_normal", "warn"), ("filtered", "var_fhs", "s2")):
        kk = E.kupiec(rn < np.asarray(s[key])[m], a)
        pts.append({"x": 100 * kk["rate"], "y": kk["p"], "color": colr, "label": f"{name} {100 * kk['rate']:.2f}%"})
        verdict.append((name, kk))
    ok = rate[pv > 0.05] * 100
    nm, fh = verdict
    say = lambda kk: "passes" if kk["p"] >= 0.05 else ("fails: too many breaches" if kk["rate"] > a else "fails: too few breaches")
    return {
        "kind": "lines", "x": rate * 100, "xlabel": f"Breach rate over {n:,} test days, %", "ylabel": "Kupiec p-value", "ymin": 0, "ymax": 1,
        "series": [{"name": "p-value for that breach rate", "color": "accent", "y": pv, "area": True}],
        "hlines": [{"y": 0.05, "label": "5% significance", "color": "bad"}],
        "markers": [{"x": 100 * a, "label": f"target {100 * a:.0f}%"}],
        "points": pts,
        "notes": [f"With {k} test year{'s' if k > 1 else ''} ({n:,} days) a {100 * (1 - a):.0f}% VaR passes if its breach rate lies "
                  f"between {ok.min():.2f}% and {ok.max():.2f}%. More data narrows the band.",
                  f"Normal VaR: {nm[1]['exceed']} breaches, p = {nm[1]['p']:.2g}, {say(nm[1])}. "
                  f"Filtered historical VaR: {fh[1]['exceed']} breaches, p = {fh[1]['p']:.2g}, {say(fh[1])}."],
    }


# ===================================================================== volatility: text
_Q_PERS = 0.10 + 0.88
_Q_N, _Q_X = 1000, 25
_Q_P = float(_kupiec_p(_Q_X, _Q_N, 0.01))

explainer("markets/volatility", {
    "title": "Why tomorrow's risk is forecastable when tomorrow's return is not",
    "kicker": "Time Series Lab · Volatility",
    "summary": "Prices are close to a random walk, but the size of their moves is not: big days come in clusters. "
               "This page builds the five volatility models in the live view, from a 21-day average to GARCH and HAR with the VIX, "
               "and shows how they are scored out of sample and turned into a Value-at-Risk that can be checked against what happened.",
    "sections": [
        {"id": "what", "title": "Volatility: the size of the move, not its direction", "blocks": [
            {"type": "p", "text": "A stock's daily return r is the change in the log of its price from one close to the next. "
                                  "Its sign is close to a coin flip. Its <b>volatility</b> σ is the typical size of r, ignoring the sign: "
                                  "the standard deviation of returns. Volatility sets how wide the range of tomorrow's outcomes is, "
                                  "which is what risk limits, option prices and position sizes depend on."},
            {"type": "figure", "figure": "vol-cluster", "title": "Big days come in clusters",
             "caption": "Each thin spike is the largest move in a short block of days; the thicker line is a 21-day average. "
                        "Quiet years and stormy months alternate. Switch ticker: single stocks show the same pattern.",
             "controls": [TICKER_CTRL]},
            {"type": "p", "text": "This is <b>volatility clustering</b>: large moves tend to follow large moves, of either sign, and calm "
                                  "follows calm. News arrives in bursts, traders react to each other, and leverage and margin calls "
                                  "force selling after falls. Whatever the cause, it leaves a clear statistical fingerprint."},
            {"type": "figure", "figure": "vol-acf", "title": "Returns forget; their size remembers",
             "caption": "Autocorrelation: how strongly a series today is related to itself a number of days later. "
                        "Between the two horizontal lines is what pure noise would show.",
             "controls": [TICKER_CTRL]},
            {"type": "callout", "tone": "info", "title": "Measuring one day's variance",
             "text": "Squared returns are a noisy measure of a single day's variance. The live models use a better proxy, "
                     "<b>realised variance</b> (RV): the Garman–Klass range estimator, built from the day's open, high, low and close, "
                     "plus the squared overnight gap. The models forecast the <i>log</i> of tomorrow's RV, which is much closer to normal."},
        ]},
        {"id": "smooth", "title": "Averaging the past: 21-day history and EWMA", "blocks": [
            {"type": "p", "text": "The simplest forecast of tomorrow's variance is the average of the last 21 trading days (about a month). "
                                  "It reacts slowly to a shock, and a big day affects it for exactly 21 days, then falls out suddenly. "
                                  "The <b>exponentially weighted moving average</b> (EWMA) fixes both: every past day counts, with weights "
                                  "that shrink geometrically."},
            {"type": "eq", "text": "σ²ₜ₊₁ = λ σ²ₜ + (1 − λ) r²ₜ\n     = (1 − λ) [ r²ₜ + λ r²ₜ₋₁ + λ² r²ₜ₋₂ + … ]",
             "where": [["σ²ₜ₊₁", "variance forecast for tomorrow, made at today's close"],
                       ["r²ₜ", "today's squared return"],
                       ["λ", f"decay factor; RiskMetrics uses {LAMBDA}, as does the live model"]]},
            {"type": "figure", "figure": "vol-ewma", "title": "How much each past day counts",
             "caption": "Move λ. A smaller λ reacts faster and forgets faster; λ close to 1 approaches a long average.",
             "controls": [{"k": "lam", "label": "λ, decay factor", "min": 0.8, "max": 0.99, "step": 0.01, "value": LAMBDA}]},
            {"type": "p", "text": "Neither method has a long-run level: after a calm decade EWMA forecasts calm for ever, and after a crash it "
                                  "forecasts the crash's volatility until new data arrive. That is the gap GARCH fills."},
        ]},
        {"id": "garch", "title": "GARCH(1,1): shocks, memory and a long-run level", "blocks": [
            {"type": "p", "text": "GARCH (generalised autoregressive conditional heteroskedasticity) adds a constant to the EWMA recursion "
                                  "and lets the data choose the weights."},
            {"type": "eq", "text": "σ²ₜ₊₁ = ω + α ε²ₜ + β σ²ₜ        εₜ = rₜ − μ\nlong-run variance  σ̄² = ω / (1 − α − β)",
             "where": [["α", "reaction: how much of today's squared surprise enters tomorrow's variance"],
                       ["β", "memory: how much of today's variance forecast carries over"],
                       ["α + β", "persistence; below 1 means shocks fade and variance reverts to σ̄²"],
                       ["ω", "a floor that pulls variance back towards its long-run level"],
                       ["μ", "the mean daily return"]]},
            {"type": "p", "text": "Taking expectations, the extra variance left after h days is (α + β)<sup>h</sup> times what it was. "
                                  "The <b>half-life</b> of a shock is ln 0.5 / ln(α + β) days. EWMA is the special case ω = 0, "
                                  "α + β = 1: persistence exactly 1, so it never mean-reverts."},
            {"type": "figure", "figure": "vol-garch", "title": "What a four-standard-deviation day does to the forecast",
             "caption": "Expected annualised volatility after one big surprise. The dashed curve is the model fitted to the data; "
                        "the sliders set your own α and β with the same long-run level.",
             "controls": [{"k": "alpha", "label": "α, reaction", "min": 0.02, "max": 0.25, "step": 0.01, "value": 0.10},
                          {"k": "beta", "label": "β, memory", "min": 0.6, "max": 0.97, "step": 0.01, "value": 0.85}]},
            {"type": "steps", "title": "How the live model fits GARCH", "items": [
                f"Take the {GARCH_YEARS} years of daily returns before the test year.",
                "For trial values of μ, ω, α and β, run the recursion through those years and add up the normal log-likelihood "
                "of each day's return given its forecast variance.",
                "Search (Nelder–Mead) for the values with the highest likelihood, subject to ω > 0, α, β ≥ 0 and α + β < 1.",
                "Freeze them and run the recursion through the test year, one day at a time, to forecast each next day.",
            ]},
            {"type": "figure", "figure": "vol-garch-years", "title": "The fitted parameters, year by year",
             "caption": "One fit per test year, each on the years before it."},
            {"type": "quiz", "q": "A GARCH model has α = 0.10 and β = 0.88. Roughly how long until half of a shock's extra variance has gone?",
             "options": ["About 3 days", f"About {_half_life(_Q_PERS):.0f} days", "Never: the shock is permanent"], "answer": 1,
             "why": f"Persistence is α + β = {_Q_PERS:.2f}, so the half-life is ln 0.5 / ln {_Q_PERS:.2f} ≈ {_half_life(_Q_PERS):.0f} days. "
                    "It would be permanent only if α + β reached 1."},
        ]},
        {"id": "har", "title": "HAR: a day, a week and a month", "blocks": [
            {"type": "p", "text": "The heterogeneous autoregressive model (HAR, Corsi 2009) starts from an observation about who trades: "
                                  "some react to yesterday, some rebalance weekly, some think in months. So it forecasts tomorrow's log "
                                  "variance from three averages of the past, with weights found by ordinary least squares."},
            {"type": "eq", "text": "log RVₜ₊₁ = b₀ + b_d log RVₜ + b_w log RV⁽⁵⁾ₜ + b_m log RV⁽²²⁾ₜ\n"
                                   "HAR-X adds  + b_v log(VIX²ₜ / 252) + b_n · (down-moveₜ)²",
             "where": [["RVₜ", "today's realised variance"],
                       ["RV⁽⁵⁾ₜ, RV⁽²²⁾ₜ", "the mean of RV over the last 5 and 22 trading days"],
                       ["VIX", "the market's 30-day implied volatility from S&P 500 options, as a daily variance"],
                       ["down-move", "yesterday's return if negative, relative to the monthly RV: falls raise volatility more than rises"]]},
            {"type": "figure", "figure": "vol-har-flow", "title": "HAR-X as a weighted sum",
             "caption": "Arrow labels are the fitted weights; widths scale with their size. Each box shows today's input, "
                        "converted to annualised volatility."},
            {"type": "figure", "figure": "vol-har-parts", "title": "Three horizons and the VIX through one year",
             "caption": "Pick a year. The daily measure is noisy; the monthly one is smooth but late; the VIX looks forward.",
             "controls": [{"k": "year", "label": "Year", "min": C.YEAR_MIN, "max": C.YEAR_MAX, "step": 1, "value": 2020}]},
            {"type": "p", "text": "Three lagged terms with fixed windows approximate a slowly decaying memory like the one in the "
                                  "autocorrelation chart above, without any nonlinear fitting. The VIX brings in what option traders "
                                  "expect, which the past alone cannot know."},
        ]},
        {"id": "evaluate", "title": "Walk-forward evaluation and out-of-sample R²", "blocks": [
            {"type": "p", "text": "A forecast is only worth what it does on data it has not seen. The live view never scores a model on "
                                  "the years it was fitted on."},
            {"type": "steps", "title": "One walk-forward pass", "items": [
                f"For each test year from {START} on, fit every model only on days whose target falls before that year.",
                "Each day of the test year, forecast the next day's log RV with the frozen model and today's inputs.",
                "Correct the level-based forecasts (21-day, EWMA, GARCH) by their average in-sample error, since log of a mean is not a mean of logs.",
                "Move to the next year, refit, and repeat. Score every forecast together at the end.",
            ]},
            {"type": "eq", "text": "R²_oos = 1 − Σ (yₜ − ŷₜ)² / Σ (yₜ − ȳ)²",
             "where": [["yₜ", "log realised variance of the day forecast"], ["ŷₜ", "the forecast made the day before"],
                       ["ȳ", "the mean of yₜ over the test days"]]},
            {"type": "figure", "figure": "vol-r2", "title": "How much of tomorrow's variance each model explains",
             "caption": "Out-of-sample R² of log variance across all test years: the same numbers as the live view's default run "
                        "for each ticker.",
             "controls": [TICKER_CTRL]},
            {"type": "compare", "items": [
                {"title": "Level models", "points": ["21-day average, EWMA, GARCH", "Use squared returns only",
                                                     "One or a few parameters", "Robust, but slow to see regime changes"]},
                {"title": "Regression models", "points": ["HAR and HAR-X", "Use realised variance and the VIX",
                                                          "Weights fitted by least squares each year", "Several horizons at once"]},
            ]},
            {"type": "try", "label": "Run the walk-forward from 2008, through the financial crisis",
             "params": {"ticker": TICKER, "start_year": 2008, "m_hist21": True, "m_ewma": True, "m_garch": True, "m_har": True,
                        "m_har_x": True, "var_model": "har_x"}},
        ]},
        {"id": "var", "title": "Value at Risk and the Kupiec back-test", "blocks": [
            {"type": "p", "text": "A <b>one-day 99% Value at Risk</b> is the loss that tomorrow's return should fall below on only 1% of days. "
                                  "Given a volatility forecast it is one line."},
            {"type": "eq", "text": "VaRₜ₊₁ = q_α · σ̂ₜ₊₁\nbreach on day t+1  ⇔  rₜ₊₁ < VaRₜ₊₁",
             "where": [["σ̂ₜ₊₁", "the volatility forecast"],
                       ["q_α", "the α quantile of standardised returns: −2.33 for a normal at α = 1%, or the empirical quantile "
                               "of past returns divided by their own forecasts (filtered historical simulation)"]]},
            {"type": "figure", "figure": "vol-var", "title": "Thresholds and breaches",
             "caption": "Red dots are days that lost more than the normal VaR. Pick a test year.",
             "controls": [{"k": "year", "label": "Test year", "min": START, "max": C.YEAR_MAX, "step": 1, "value": 2020}, CONF_CTRL]},
            {"type": "p", "text": "If the model is right, each day is a breach with probability α, independently. "
                                  "<b>Kupiec's test</b> asks whether the observed breach count is plausible under that assumption:"},
            {"type": "eq", "text": "LR = −2 ln [ (1 − α)^(n−x) α^x ] + 2 ln [ (1 − x/n)^(n−x) (x/n)^x ]   ~ χ²₁",
             "where": [["n", "test days"], ["x", "breaches"], ["α", "target breach rate"],
                       ["p-value", "below 0.05: reject the VaR, too many or too few breaches"]]},
            {"type": "figure", "figure": "vol-kupiec", "title": "Which breach rates pass",
             "caption": "The curve is the p-value for every breach rate; the dots are the two VaR versions built on HAR-X. "
                        "Use fewer years to see how hard it is to reject a bad model on little data.",
             "controls": [{"k": "years", "label": "Test years used", "min": 1, "max": len(C.TEST_YEARS), "step": 1, "value": len(C.TEST_YEARS)},
                          CONF_CTRL]},
            {"type": "p", "text": "The normal quantile assumes thin tails. Daily returns have fat tails, so a normal VaR is breached "
                                  "more often than it promises. Filtered historical simulation takes the quantile from the model's own "
                                  "past standardised returns instead."},
            {"type": "quiz", "q": f"A 99% VaR is back-tested over {_Q_N:,} days and breached {_Q_X} times. What does Kupiec's test say?",
             "options": ["It passes: breaches are expected", "It fails: the model understates risk", "It fails: the model overstates risk"],
             "answer": 1 if _Q_P < 0.05 else 0,
             "why": f"The target is {_Q_N // 100} breaches. {_Q_X} gives p = {_Q_P:.2g}, below 0.05, and the rate is above the target: "
                    "losses past the threshold happen too often." if _Q_P < 0.05 else
                    f"p = {_Q_P:.2g} is above 0.05, so the count is consistent with the target."},
            {"type": "try", "label": "Build the VaR on GARCH at 95% instead", "params": {"ticker": TICKER, "var_model": "garch", "var_alpha": 0.05}},
        ]},
        {"id": "limits", "title": "What these models do not do", "blocks": [
            {"type": "callout", "tone": "warn", "title": "Limits, and not investment advice",
             "text": "Every model here forecasts one day ahead from its own past and the VIX. None foresees a crash before it starts: "
                     "they react within days, so a sudden turbulent spell can bring several breaches before the forecast catches up. Kupiec's test checks "
                     "only the number of breaches, not whether they bunch together. This is a teaching tool built on historical data; "
                     "it is not investment or risk-management advice."},
            {"type": "list", "items": [
                "One-day forecasts say little about risk over a month; that needs the model's longer-horizon path.",
                "Fat tails mean a normal VaR understates losses; filtered historical simulation helps but relies on the past.",
                "Six tickers and one index are a small sample of markets. Try NVDA in the live view for a much more volatile stock.",
            ]},
            {"type": "try", "label": "See the same models on Nvidia", "params": {"ticker": "NVDA"}},
        ]},
    ],
    "takeaways": [
        "Returns are close to unpredictable, but their size clusters and is strongly forecastable.",
        "EWMA weights past squared returns geometrically; GARCH adds a long-run level that variance reverts to.",
        "Persistence α + β sets how long a shock lasts: the half-life is ln 0.5 / ln(α + β) days.",
        "HAR combines daily, weekly and monthly realised variance; HAR-X adds the forward-looking VIX.",
        "A VaR is only as good as its back-test: Kupiec's test checks the breach count against the promised rate.",
    ],
    "glossary": [
        ["Volatility", "The standard deviation of returns: the typical size of a move."],
        ["Realised variance", "A measure of one day's variance from intraday prices (here Garman–Klass plus the overnight gap)."],
        ["EWMA", "Exponentially weighted moving average: recent days count more, with weights falling by λ per day."],
        ["GARCH(1,1)", "Variance tomorrow = constant + α × today's squared surprise + β × today's variance."],
        ["Persistence", "α + β in GARCH: the share of extra variance that survives each day."],
        ["HAR", "A regression of tomorrow's log variance on its daily, weekly and monthly averages."],
        ["Value at Risk", "A loss threshold that should be exceeded on a set share of days, such as 1%."],
        ["Kupiec test", "A likelihood-ratio test of whether a VaR's breach rate matches its target."],
    ],
}, order=1)


# ===================================================================== returns: data
@functools.lru_cache(maxsize=8)
def _ret_series(ticker: str = TICKER):
    d = C.panel(ticker, with_vix=False)
    r = d["ret"]
    return pd.DataFrame({"r": r, "next": r.shift(-1)}).dropna()


@functools.lru_cache(maxsize=8)
def _design(ticker: str = TICKER):
    return RET.design(ticker, START)


@functools.lru_cache(maxsize=8)
def _hist_mean(ticker: str = TICKER):
    """The live 'Historical mean' model, walk-forward (a mean per fold, so instant)."""
    P, cols, dates, tdates, ys, Xs = _design(ticker)
    y = P["y"].values
    has = ~np.isnan(y)
    f = np.full(len(P), np.nan)
    test = np.zeros(len(P), dtype=bool)
    for year in range(START, _end_year() + 1):
        tr, va, te = SM.fold_masks(dates, tdates, year)
        if te.sum():
            f[te] = np.mean(y[(tr | va) & has])
            test |= te
    test &= has
    return dates[test], y[test], f[test]


FEATURE_LABELS = {**{f"z_lag{k}": f"Return {k} day{'s' if k > 1 else ''} ago" for k in range(1, 11)},
                  "mom5": "Momentum, 1 week", "mom21": "Momentum, 1 month", "mom63": "Momentum, 3 months",
                  "mom252": "Momentum, 1 year", "vol5_63": "Vol ratio 5/63 days", "vol21_63": "Vol ratio 21/63 days",
                  "rv1_21": "RV today / 21-day RV", "ma50": "Price vs 50-day mean", "ma200": "Price vs 200-day mean",
                  "rsi14": "RSI, 14 days", "vlm63": "Volume vs 63-day mean", "log_vix": "VIX level", "vix_chg": "VIX change",
                  "vix_rel63": "VIX vs 63-day mean", "spy_z_lag1": "S&P 500 return today"}


# ===================================================================== returns: figures
@figure("ret-scatter")
def _ret_scatter(p):
    n = p.num("days", 1, 1, 60, integer=True)
    t = _ticker(p)
    s = _ret_series(t)
    past = s["r"].rolling(n).sum()
    ok = past.notna()
    x, y = past[ok].values * 100, s["next"][ok].values * 100
    slope, icpt = np.polyfit(x, y, 1)
    rho = float(np.corrcoef(x, y)[0, 1])
    k = C.stride(len(x), 1500)
    xs = np.array([x.min(), x.max()])
    return {
        "kind": "scatter", "xlabel": "Today's return, %" if n == 1 else f"Return over the last {n} days, %", "ylabel": "Next day's return, %",
        "points": [{"x": float(a), "y": float(b), "color": "s1", "r": 2} for a, b in zip(x[::k], y[::k])],
        "lines": [{"x": xs, "y": icpt + slope * xs, "color": "accent", "name": "Least-squares line"}],
        "notes": [f"Across {len(x):,} days of {t}, the correlation is {rho:+.3f}: "
                  + ("today's return" if n == 1 else f"the past {n}-day return") + " explains "
                  f"{100 * rho ** 2:.2f}% of the variance of the next day's return, in sample and with hindsight.",
                  f"The fitted slope is {slope:+.3f}: a 1% move " + ("today" if n == 1 else "over the window") + f" shifts the expected next day by {100 * slope:+.1f} basis points, "
                  f"against a typical daily move of {np.std(y):.1f}%. (One dot in {k} is drawn.)"],
    }


@figure("ret-hit")
def _ret_hit(p):
    r2 = p.num("r2", 0.5, 0, 3) / 100
    grid = np.linspace(0, 0.03, 121)
    hit = 50 + 100 * np.arcsin(np.sqrt(grid)) / np.pi
    h = 0.5 + math.asin(math.sqrt(r2)) / math.pi
    _, y, _ = _hist_mean()
    up = float(np.mean(y[y != 0] > 0))
    need = (stats.norm.ppf(0.95) * 0.5 / (h - 0.5)) ** 2 if h > 0.5 else math.inf
    r2_up = math.sin(math.pi * (up - 0.5)) ** 2
    return {
        "kind": "lines", "x": grid * 100, "xlabel": "R² of the forecast, %", "ylabel": "Share of days with the right sign, %",
        "series": [{"name": "Hit rate if returns are normal", "color": "s2", "y": hit, "area": True}],
        "hlines": [{"y": 100 * up, "label": f"always 'up' ({TICKER}, test years)", "color": "neutral"}],
        "points": [{"x": 100 * r2, "y": 100 * h, "color": "accent", "label": f"{100 * h:.1f}%"}],
        "notes": [(f"A forecast with R² = {100 * r2:.2f}% gets the sign right on {100 * h:.1f}% of days. Telling that apart from a coin "
                   f"at the 5% level takes about {need:,.0f} trading days ({need / DAYS:.1f} years).") if r2 > 0 else
                  "With R² = 0 the forecast is a coin flip: 50% of signs right.",
                  f"Saying 'up' every day was right {100 * up:.1f}% of the time on {TICKER}'s test days; a symmetric forecast would need "
                  f"R² ≈ {100 * r2_up:.1f}% to match that."],
    }


@figure("ret-features")
def _ret_features(p):
    P, cols, dates, tdates, ys, Xs = _design()
    y = P["y"].values / P["vol63"].values
    pre = np.asarray(tdates < pd.Timestamp(f"{START}-01-01")) & ~np.isnan(y)
    post = ~pre & ~np.isnan(y)
    cols = [c for c in cols if not (TICKER == "SPY" and c == "spy_z_lag1")]   # a copy of z_lag1 for SPY
    X = P[cols].values
    cp = np.array([np.corrcoef(X[pre, j], y[pre])[0, 1] for j in range(len(cols))])
    cq = np.array([np.corrcoef(X[post, j], y[post])[0, 1] for j in range(len(cols))])
    order = np.argsort(-np.abs(cp))[:12]
    flips = int(np.sum(np.sign(cp) != np.sign(cq)))
    j = order[0]
    return {
        "kind": "bars", "labels": [FEATURE_LABELS.get(cols[i], cols[i]) for i in order], "ylabel": "Correlation with next day's return, %",
        "series": [{"name": f"Before {START} (training)", "color": "s1", "values": 100 * cp[order]},
                   {"name": f"{START} on (test)", "color": "s3", "values": 100 * cq[order]}],
        "height": 320,
        "notes": [f"The strongest feature before {START}, ‘{FEATURE_LABELS.get(cols[j], cols[j])}’, had a correlation of "
                  f"{100 * cp[j]:+.1f}%; from {START} on it was {100 * cq[j]:+.1f}%.",
                  f"{flips} of the {len(cols)} features changed sign between the two periods, and the largest correlation in either "
                  f"period is {100 * max(np.max(np.abs(cp)), np.max(np.abs(cq))):.1f}% in size."],
    }


def _ridge_path(Xtr, ytr, alphas):
    """Coefficients of ridge regression (intercept unpenalised) for each alpha, by SVD."""
    xm, ym = Xtr.mean(0), ytr.mean()
    with np.errstate(all="ignore"):           # numpy + Accelerate emits spurious FP warnings from matmul on macOS
        U, s, Vt = np.linalg.svd(Xtr - xm, full_matrices=False)
        uy = U.T @ (ytr - ym)
        B = np.array([Vt.T @ (s / (s ** 2 + a) * uy) for a in alphas])
    return B, xm, ym


@figure("ret-ridge")
def _ret_ridge(p):
    year = p.num("year", START, START, C.YEAR_MAX, integer=True)
    P, cols, dates, tdates, ys, Xs = _design()
    has = ~np.isnan(P["y"].values)
    tr, va, te = SM.fold_masks(dates, tdates, year)
    tr, va = tr & has, va & has
    Xs = Xs.astype(float)
    mu, sd = Xs[tr].mean(0), Xs[tr].std(0)                    # stocklab.fit_ridge re-standardises on the training rows
    sd[sd == 0] = 1
    Z = (Xs - mu) / sd
    alphas = np.logspace(0, 5, 41)
    B, xm, ym = _ridge_path(Z[tr], ys[tr], alphas)
    grid = np.array(RIDGE_ALPHAS, dtype=float)
    Bg, _, _ = _ridge_path(Z[tr], ys[tr], grid)
    with np.errstate(all="ignore"):
        val = [np.mean((ym + (Z[va] - xm) @ b - ys[va]) ** 2) for b in Bg]
    pick = float(grid[int(np.argmin(val))])
    zero = float(np.mean(ys[va] ** 2))
    top = np.argsort(-np.abs(B[0]))[:5]
    colours = ["s1", "s2", "s3", "s4", "accent"]
    shrink = abs(B[np.argmin(np.abs(alphas - pick)), top[0]]) / abs(B[0, top[0]])
    return {
        "kind": "lines", "x": alphas, "xlog": True, "xlabel": "Penalty α", "ylabel": "Coefficient (per standard deviation)",
        "series": [{"name": FEATURE_LABELS.get(cols[j], cols[j]), "color": c, "y": B[:, j]} for j, c in zip(top, colours)],
        "markers": [{"x": pick, "label": f"chosen α = {pick:g}"}],
        "notes": [f"For test year {year}, the validation years {year - SM.VAL_YEARS}–{year - 1} pick α = {pick:g} from "
                  f"{', '.join(f'{a:g}' for a in RIDGE_ALPHAS)}" + (", the strongest penalty on offer" if pick == grid.max() else "")
                  + f"; that shrinks the largest coefficient to {100 * shrink:.0f}% of its unpenalised size.",
                  f"On the validation years the chosen ridge's squared error is {100 * (1 - min(val) / zero):.2f}% "
                  + ("below" if min(val) < zero else "above") + " that of forecasting zero."],
    }


@figure("ret-folds")
def _ret_folds(p):
    P, cols, dates, tdates, ys, Xs = _design()
    first = _years(dates[:1])[0]
    end = _end_year()
    paths = []
    for i, year in enumerate(range(START, end + 1)):
        v0 = year - SM.VAL_YEARS
        tag = i == 0
        paths += [{"points": [[first, year], [v0, year]], "color": "s1", "width": 7, "name": "Training" if tag else None},
                  {"points": [[v0, year], [year, year]], "color": "s3", "width": 7, "name": "Validation (choose settings)" if tag else None},
                  {"points": [[year, year], [year + 1, year]], "color": "accent", "width": 7, "name": "Test (scored)" if tag else None}]
    for q in paths:
        if q["name"] is None:
            del q["name"]
    n = end - START + 1
    return {
        "kind": "paths", "paths": paths, "equal": False, "xlabel": "Data used (year)", "ylabel": "Test year", "height": 300,
        "notes": [f"{n} refits for {TICKER}: each test year gets models fitted on everything before it, from {first:.0f} on.",
                  f"The last {SM.VAL_YEARS} years before each test year choose the settings (ridge penalty, tree size, "
                  "number of trees), then the model is refit on training and validation together."],
    }


@figure("ret-costs")
def _ret_costs(p):
    cost = p.num("cost", COST, 0, 20)
    t = _ticker(p)
    s = _ret_series(t)
    s = s[(s.index.year >= START) & (s.index.year <= _end_year())]
    dates = pd.DatetimeIndex(s.index)
    pnl, pos = E.backtest(s["next"].values, s["r"].values, dates, cost_bps=cost)
    gross, _ = E.backtest(s["next"].values, s["r"].values, dates, cost_bps=0)
    bh = pd.Series(np.expm1(s["next"].values), index=dates)
    pr, pg, pb = E.perf(pnl), E.perf(gross), E.perf(bh)
    k = 5
    x = _years(dates)[::k]
    trades = float(np.sum(np.abs(np.diff(np.r_[0.0, pos])))) / (len(pos) / DAYS)
    return {
        "kind": "lines", "x": x, "xdate": True, "xlabel": "Year", "ylabel": "Growth of $1 (log scale)", "ylog": True,
        "series": [{"name": "Buy and hold", "color": "neutral", "y": (1 + bh).cumprod().values[::k]},
                   {"name": "Rule before costs", "color": "s1", "y": (1 + gross).cumprod().values[::k], "dash": True},
                   {"name": f"Rule after {cost:g} bp per trade", "color": "s2", "y": (1 + pnl).cumprod().values[::k]}],
        "notes": [f"The illustrative rule (hold {t} the day after an up day, cash after a down day) trades {trades:.0f} times a year. "
                  + (f"Costs cut its yearly growth from {100 * pg['cagr']:.1f}% to {100 * pr['cagr']:.1f}%." if cost > 0 else
                   f"Without costs it grows {100 * pg['cagr']:.1f}% a year."),
                  f"Sharpe ratio {'after costs ' if cost > 0 else ''}{pr['sharpe']:.2f} against {pb['sharpe']:.2f} for buy and hold "
                  f"(which grew {100 * pb['cagr']:.1f}% a year, with a worst fall of {-100 * pb['max_dd']:.0f}%)."],
    }


@figure("ret-dm")
def _ret_dm(p):
    t = _ticker(p)
    dates, y, f = _hist_mean(t)
    d = (y ** 2 - (y - f) ** 2) * 1e4
    dm, dmp = E.diebold_mariano(y, np.zeros_like(y), f)
    r2 = E.r2_oos(y, f)
    ad = np.sort(np.abs(d))[::-1]
    top = float(ad[: max(1, len(ad) // 20)].sum() / ad.sum())
    k = 3
    return {
        "kind": "lines", "x": _years(dates)[::k], "xdate": True, "xlabel": "Year", "ylabel": "Cumulative squared-error gain, %²",
        "hlines": [{"y": 0, "color": "neutral"}],
        "series": [{"name": "Zero's loss − historical mean's loss, summed", "color": "s1", "y": np.cumsum(d)[::k], "area": True}],
        "notes": [f"Over {len(y):,} test days of {t} the historical mean's R² against zero is {100 * r2:+.2f}%. The line rises on days the mean "
                  f"wins and falls on days it loses; the busiest 5% of days account for {100 * top:.0f}% of all the movement.",
                  f"Diebold–Mariano: statistic {dm:+.2f}, p = {dmp:.2f}. "
                  + ("The difference is not significant: the two forecasts cannot be told apart." if dmp >= 0.05 else
                     "The difference is significant at 5%.")],
    }


# ===================================================================== returns: text
explainer("markets/returns", {
    "title": "Forecasting tomorrow's return, and why it is so hard",
    "kicker": "Time Series Lab · Returns",
    "summary": "Daily returns are mostly noise: whatever is predictable is small and quickly traded away. "
               "This page explains the features and models in the live view, from ridge regression to gradient boosting and ARIMA, "
               "the walk-forward protocol that keeps the test honest, and the three questions every forecast must answer: "
               "is it more accurate than zero, does it make money after costs, and is the difference more than luck?",
    "sections": [
        {"id": "noise", "title": "Why returns are nearly unpredictable", "blocks": [
            {"type": "p", "text": "If tomorrow's return were easy to forecast, traders would buy before the rise and sell before the fall, "
                                  "and their trades would move today's price until the forecast edge disappeared. This is the "
                                  "<b>efficient-market</b> argument: public information is already in the price, so the next move is "
                                  "mainly a reaction to news nobody has yet."},
            {"type": "figure", "figure": "ret-scatter", "title": "Does the recent past predict tomorrow?",
             "caption": "Each dot is one day. Move the window: yesterday alone, a week, a month. The cloud stays round.",
             "controls": [{"k": "days", "label": "Look-back window", "min": 1, "max": 60, "step": 1, "value": 1, "unit": "days"}, TICKER_CTRL]},
            {"type": "p", "text": "The efficient-market view is not exact. Risk premia make the average return slightly positive, "
                                  "and small, unstable patterns appear and fade. The question is never \"is there any signal?\" "
                                  "but \"is there enough to survive noise, estimation error and trading costs?\""},
            {"type": "figure", "figure": "ret-hit", "title": "What a small R² buys",
             "caption": "If forecast and return are jointly normal with correlation ρ, the sign is right with probability "
                        "½ + arcsin(ρ)/π, and R² = ρ². Move the R²; the dot shows the hit rate it implies.",
             "controls": [{"k": "r2", "label": "R² of the forecast", "min": 0, "max": 3, "step": 0.05, "value": 0.5, "unit": "%"}]},
        ]},
        {"id": "features", "title": "The features: what the models are allowed to see", "blocks": [
            {"type": "p", "text": "Every feature on a row dated t uses only prices known at the close of day t, and the target is the "
                                  "next day's log return. All features are <b>scale-free</b>: returns are divided by the last 63 days' "
                                  "volatility, so a calm year and a wild year look alike to the model."},
            {"type": "cards", "items": [
                {"title": "Lagged returns", "text": "Each of the last ten daily returns, in units of recent volatility."},
                {"title": "Momentum and trend", "text": "Summed returns over a week, a month, a quarter and a year; price against its 50- and 200-day means; RSI."},
                {"title": "Volatility state", "text": "Short against long volatility, today's realised variance against its monthly mean, and volume."},
                {"title": "The VIX", "text": "Its level, its daily change and its level against the last quarter. Single stocks also see the S&P 500's day."},
            ]},
            {"type": "figure", "figure": "ret-features", "title": "Each feature's correlation with the next day, before and after",
             "caption": "The twelve features most correlated with the next day's (volatility-scaled) return before the test years, "
                        "and the same correlation in the test years."},
            {"type": "callout", "tone": "info", "title": "Why the target is divided by volatility",
             "text": "Ridge and boosting learn yₜ₊₁ / σ⁽⁶³⁾ₜ, clipped at ±5, and the forecast is multiplied back by σ⁽⁶³⁾ₜ. "
                     "Otherwise the few most volatile weeks would dominate the squared-error loss."},
        ]},
        {"id": "models", "title": "Three models: ridge, gradient boosting and ARIMA", "blocks": [
            {"type": "p", "text": "<b>Ridge regression</b> is ordinary least squares with a penalty on the size of the coefficients. "
                                  "With weak signals and correlated features, plain least squares fits the noise; the penalty pulls every "
                                  "coefficient towards zero and trades a little bias for much less variance."},
            {"type": "eq", "text": "β̂ = argmin  Σ (yₜ − xₜ·β)² + α ‖β‖²",
             "where": [["xₜ", "the standardised features on day t"], ["yₜ", "the volatility-scaled next-day return"],
                       ["α", f"the penalty, chosen on the validation years from {', '.join(f'{a:g}' for a in RIDGE_ALPHAS)}"]]},
            {"type": "figure", "figure": "ret-ridge", "title": "Coefficients shrink as the penalty grows",
             "caption": "The five largest coefficients at a small penalty, traced as α rises. Pick the test year whose training data are used.",
             "controls": [{"k": "year", "label": "Test year", "min": START, "max": C.YEAR_MAX, "step": 1, "value": START}]},
            {"type": "compare", "items": [
                {"title": "Gradient boosting", "points": ["Adds hundreds of small decision trees, each fitting what the others missed",
                                                          "Captures thresholds and interactions (for example: momentum only when the VIX is low)",
                                                          "Tree size, learning rate and number of trees chosen on validation",
                                                          "At least 400 rows per leaf, to stop it chasing single days"]},
                {"title": "ARIMA(p, 0, q)", "points": ["A classic time-series model of returns on their own past values and past errors",
                                                       "Order chosen once by AIC on data before the first test year",
                                                       "Uses no other features",
                                                       "Off by default in the live view because it is slower"]},
            ]},
            {"type": "eq", "text": "ARIMA(p, 0, q):  rₜ = c + φ₁ rₜ₋₁ + … + φ_p rₜ₋ₚ + εₜ + θ₁ εₜ₋₁ + … + θ_q εₜ₋_q",
             "where": [["φ", "autoregressive weights on past returns"], ["θ", "moving-average weights on past forecast errors"],
                       ["εₜ", "the part of today's return the model did not expect"]]},
            {"type": "p", "text": "The benchmark is <b>zero</b>: forecast no change every day. A second baseline, the <b>historical mean</b>, "
                                  "forecasts the average past return, which is almost always slightly positive."},
        ]},
        {"id": "walk", "title": "Walk-forward refits", "blocks": [
            {"type": "p", "text": "A model tested on data it was fitted on, or tuned on, will look better than it is. The live view refits "
                                  "every model once per test year, and a row is never used for fitting if its target falls in or after "
                                  "the test year."},
            {"type": "figure", "figure": "ret-folds", "title": "Which data each refit sees",
             "caption": "One row per test year. Fitting uses only the past; the scored year is always new."},
            {"type": "steps", "title": "One refit", "items": [
                f"Split the past into training data and the last {SM.VAL_YEARS} years (validation).",
                "Fit each candidate setting on the training data and score it on validation.",
                "Refit the best setting on training and validation together.",
                "Forecast every day of the test year, then move on a year.",
            ]},
            {"type": "callout", "tone": "tip", "title": "Leaks are easy to miss",
             "text": "Standardising features with the mean and spread of the whole sample, choosing features after looking at the test years, "
                     "or picking the best of many models on test results all leak the future. The live code fits its scaler only "
                     f"on data before {START - SM.VAL_YEARS}, before the first validation window."},
        ]},
        {"id": "scores", "title": "Scoring: accuracy, direction and money after costs", "blocks": [
            {"type": "eq", "text": "R²_oos = 1 − Σ (yₜ − ŷₜ)² / Σ yₜ²\nhit rate = share of days with sign(ŷₜ) = sign(yₜ)\n"
                                   "Sharpe = mean daily P&L / std of daily P&L × √252",
             "where": [["R²_oos", "out-of-sample R² against forecasting zero (Campbell–Thompson); negative means worse than zero"],
                       ["hit rate", "compared with the share of up days, since 'always up' is the real bar"],
                       ["P&L", "long when the forecast is positive, in cash otherwise, minus a cost for each switch"]]},
            {"type": "figure", "figure": "ret-costs", "title": "A simple timing rule, before and after costs",
             "caption": "Not one of the live models: a one-line rule to show what trading costs do. Move the cost per trade.",
             "controls": [{"k": "cost", "label": "Cost per trade", "min": 0, "max": 20, "step": 1, "value": COST, "unit": "bp"}, TICKER_CTRL]},
            {"type": "p", "text": "A timing strategy is out of the market part of the time, so it misses part of the market's average gain "
                                  "and must make that up by avoiding the bad days. Each switch costs money, and the costs add up quickly "
                                  "for a forecast that changes sign often."},
            {"type": "quiz", "q": "A model gets the direction right on 53% of days. Is it skilful?",
             "options": ["Yes: anything above 50% beats a coin", "Not necessarily: compare it with always saying 'up'",
                         "No: hit rates are meaningless"], "answer": 1,
             "why": "If more than half of days are up days (the horizontal line in the hit-rate chart shows the share for SPY), "
                    "saying 'up' every day already scores above 50%. The live view tests each hit rate against that share, not against 50%."},
        ]},
        {"id": "dm", "title": "More than luck? The Diebold–Mariano test", "blocks": [
            {"type": "p", "text": "Two forecasts will never have exactly the same error. The <b>Diebold–Mariano</b> test asks whether the "
                                  "difference in their average loss is larger than its own noise. It works on the daily loss "
                                  "differential and allows for its autocorrelation."},
            {"type": "eq", "text": "dₜ = (yₜ − 0)² − (yₜ − ŷₜ)²\nDM = d̄ / √( V̂(d̄) )   ≈ N(0, 1)",
             "where": [["dₜ", "how much better the model did than zero on day t"],
                       ["d̄", "its mean over the test days"],
                       ["V̂(d̄)", "Newey–West variance of the mean, with 5 lags"],
                       ["DM > 0", "the model beat zero; |DM| > 1.96 means significant at 5%"]]},
            {"type": "figure", "figure": "ret-dm", "title": "Historical mean against zero, day by day",
             "caption": "The live view's 'Historical mean' model, recomputed here walk-forward. The test is whether the final gain is distinguishable from zero.",
             "controls": [TICKER_CTRL]},
            {"type": "quiz", "q": "A model's out-of-sample R² against zero is −0.3%. What does that mean?",
             "options": ["It explains 0.3% of returns", "Its squared errors are larger than those of forecasting zero",
                         "It is significantly worse than zero"], "answer": 1,
             "why": "R²_oos below zero means Σ(y − ŷ)² > Σ y². Whether the gap is significant is the Diebold–Mariano test's job."},
            {"type": "try", "label": "Run every model on the S&P 500, including ARIMA",
             "params": {"ticker": TICKER, "start_year": START, "m_hist_mean": True, "m_ridge": True, "m_gbm": True, "m_arima": True},
             "note": "The server refits every model for every test year, so it takes several seconds."},
            {"type": "try", "label": "Try the same on Nvidia", "params": {"ticker": "NVDA", "start_year": START}},
        ]},
        {"id": "limits", "title": "What a null result means", "blocks": [
            {"type": "callout", "tone": "warn", "title": "Limits, and not investment advice",
             "text": "These are daily forecasts on six US tickers with public data and simple costs. A null result here does not prove "
                     "that no strategy works, and a good back-test would not prove that one does: markets change, and testing many "
                     "ideas on the same history finds some that worked by chance. Nothing on this page is investment advice."},
            {"type": "list", "items": [
                "A small R² can still matter at scale, at other horizons or with other data; at a daily horizon, costs take a large bite (see the cost chart).",
                "The test years are one path of history; a different decade could rank the models differently.",
                "Volatility, unlike returns, is forecastable: see the volatility page.",
            ]},
            {"type": "try", "label": "Only the last five years", "params": {"ticker": TICKER, "start_year": 2022}},
        ]},
    ],
    "takeaways": [
        "Competition between traders removes most predictable patterns, so daily returns are mostly noise.",
        f"Even a real edge is small: an R² of 0.5% would lift the hit rate by only {100 * math.asin(math.sqrt(0.005)) / math.pi:.1f} points above 50%.",
        "Walk-forward refits keep the future out of fitting, tuning and scaling.",
        "Score three ways: R² against zero, hit rate against always 'up', and Sharpe ratio after costs against buy and hold.",
        "Diebold–Mariano tells you whether a better score is more than luck; a better score alone is not evidence.",
    ],
    "glossary": [
        ["Efficient market", "One where prices already reflect available information, so future moves are mostly unpredictable."],
        ["Out-of-sample R²", "1 − squared errors of the model / squared errors of forecasting zero, on data not used for fitting."],
        ["Ridge regression", "Least squares with a penalty on the size of the coefficients."],
        ["Gradient boosting", "A sum of many small decision trees, each fitted to the errors of the previous ones."],
        ["ARIMA", "A model of a series on its own past values and past forecast errors."],
        ["Walk-forward", "Refit on the past, forecast the next period, move forward, repeat."],
        ["Sharpe ratio", "Average return divided by its volatility, annualised: return per unit of risk."],
        ["Diebold–Mariano", "A test of whether two forecasts differ in accuracy by more than chance."],
    ],
}, order=2)


# ===================================================================== neural: data
@functools.lru_cache(maxsize=64)
def _trap(ticker: str = TICKER, share: float = NN.TRAP_TRAIN_SHARE):
    price = C.prices(ticker)["AdjClose"]
    vals = price.values.astype(float)
    split = int(len(vals) * share)
    return price.index, vals, split


def _lstm_weights(k: int, h: int) -> int:
    return 4 * h * (k + h) + 8 * h          # PyTorch LSTM: four gates, two bias vectors


def _gru_weights(k: int, h: int) -> int:
    return 3 * h * (k + h) + 6 * h


# ===================================================================== neural: figures
@figure("nn-memory")
def _nn_memory(p):
    f = p.num("forget", 0.9, 0.5, 0.99)
    t = np.arange(0, 61)
    win = NN.RNN_WINDOW
    return {
        "kind": "lines", "x": t, "xlabel": "Steps after the input was written", "ylabel": "Share still in the cell state", "ymin": 0, "ymax": 1,
        "series": [{"name": f"Forget gate f = {f:.2f}", "color": "s2", "y": f ** t, "area": True}],
        "hlines": [{"y": 1, "label": "f = 1: nothing forgotten", "color": "neutral"}],
        "markers": [{"x": win, "label": f"live window: {win} days"}],
        "notes": [f"With f = {f:.2f} the cell keeps half of a memory for {_half_life(f):.1f} steps.",
                  f"After the {win} days the live networks see, {100 * f ** win:.1f}% of the first day's input is left."],
    }


@figure("nn-lstm")
def _nn_lstm(p):
    P, cols, dates, tdates, ys, Xs = _design()
    k, h = len(cols), RNN_HIDDEN
    per = h * (k + h) + 2 * h
    gates = [("forget", "Forget gate f"), ("input", "Input gate i"), ("cand", "Candidate c̃"), ("output", "Output gate o")]
    nodes = [{"id": "in", "x": 10, "y": 95, "w": 120, "h": 64, "title": "Input", "value": f"{k} features", "sub": "xₜ and hₜ₋₁"}]
    nodes += [{"id": g, "x": 175, "y": 6 + 58 * i, "w": 140, "h": 46, "title": t, "value": f"{per:,} weights",
               "fill_color": ["s1", "s2", "s3", "s4"][i]} for i, (g, t) in enumerate(gates)]
    nodes += [{"id": "cell", "x": 360, "y": 40, "w": 115, "h": 70, "title": "Cell cₜ", "sub": "f·cₜ₋₁ + i·c̃", "hot": True},
              {"id": "hid", "x": 360, "y": 165, "w": 115, "h": 62, "title": "Hidden hₜ", "sub": "o·tanh(cₜ)"},
              {"id": "out", "x": 500, "y": 165, "w": 90, "h": 62, "title": "Forecast", "value": f"{h + 1} weights"}]
    edges = [{"from": "in", "to": g, "width": 2} for g, _ in gates]
    edges += [{"from": "forget", "to": "cell", "label": "keep", "color": "s1", "width": 3, "flow": True},
              {"from": "input", "to": "cell", "label": "write", "color": "s2", "width": 3, "flow": True},
              {"from": "cand", "to": "cell", "color": "s3", "width": 2.5},
              {"from": "output", "to": "hid", "label": "read", "color": "s4", "width": 3},
              {"from": "cell", "to": "hid", "width": 3, "flow": True},
              {"from": "hid", "to": "out", "width": 3, "flow": True}]
    total = _lstm_weights(k, h) + h + 1
    return {
        "kind": "flow", "height": 250, "nodes": nodes, "edges": edges,
        "notes": [f"The live LSTM has {h} hidden units on {k} features: 4 × {per:,} gate weights plus a {h + 1}-weight output, "
                  f"{total:,} in all. A GRU of the same size has {_gru_weights(k, h) + h + 1:,}.",
                  f"The tutorial LSTM in the trap has {TRAP_HIDDEN} units on one input (the price): {_lstm_weights(1, TRAP_HIDDEN) + TRAP_HIDDEN + 1:,} weights."],
    }


@figure("nn-scaler")
def _nn_scaler(p):
    share = p.num("share", NN.TRAP_TRAIN_SHARE, 0.5, 0.9)
    tk = _ticker(p)
    idx, v, split = _trap(tk, round(share, 4))
    lo_a, hi_a = v.min(), v.max()
    lo_c, hi_c = v[:split].min(), v[:split].max()
    leaky, clean = (v - lo_a) / (hi_a - lo_a), (v - lo_c) / (hi_c - lo_c)
    k = C.stride(len(v), NN.TRAP_MAX_POINTS)
    x = _years(idx)[::k]
    above = float(np.mean(v[split:] > hi_c))
    sx = _years(idx[split:split + 1])[0]
    return {
        "kind": "lines", "x": x, "xdate": True, "xlabel": "Year", "ylabel": f"Scaled {tk} price",
        "series": [{"name": "Scaler fitted on all prices (leaky)", "color": "bad", "y": leaky[::k]},
                   {"name": "Scaler fitted on training prices only", "color": "s2", "y": clean[::k]}],
        "hlines": [{"y": 1, "label": "highest training price", "color": "neutral"}],
        "bands": [{"x0": float(sx), "x1": float(x[-1]), "label": "test"}],
        "notes": [f"For {tk}, with the first {100 * share:.0f}% as training data, the test prices reach {v[split:].max() / hi_c:.1f}× the training maximum; "
                  f"{100 * above:.0f}% of test days are above anything seen in training.",
                  "The leaky scaler squeezes the whole test range into the same 0–1 box as training, so the network never meets "
                  f"an unfamiliar input. With the clean scaler, test inputs reach {clean[split:].max():.1f} on a scale where training ended at 1."],
    }


@functools.lru_cache(maxsize=1)
def _trap_test():
    idx, v, split = _trap()
    t = np.arange(split, len(v))
    return idx, v, split, t, v[t], v[t - 1]


@figure("nn-persistence")
def _nn_persistence(p):
    idx, v, split, t, actual, naive = _trap_test()
    m = slice(-120, None)
    return {
        "kind": "lines", "x": _years(idx[t])[m], "xdate": True, "xlabel": "Date", "ylabel": f"{TICKER} price, $",
        "series": [{"name": "Actual price", "color": "s1", "y": actual[m]},
                   {"name": "Forecast: tomorrow = today", "color": "warn", "y": naive[m], "dash": True}],
        "notes": [f"On all {len(t):,} test days, 'tomorrow = today' has R² = {_r2(actual, naive):.4f} on the price and is off by "
                  f"{100 * np.mean(np.abs(actual - naive) / actual):.2f}% on average.",
                  "From a distance the two lines are one; zoomed in, the forecast is the actual price shifted right by a day."],
    }


@figure("nn-echo")
def _nn_echo(p):
    span = p.num("span", 5, 1, 30, integer=True)
    idx, v, split, t, actual, naive = _trap_test()
    smooth = pd.Series(v).ewm(span=span, adjust=False).mean().values[t - 1]
    da = np.diff(actual)

    def xc(pred):
        dp = np.diff(pred)
        return [float(np.corrcoef(dp[lag:], da[:len(da) - lag])[0, 1]) for lag in range(6)]

    xn, xs = xc(naive), xc(smooth)
    peak = int(np.argmax(xs))
    return {
        "kind": "bars", "labels": [f"lag {k}" for k in range(6)], "ylabel": "Correlation of forecast change with actual change",
        "series": [{"name": "Tomorrow = today", "color": "warn", "values": xn},
                   {"name": f"Smoothed recent prices (span {span})", "color": "s2", "values": xs}],
        "notes": [f"The smoothed forecast's daily changes correlate most with the actual change {peak} day{'s' if peak != 1 else ''} earlier "
                  f"(ρ = {xs[peak]:.2f}); with the change on the day it forecasts, only {xs[0]:+.2f}.",
                  "A forecast that only follows past prices echoes them: its peak is at lag 1, not lag 0. "
                  "The live trap job draws the same chart for the trained LSTM."],
    }


@figure("nn-level-change")
def _nn_level_change(p):
    idx, v, split, t, actual, naive = _trap_test()
    drift = float(np.mean(np.diff(np.log(v[:split]))))
    fc = {"Tomorrow = today": naive,
          "Today + training drift": naive * math.exp(drift),
          "10-day smoothed price": pd.Series(v).ewm(span=10, adjust=False).mean().values[t - 1]}
    lvl = [_r2(actual, f) for f in fc.values()]
    chg = [_r2(actual - naive, f - naive) for f in fc.values()]
    return {
        "kind": "bars", "labels": list(fc), "ylabel": "R² on the test days",
        "series": [{"name": "R² on the price level", "color": "s1", "values": lvl},
                   {"name": "R² on the daily change", "color": "bad", "values": chg}],
        "notes": [f"All three forecasts score R² between {min(lvl):.3f} and {max(lvl):.3f} on the price level. On the daily change, "
                  f"which is what a trader needs, the best manages {max(chg):+.4f}.",
                  "R² on prices rewards knowing where the price is today. It says nothing about where it goes next."],
    }


@figure("nn-honest")
def _nn_honest(p):
    P, cols, dates, tdates, ys, Xs = _design()
    has = ~np.isnan(P["y"].values)
    tr, va, te = SM.fold_masks(dates, tdates, START)
    n_tr = int(np.sum(tr & has)) - (NN.RNN_WINDOW - 1)
    return {
        "kind": "stats",
        "items": [{"label": "Target", "value": "next-day return", "note": "divided by 63-day volatility, not the price"},
                  {"label": "Input window", "value": f"{NN.RNN_WINDOW} days", "note": f"× {len(cols)} scale-free features"},
                  {"label": "Network", "value": f"{RNN_HIDDEN} units", "note": f"{_lstm_weights(len(cols), RNN_HIDDEN) + RNN_HIDDEN + 1:,} LSTM weights, dropout 0.2"},
                  {"label": f"Training sequences for {START}", "value": f"{n_tr:,}", "note": f"{TICKER}, every day before {START - SM.VAL_YEARS}"},
                  {"label": "Early stopping", "value": f"{SM.VAL_YEARS} validation years", "note": f"stops after {RNN_PATIENCE} epochs without improvement"},
                  {"label": "Benchmark", "value": "forecast zero", "note": "plus buy and hold for the trading test", "tone": "good"}],
        "notes": [f"The network has about {(_lstm_weights(len(cols), RNN_HIDDEN) + RNN_HIDDEN + 1) / n_tr:.1f} weights per training "
                  f"sequence for the first test year, so it can memorise noise easily; early stopping on validation years limits that.",
                  "Each test year gets its own refit, exactly as ridge and boosting on the returns page."],
    }


# ===================================================================== neural: text
explainer("markets/neural", {
    "title": "Neural networks on prices: how an LSTM works, and how it can fool you",
    "kicker": "Time Series Lab · Neural nets",
    "summary": "Recurrent networks such as the LSTM and GRU read a sequence one step at a time and carry a memory between steps. "
               "A popular tutorial uses one to forecast stock prices and reports an R² close to 1. "
               "This page explains the network, shows why that number is no evidence of skill, and what an honest test looks like.",
    "sections": [
        {"id": "rnn", "title": "Networks with memory", "blocks": [
            {"type": "p", "text": "A feed-forward network sees one input and gives one output. A <b>recurrent</b> network reads a sequence "
                                  "x₁, x₂, …, xₜ and keeps a <b>hidden state</b> hₜ, updated at every step from the new input and the "
                                  "previous state. After the last step, a linear layer turns hₜ into the forecast."},
            {"type": "eq", "text": "simple RNN:  hₜ = tanh(W xₜ + U hₜ₋₁ + b)",
             "where": [["xₜ", "the input at step t (here, one day's features)"], ["hₜ", "the hidden state, a vector of numbers"],
                       ["W, U, b", "weights learned by gradient descent"]]},
            {"type": "p", "text": "Training sends errors backwards through every step. In a simple RNN the gradient is multiplied by U and "
                                  "by the slope of tanh once per step, so over long sequences it shrinks towards zero (or blows up), "
                                  "and the network cannot learn long memories. The LSTM was designed to fix that."},
        ]},
        {"id": "lstm", "title": "Inside an LSTM: gates and a cell state", "blocks": [
            {"type": "p", "text": "The <b>long short-term memory</b> cell keeps a separate <b>cell state</b> cₜ that is changed only by "
                                  "multiplication by a gate and addition, so information and gradients can pass through many steps. "
                                  "Three gates, each a number between 0 and 1 for every unit, decide what happens to it."},
            {"type": "eq", "text": "fₜ = σ(W_f xₜ + U_f hₜ₋₁ + b_f)       forget\niₜ = σ(W_i xₜ + U_i hₜ₋₁ + b_i)       input\n"
                                   "c̃ₜ = tanh(W_c xₜ + U_c hₜ₋₁ + b_c)    candidate\noₜ = σ(W_o xₜ + U_o hₜ₋₁ + b_o)       output\n"
                                   "cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ\nhₜ = oₜ ⊙ tanh(cₜ)",
             "where": [["σ", "the logistic function, squashing to 0–1"], ["⊙", "element-by-element multiplication"],
                       ["fₜ", "how much of the old memory to keep"], ["iₜ, c̃ₜ", "how much new information to write, and what"],
                       ["oₜ", "how much of the memory to show as the output hₜ"]]},
            {"type": "figure", "figure": "nn-lstm", "title": "One LSTM step, with the live model's sizes",
             "caption": "Each gate is its own small layer reading today's input and yesterday's hidden state."},
            {"type": "figure", "figure": "nn-memory", "title": "The forget gate sets how long memories last",
             "caption": "If the forget gate stays at f and nothing new is written, a memory decays as fᵗ. Move f.",
             "controls": [{"k": "forget", "label": "Forget gate f", "min": 0.5, "max": 0.99, "step": 0.01, "value": 0.9}]},
            {"type": "compare", "items": [
                {"title": "LSTM", "points": ["Separate cell state and hidden state", "Forget, input and output gates", "Four weight blocks per layer"]},
                {"title": "GRU", "points": ["One state vector", "An update gate (forget and input in one) and a reset gate",
                                            "Three weight blocks: a quarter fewer weights", "So a little faster to train"]},
            ]},
        ]},
        {"id": "recipe", "title": "The tutorial recipe", "blocks": [
            {"type": "p", "text": "A widely copied tutorial forecasts a stock's price with an LSTM and gets a chart where the forecast "
                                  "hugs the actual price. The live view reproduces it exactly:"},
            {"type": "steps", "items": [
                "Take the daily adjusted closing price.",
                "Scale it to the range 0–1 with a min–max scaler fitted on the whole series.",
                f"Make windows of the last {TRAP_WINDOW} scaled prices, each with the next price as its target.",
                f"Train an LSTM with {TRAP_HIDDEN} units on the first {100 * NN.TRAP_TRAIN_SHARE:.0f}% of the dates, test on the rest.",
                "Plot the forecast against the actual price, and report R² or error on the price.",
            ]},
            {"type": "p", "text": "Two things are wrong, and they work together. The scaler has seen the test prices, and the score "
                                  "measures the price level, which any forecast that stays near today's price gets almost right."},
        ]},
        {"id": "leak", "title": "Trap 1: the scaler sees the future", "blocks": [
            {"type": "p", "text": "Min–max scaling divides by the highest price. Fitted on the whole series, it uses the test years' "
                                  "maximum, which is information from the future. Prices tend to rise over decades, so the test prices "
                                  "of a stock that has grown lie largely above anything in the training years."},
            {"type": "figure", "figure": "nn-scaler", "title": "The same prices, scaled two ways",
             "caption": "Move the training share. Above the horizontal line, the clean-scaled network sees prices it has never trained on.",
             "controls": [{"k": "share", "label": "Training share", "min": 0.5, "max": 0.9, "step": 0.05, "value": NN.TRAP_TRAIN_SHARE}, TICKER_CTRL]},
            {"type": "callout", "tone": "warn", "title": "Fit every transformation on training data only",
             "text": "Scalers, feature selection, outlier rules and hyper-parameter choices are all part of the model. Fitting any of them "
                     "on data that include the test period lets the future leak in, even when the model itself is trained correctly."},
        ]},
        {"id": "persist", "title": "Trap 2: tomorrow ≈ today", "blocks": [
            {"type": "p", "text": "Day to day, a price changes by a small fraction of its level. So a forecast that simply repeats "
                                  "today's price is never far off, and on a chart covering years it is indistinguishable from the truth."},
            {"type": "figure", "figure": "nn-persistence", "title": "The last 120 test days, against 'tomorrow = today'",
             "caption": "The no-skill forecast. Any model on prices has to beat this, not zero."},
            {"type": "figure", "figure": "nn-level-change", "title": "R² on prices against R² on changes",
             "caption": "Three forecasts with no skill. Scored on the price level, all look excellent; scored on the change, none does."},
            {"type": "p", "text": "Trained on price windows with a squared-error loss, a network's easiest way to a small error is to output "
                                  "something close to the last price in its window. That is what the live trap job checks: if the LSTM "
                                  "has learned to echo, its forecast changes line up with yesterday's actual change, not today's."},
            {"type": "figure", "figure": "nn-echo", "title": "The echo: forecasts that lag reality by a day",
             "caption": "Correlation between each forecast's daily change and the actual change some days earlier. "
                        "Move the smoothing span; the peak stays at lag 1.",
             "controls": [{"k": "span", "label": "Smoothing span", "min": 1, "max": 30, "step": 1, "value": 5, "unit": "days"}]},
            {"type": "quiz", "q": "A price-forecasting LSTM reports an R² of 0.98 on the test years. What should you compare it with first?",
             "options": ["An R² of 0", "The forecast 'tomorrow's price = today's price'", "A bigger LSTM"], "answer": 1,
             "why": "On price levels, repeating today's price already scores an R² near 1. A model is only useful if it beats that, "
                    "which means forecasting the change."},
            {"type": "try", "label": "Train the tutorial LSTM and its leak-free twin", "params": {"ticker": TICKER, "trap_epochs": 8, "window": TRAP_WINDOW},
             "note": "Opens the live view; press 'Train the LSTM trap'. Two networks train on the server, which takes some seconds."},
        ]},
        {"id": "honest", "title": "What an honest test looks like", "blocks": [
            {"type": "p", "text": "The fix is to ask the same question as on the returns page: forecast the next day's <b>return</b>, "
                                  "with leak-free features, refit walk-forward, and compare with forecasting zero and with buy and hold."},
            {"type": "figure", "figure": "nn-honest", "title": "The live walk-forward LSTM and GRU",
             "caption": "Settings from the server code that trains the networks."},
            {"type": "steps", "title": "Each refit", "items": [
                f"Build sequences of the last {NN.RNN_WINDOW} days of the same features ridge and boosting use.",
                "Train on days before the validation years; after each epoch score the validation years.",
                f"Stop when validation error has not improved for {RNN_PATIENCE} epochs, and keep the best epoch's weights.",
                "Forecast the test year, scale back by volatility, and score with R², hit rate, Diebold–Mariano and the long/flat back-test.",
            ]},
            {"type": "quiz", "q": "Which step of the tutorial leaks information from the test period?",
             "options": ["Using a 60-day window", "Fitting the min–max scaler on the whole price series", "Training for 10 epochs"], "answer": 1,
             "why": "The scaler's maximum comes from the test years. Window length and epochs affect accuracy but use only training data."},
            {"type": "try", "label": "Train a GRU walk-forward on returns",
             "params": {"ticker": TICKER, "net": "gru", "rnn_epochs": 6, "start_year": START},
             "note": "Press the walk-forward button in the live view. One network per test year is trained, so it takes a while."},
            {"type": "callout", "tone": "warn", "title": "Limits, and not investment advice",
             "text": "Neural networks are not wrong for markets; they are a flexible model class that finds whatever pattern lowers "
                     "the loss, including leaks and trivial ones. A small network on public daily data faces the same noise as every "
                     "other model on the returns page. Nothing here is investment advice."},
        ]},
    ],
    "takeaways": [
        "An LSTM carries a cell state that gates keep, overwrite or reveal; the GRU does the same with fewer weights.",
        "Fitting a scaler on the whole series leaks the test period's range into training.",
        "On price levels, 'tomorrow = today' scores an R² near 1, so a high R² on prices is no evidence of skill.",
        "A model that echoes the last price shows a cross-correlation peak at lag 1: it follows the price instead of leading it.",
        "Honest evaluation forecasts returns, refits walk-forward and compares with zero and buy and hold.",
    ],
    "glossary": [
        ["Recurrent network", "A network that reads a sequence step by step, carrying a hidden state."],
        ["LSTM", "Long short-term memory: a recurrent cell with forget, input and output gates and a cell state."],
        ["GRU", "Gated recurrent unit: a simpler gated cell with update and reset gates."],
        ["Gate", "A layer with outputs between 0 and 1 that scales how much information passes."],
        ["Min–max scaler", "Maps values to 0–1 using the smallest and largest values it was fitted on."],
        ["Leakage", "Information from the test period reaching the model through training or preprocessing."],
        ["Persistence forecast", "Tomorrow's value = today's value: the no-skill benchmark for prices."],
        ["Early stopping", "Ending training when the validation error stops improving."],
    ],
}, order=3)
