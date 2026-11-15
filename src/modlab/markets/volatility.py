"""Per-ticker walk-forward forecasts of next-day log realised variance and a
1-day Value-at-Risk backtest. Mirrors stocklab.experiments.vol_experiment
(hist21 / EWMA / GARCH / HAR / HAR-X) for one ticker and a range of test years.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import lfilter

from stocklab import data as D
from stocklab import evaluate as E
from stocklab import models as M

from ..server.cache import memo
from .common import iso, kupiec_dict, panel, year_range
from .garch import garch_year

VOL_MODELS = ["hist21", "ewma", "garch", "har", "har_x"]
LEVEL_MODELS = ("hist21", "ewma", "garch")          # level forecasts with a log-bias calibration
HAR_COLS = {"har": ["lrv_d", "lrv_w", "lrv_m"], "har_x": ["lrv_d", "lrv_w", "lrv_m", "lvix2", "neg_ret"]}


def ewma_var(ret: pd.Series, lam: float = 0.94) -> pd.Series:
    """stocklab.models.ewma_var with the recursion as an IIR filter."""
    r2 = ret.values ** 2
    s0 = np.nanmean(r2[:63])
    tail = lfilter([1 - lam], [1.0, -lam], r2[1:], zi=[lam * s0])[0]
    return pd.Series(np.r_[s0, tail], index=ret.index)


@memo(maxsize=16)
def vol_frame(ticker: str) -> pd.DataFrame:
    d = panel(ticker, with_vix=True)
    F, target = D.har_features(d)
    fr = F.copy()
    fr["y"] = target
    fr["ret_next"] = d["ret"].shift(-1)
    fr["hist21"] = np.log(d["rv"].rolling(21).mean())
    fr["ewma"] = np.log(ewma_var(d["ret"].fillna(0)))
    fr["target_date"] = pd.Series(d.index, index=d.index).shift(-1)
    fr = fr.dropna(subset=["lrv_d", "lrv_w", "lrv_m", "lvix2", "hist21", "ewma", "neg_ret"])
    return fr


def _lstsq(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = np.c_[np.ones(len(X)), X]
    with np.errstate(all="ignore"):
        return np.linalg.lstsq(A, y, rcond=None)[0]


def _predict(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    # errstate: numpy 2.0 + Accelerate emits spurious FP warnings from matmul on macOS
    with np.errstate(all="ignore"):
        return coef[0] + X @ coef[1:]


def vol_summary(scores: dict, har_coef: dict) -> dict:
    """What the tiles and insights state: the best model by out-of-sample R² of log
    variance, the baseline it is compared with (the 21-day average when selected, else
    the weakest model), the gain in R², and the latest yearly HAR coefficients."""
    ranked = sorted(scores, key=lambda m: -scores[m]["r2_log"])
    best = ranked[0]
    base = ("hist21" if "hist21" in scores else ranked[-1]) if len(ranked) > 1 else None
    latest = {m: {"year": max(by), "coef": by[max(by)]} for m, by in har_coef.items() if by}
    return {"best": best, "baseline": base,
            "gain_over_baseline": None if base is None else scores[best]["r2_log"] - scores[base]["r2_log"],
            "har_latest": latest}


@memo(maxsize=64)
def volatility(ticker: str, models: tuple, start_year: int, end_year: int,
               var_alpha: float, var_model: str) -> dict:
    V, _ = vol_frame(ticker)
    dates, tdates = pd.DatetimeIndex(V.index), pd.DatetimeIndex(V["target_date"])
    y = V["y"].values
    has = ~np.isnan(y)
    n = len(V)
    run = [m for m in VOL_MODELS if m in models or m == var_model]
    preds = {m: np.full(n, np.nan) for m in run}
    resid = {m: np.full(n, np.nan) for m in run}
    fhs_q = np.full(n, np.nan)
    test = np.zeros(n, dtype=bool)
    years, garch_params, har_coef = [], {}, {}
    ret_next = V["ret_next"].values

    for year in year_range(start_year, end_year):
        tr, va, te = M.fold_masks(dates, tdates, year)
        if te.sum() == 0:
            continue
        years.append(year)
        test |= te
        fit = (tr | va) & has
        level = {}
        if "hist21" in run:
            level["hist21"] = V["hist21"].values
        if "ewma" in run:
            level["ewma"] = V["ewma"].values
        if "garch" in run:
            g, _ = garch_year(ticker, year)
            if g is not None:
                series, th = g
                level["garch"] = np.log(series.reindex(dates).values)
                garch_params[year] = th
        fitted = {}
        for m, lv in level.items():
            ok = fit & ~np.isnan(lv)
            c = np.mean(y[ok] - lv[ok])
            preds[m][te] = lv[te] + c
            resid[m][te] = np.var(y[ok] - lv[ok] - c)
            fitted[m] = (ok, lv[ok] + c)
        for m in ("har", "har_x"):
            if m not in run:
                continue
            Xf = V[HAR_COLS[m]].values
            coef = _lstsq(Xf[fit], y[fit])
            fy = _predict(coef, Xf[fit])
            preds[m][te] = _predict(coef, Xf[te])
            resid[m][te] = np.var(y[fit] - fy)
            fitted[m] = (fit, fy)
            har_coef.setdefault(m, {})[year] = [float(v) for v in coef]
        # filtered historical simulation for the VaR model: empirical quantile of
        # returns standardised by the model's own in-sample volatility
        ok, fy = fitted[var_model]
        sig_in = np.sqrt(np.exp(fy + 0.5 * np.var(y[ok] - fy)))
        z = ret_next[ok] / sig_in
        z = z[~np.isnan(z)]
        fhs_q[te] = np.quantile(z, var_alpha)

    if not years:
        raise ValueError("no test data in the requested years")
    test &= has
    for m in run:
        test &= ~np.isnan(preds[m])
    yt = y[test]
    scores = {m: E.vol_scores(yt, preds[m][test], resid[m][test]) for m in VOL_MODELS if m in models}

    # Value-at-Risk (1 day, mean zero): normal quantile and FHS quantile
    sig = np.sqrt(np.exp(preds[var_model][test] + 0.5 * resid[var_model][test]))
    rn = ret_next[test]
    thr_norm = stats.norm.ppf(var_alpha) * sig
    thr_fhs = fhs_q[test] * sig
    normal = kupiec_dict(rn < thr_norm, var_alpha)
    fhs = kupiec_dict(rn < thr_fhs, var_alpha)
    var = {"model": var_model, "alpha": var_alpha, **normal, "normal": normal, "fhs": fhs}

    tdx = dates[test]
    return {
        "summary": vol_summary(scores, har_coef),
        "ticker": ticker,
        "years": years,
        "models": scores,
        "series": {
            "dates": iso(tdx),
            "realized": yt,
            "forecasts": {m: preds[m][test] for m in VOL_MODELS if m in models},
            "next_return": rn,
            "var_normal": thr_norm,
            "var_fhs": thr_fhs,
            "breach_normal": np.nonzero(rn < thr_norm)[0],      # indices of the test days past each VaR
            "breach_fhs": np.nonzero(rn < thr_fhs)[0],
        },
        "var": var,
        "params": {"garch": garch_params, "har": har_coef},
        "n_test": int(test.sum()),
        "target": "log of next-day realised variance (Garman-Klass + overnight squared return)",
    }
