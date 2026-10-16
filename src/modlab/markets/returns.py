"""Walk-forward forecasts of next-day returns for one ticker (or the pooled panel).

Same protocol and features as stocklab.experiments.returns_experiment: for each
test year Y every model is refit on rows whose target falls before Y (last two
years held out for hyper-parameter selection); ridge / gradient boosting learn
the volatility-scaled return y / vol63 (winsorised at +-5) and their forecasts
are scaled back by vol63. ARIMA works on % returns with the order chosen once
by AIC on data before the first test year. "zero" is the benchmark.

Speed: stocklab's gradient-boosting search fits 4 configurations x 400
iterations per year. On a few thousand rows that is dominated by (a) OpenMP
start-up cost per boosting iteration and (b) Python overhead in the tree grower,
which holds the GIL. So the yearly ridge/GBM fits run in a persistent pool of
worker *processes* (spawned lazily, one OpenMP thread each). Same models and
hyper-parameter search, identical results; ~4x faster than one process.
Set MODLAB_MARKETS_PROCS=0 to use threads instead (e.g. where spawning
processes is not allowed).
"""
from __future__ import annotations

import multiprocessing
import os
import threading
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from stocklab import data as D
from stocklab import evaluate as E
from stocklab import models as M

from ..server.cache import memo
from .common import TICKERS, panel, weekly_equity, year_range

RET_MODELS = ["hist_mean", "ridge", "gbm", "arima"]
POOLED_MODELS = ["hist_mean", "ridge", "gbm"]
WORKERS = int(os.environ.get("MODLAB_MARKETS_WORKERS", 0)) or max(1, min(6, (os.cpu_count() or 2) - 1))
USE_PROCS = os.environ.get("MODLAB_MARKETS_PROCS", "1") != "0"
_pool_lock = threading.Lock()
_pool = None
_warmed = False


def _noop():
    import sklearn.ensemble  # noqa: F401  (pay the import cost while warming up)
    return os.getpid()


def process_pool():
    """Persistent, lazily created process pool (None if disabled/unavailable)."""
    global _pool
    if not USE_PROCS:
        return None
    with _pool_lock:
        if _pool is None:
            try:
                _pool = ProcessPoolExecutor(max_workers=WORKERS, mp_context=multiprocessing.get_context("spawn"))
            except Exception:  # pragma: no cover - platform without process support
                return None
        return _pool


def warm_pool() -> None:
    """Start the workers in the background so the first /returns call is fast."""
    global _warmed
    pool = process_pool()
    if pool is not None and not _warmed:
        _warmed = True
        for _ in range(WORKERS):
            pool.submit(_noop)


def _reset_pool():
    global _pool, _warmed
    with _pool_lock:
        _pool, _warmed = None, False


# ----------------------------------------------------------------------------
# Feature panel
# ----------------------------------------------------------------------------
def _ticker_frame(t: str) -> pd.DataFrame:
    d = panel(t, with_vix=True)
    spy = panel("SPY", with_vix=True)
    X, aux = D.build_features(d, spy_ret=None if t == "SPY" else spy["ret"])
    if t == "SPY":
        X["spy_z_lag1"] = X["z_lag1"]
    fr = X.join(aux)
    fr["ticker"] = t
    fr["target_date"] = pd.Series(d.index, index=d.index).shift(-1)
    return fr


@memo(maxsize=8)
def return_panel(scope: str):
    """stocklab.experiments.build_return_panel for one ticker or all ('pooled')."""
    tickers = TICKERS if scope == "pooled" else [scope]
    P = pd.concat([_ticker_frame(t) for t in tickers]).reset_index().rename(columns={"index": "Date"})
    if "Date" not in P.columns:
        P = P.rename(columns={P.columns[0]: "Date"})
    cols = [c for c in P.columns if c not in ("Date", "ticker", "target_date", "vol63", "y")]
    P = P.dropna(subset=cols + ["vol63"]).sort_values(["ticker", "Date"]).reset_index(drop=True)
    return P, cols


def design(scope: str, start_year: int):
    """Standardised, clipped features (scaler fit before the first validation
    window, so leak-free for every fold) and the vol-scaled target."""
    (P, cols), _ = return_panel(scope)
    dates = pd.DatetimeIndex(P["Date"])
    tdates = pd.DatetimeIndex(P["target_date"])
    ys = np.nan_to_num((P["y"] / P["vol63"]).clip(-5, 5).values)
    pre = np.asarray(tdates < pd.Timestamp(f"{start_year - M.VAL_YEARS}-01-01"))
    with np.errstate(all="ignore"):
        sc = StandardScaler().fit(P.loc[pre, cols].values)
        Xs = np.clip(sc.transform(P[cols].values), -5, 5).astype(np.float32)
    return P, cols, dates, tdates, ys, Xs


# ----------------------------------------------------------------------------
# Model fitting
# ----------------------------------------------------------------------------
def _fit_year(model: str, Xs, ys, tr, va, te):
    """One yearly refit (runs in a worker); returns (test-row predictions, info)."""
    with threadpool_limits(limits=1, user_api="openmp"), np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = M.fit_ridge if model == "ridge" else M.fit_gbm
        p, info = fit(Xs, ys, tr, va)
    return p[te], info


def _start_fits(tasks: list[tuple]):
    """Start the fits (argument tuples for _fit_year) and return a function that
    collects their results, so other work (ARIMA) can overlap with them.
    Uses the process pool when available, else a thread pool."""
    def threaded():
        with ThreadPoolExecutor(max_workers=max(1, min(WORKERS, len(tasks)))) as ex:
            return list(ex.map(lambda t: _fit_year(*t), tasks))

    pool = process_pool()
    if pool is None or not tasks:
        return threaded
    try:
        futs = [pool.submit(_fit_year, *t) for t in tasks]
    except (BrokenProcessPool, RuntimeError):
        _reset_pool()
        return threaded

    def collect():
        try:
            return [f.result() for f in futs]
        except BrokenProcessPool:
            _reset_pool()
            return threaded()
    return collect


@memo(maxsize=32)
def arima_ticker(ticker: str, start_year: int, end_year: int):
    """Per-ticker ARIMA(p,0,q) one-step forecasts (as fractions) via stocklab."""
    d = panel(ticker, with_vix=False)
    with warnings.catch_warnings():
        f, order = M.arima_forecasts(d["ret"] * 100, d.index, years=year_range(start_year, end_year))
    return f / 100, list(order)


def walk_forward(scope: str, models: tuple, start_year: int, end_year: int):
    P, cols, dates, tdates, ys, Xs = design(scope, start_year)
    n = len(P)
    has = ~np.isnan(P["y"].values)
    preds = {m: np.full(n, np.nan) for m in models}
    info = {m: {} for m in models}
    folds, test = [], np.zeros(n, dtype=bool)
    for year in year_range(start_year, end_year):
        tr, va, te = M.fold_masks(dates, tdates, year)
        if te.sum() == 0:
            continue
        folds.append((year, tr & has, va & has, te))
        test |= te

    tickers = P["ticker"].values
    if "hist_mean" in models:
        for year, tr, va, te in folds:
            for t in np.unique(tickers):
                g = tickers == t
                preds["hist_mean"][te & g] = P.loc[(tr | va) & g, "y"].mean()

    # GBM first: the slowest tasks should start first
    tasks = [(m, f) for m in ("gbm", "ridge") if m in models for f in folds]
    if tasks:
        collect = _start_fits([(m, Xs, ys, tr, va, te) for m, (year, tr, va, te) in tasks])

    if "arima" in models:                      # runs here while the pool fits ridge/GBM
        for t in np.unique(tickers):
            (f, order), _ = arima_ticker(t, start_year, end_year)
            g = tickers == t
            preds["arima"][g] = f.reindex(P.loc[g, "Date"]).values
            info["arima"][t if scope == "pooled" else "order"] = order

    if tasks:
        vol63 = P["vol63"].values
        for (m, (year, tr, va, te)), (p, i) in zip(tasks, collect()):
            preds[m][te] = p * vol63[te]
            info[m][year] = i
    return P, preds, info, test & has


def score_and_backtest(P, preds: dict, test: np.ndarray, models: list, cost_bps: float) -> dict:
    """Scores (stocklab.evaluate.return_scores), long/flat backtests and weekly equity."""
    R = P.loc[test, ["Date", "ticker", "y"]].copy()
    for m in models:
        R[m] = preds[m][test]
    R["zero"] = 0.0
    R = R.dropna().reset_index(drop=True)
    names = ["zero"] + list(models)
    with np.errstate(all="ignore"):
        sc = E.return_scores(R, names)
    up = sc.pop("_always_up_hit_rate")
    keys = ("r2_oos", "rmse_pct", "mae_pct", "hit_rate", "hit_p", "dm_stat", "dm_p", "n")
    out_models = {m: {k: sc[m].get(k) for k in keys} for m in names}
    out_models["zero"]["n"] = int(len(R))

    pnl = {}
    tickers = list(dict.fromkeys(R["ticker"]))
    for k in list(models) + ["buy_hold"]:
        parts = []
        for t in tickers:
            Rt = R[R.ticker == t]
            if k == "buy_hold":
                s = pd.Series(np.expm1(Rt["y"].values), index=pd.DatetimeIndex(Rt["Date"]))
            else:
                s = E.backtest(Rt["y"].values, Rt[k].values, pd.DatetimeIndex(Rt["Date"]), cost_bps=cost_bps)[0]
            parts.append(s)
        # one ticker: its own long/flat P&L; pooled: equal-weight across tickers (as stocklab)
        pnl[k] = parts[0] if len(parts) == 1 else pd.concat(parts, axis=1).fillna(0).mean(axis=1)
    backtest = {k: E.perf(v) for k, v in pnl.items()}
    for k in models:
        f = R[k].values
        backtest[k]["time_in_market"] = float(np.mean(f > 0))
    return {"models": out_models, "always_up_hit_rate": up, "backtest": backtest,
            "equity": weekly_equity(pnl), "n_test": int(len(R)),
            "test_start": R["Date"].min().strftime("%Y-%m-%d"), "test_end": R["Date"].max().strftime("%Y-%m-%d")}


@memo(maxsize=64)
def returns(scope: str, models: tuple, start_year: int, end_year: int, cost_bps: float) -> dict:
    P, preds, info, test = walk_forward(scope, models, start_year, end_year)
    out = {"ticker": scope, "years": year_range(start_year, end_year)}
    out.update(score_and_backtest(P, preds, test, list(models), cost_bps))
    out["hyperparams"] = info
    out["cost_bps"] = cost_bps
    out["target"] = "next-day close-to-close log return"
    return out
