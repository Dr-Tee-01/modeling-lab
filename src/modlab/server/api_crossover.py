"""/api/crossover: the market data through the spread and shape toolkits, live."""
from __future__ import annotations

import numpy as np
from flask import Blueprint

from ..crossover import study as S
from .api import ApiError, choice, choices, endpoint, flag, num
from .cache import memo

bp = Blueprint("crossover", __name__, url_prefix="/api/crossover")


def _value(fn, *args):
    try:
        out, cached = fn(*args)
    except ValueError as e:              # parameters that leave too little data to analyse
        raise ApiError(str(e), "insufficient_data", 422)
    return out, cached


@memo(maxsize=16)
def _random_walk(ticker, sims, perms, seed):
    return S.random_walk(ticker, sims, perms, seed)


@bp.get("/random-walk")
@endpoint
def random_walk():
    out, cached = _value(_random_walk, choice("ticker", "SPY", S.TICKERS), num("sims", 2000, 100, 5000, integer=True),
                         num("perms", 200, 50, 1000, integer=True), num("seed", 312, 0, 2**31 - 1, integer=True))
    return dict(out, _cached=cached)


@bp.get("/walks")
@endpoint
def walks():
    seed = num("seed", -1, -1, 2**31 - 1, integer=True)
    return S.walk_paths(choice("ticker", "SPY", S.TICKERS), num("n", 20, 1, 100, integer=True), None if seed < 0 else seed)


@memo(maxsize=32)
def _events(ratio, gap, horizon):
    return S.vix_events(ratio, gap, horizon)


@bp.get("/vix-events")
@endpoint
def vix_events():
    out, cached = _value(_events, num("ratio", 1.6, 1.2, 4), num("gap", 63, 5, 252, integer=True),
                         num("horizon", 60, 10, 120, integer=True))
    return dict(out, _cached=cached)


@bp.get("/vix-model")
@endpoint
def vix_model():
    comp = int(choice("compartments", "1", ["1", "2"]))
    out = S.vix_dose_model(half_life=num("half_life", 12.5, 0.5, 120), threshold=num("threshold", 3.4, 0.2, 20),
                           compartments=comp, fit=flag("fit"), a=num("a", 0.77, 0, 1),
                           hl_fast=num("hl_fast", 5.9, 0.2, 60), hl_slow=num("hl_slow", 44, 1, 400))
    idx, x, m = out.pop("_series")
    window = choice("window", "all", ["all", "2008", "2020", "2025"])
    ranges = {"all": (None, None), "2008": ("2007-06-01", "2010-01-01"),
              "2020": ("2019-10-01", "2021-01-01"), "2025": ("2024-10-01", "2026-12-31")}
    lo, hi = ranges[window]
    mask = np.ones(len(x), bool) if lo is None else (idx >= lo) & (idx < hi)
    sel = np.nonzero(mask)[0]
    step = -(-len(sel) // num("points", 1500, 100, 9000, integer=True))      # ceil, so we never exceed points
    sel = sel[::step]
    out.update(window=window, dates=[idx[i].strftime("%Y-%m-%d") for i in sel], vix=x[sel], model=m[sel])
    return out


@memo(maxsize=32)
def _shapes(spy_depth, stock_depth, landmarks, min_days, tickers):
    return S.crash_shapes(spy_depth, stock_depth, landmarks, min_days, tickers)


def _shape_params():
    return (num("spy_depth", 0.10, 0.05, 0.5), num("stock_depth", 0.20, 0.05, 0.8),
            num("landmarks", 48, 12, 200, integer=True), num("min_days", 15, 5, 252, integer=True),
            tuple(choices("tickers", S.TICKERS, S.TICKERS)))


@bp.get("/crash-shapes")
@endpoint
def crash_shapes():
    out, cached = _value(_shapes, *_shape_params())
    return dict({k: v for k, v in out.items() if k != "_model"}, _cached=cached)


@bp.get("/crash-sample")
@endpoint
def crash_sample():
    """A crash shape at b = (b1, b2, b3) standard deviations along the modes."""
    out, cached = _value(_shapes, *_shape_params())
    b = [num(f"b{i}", 0, -4, 4) for i in (1, 2, 3)]
    return {"s": out["s"], "curve": out["_model"].sample(b), "b": b, "_cached": cached}
