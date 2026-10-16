"""/api/markets: walk-forward return and volatility forecasting, computed live
from the daily price CSVs (see modlab.markets). Slow neural-network work runs
as background jobs ("rnn_returns", "lstm_trap") through /api/jobs.
"""
from __future__ import annotations

import numpy as np
from flask import Blueprint

from ..markets import common as C
from ..markets import neural, returns as RET, stylized as STY, volatility as VOL
from .api import ApiError, choice, choices, endpoint, num
from .cache import memo
from .jobs import register

bp = Blueprint("markets", __name__, url_prefix="/api/markets")

RETURN_TICKERS = C.TICKERS + ["pooled"]


def _ticker(options=C.TICKERS) -> str:
    return choice("ticker", "SPY", options)


def _years() -> tuple[int, int]:
    start = num("start_year", 2015, C.YEAR_MIN, C.YEAR_MAX, integer=True)
    end = num("end_year", 2026, C.YEAR_MIN, C.YEAR_MAX, integer=True)
    if start > end:
        raise ApiError(f"start_year ({start}) must not be after end_year ({end}).", "invalid_parameter")
    return start, end


def _with_cache_flag(pair) -> dict:
    value, cached = pair
    return {**value, "_cached": cached}


# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------
@bp.get("/tickers")
@endpoint
def tickers():
    RET.warm_pool()          # start the model-fitting workers while the page loads
    ranges = {t: {"start": C.prices(t).index[0].strftime("%Y-%m-%d"),
                  "end": C.prices(t).index[-1].strftime("%Y-%m-%d")} for t in C.TICKERS}
    return {
        "tickers": C.TICKERS,
        "start": min(r["start"] for r in ranges.values()),
        "end": max(r["end"] for r in ranges.values()),
        "test_years": C.TEST_YEARS,
        "year_bounds": [C.YEAR_MIN, C.YEAR_MAX],
        "ranges": ranges,
        "vol_models": VOL.VOL_MODELS,
        "return_models": RET.RET_MODELS,
        "pooled_models": RET.POOLED_MODELS,
        "rnn_available": C.torch_available(),
    }


@memo(maxsize=32)
def _prices(ticker: str, every: int) -> dict:
    p = C.prices(ticker)["AdjClose"]
    s = p.iloc[::every]
    lr = np.log(s).diff()
    return {"ticker": ticker, "every": every, "dates": C.iso(s.index), "price": s.values,
            "log_return": lr.values}


@bp.get("/prices")
@endpoint
def prices():
    t = _ticker()
    every = num("every", 1, 1, 20, integer=True)
    return _with_cache_flag(_prices(t, every))


@bp.get("/volatility")
@endpoint
def volatility():
    t = _ticker()
    models = choices("models", VOL.VOL_MODELS, VOL.VOL_MODELS)
    start, end = _years()
    alpha = num("var_alpha", 0.01, 0.001, 0.1)
    var_model = choice("var_model", "har_x", VOL.VOL_MODELS)
    models = tuple(m for m in VOL.VOL_MODELS if m in models)       # canonical order -> cache hits
    return _with_cache_flag(VOL.volatility(t, models, start, end, float(alpha), var_model))


@bp.get("/returns")
@endpoint
def returns():
    t = _ticker(RETURN_TICKERS)
    pooled = t == "pooled"
    allowed = RET.POOLED_MODELS if pooled else RET.RET_MODELS
    models = choices("models", allowed, RET.RET_MODELS)
    if pooled and "arima" in models:
        raise ApiError("ticker=pooled supports hist_mean, ridge and gbm (ARIMA is fit per ticker).",
                       "invalid_parameter")
    start, end = _years()
    cost = num("cost_bps", 5, 0, 100)
    models = tuple(m for m in RET.RET_MODELS if m in models)
    return _with_cache_flag(RET.returns(t, models, start, end, float(cost)))


@bp.get("/stylized")
@endpoint
def stylized():
    t = _ticker()
    max_lag = num("max_lag", 100, 5, 250, integer=True)
    return _with_cache_flag(STY.stylized(t, max_lag))


# ----------------------------------------------------------------------------
# Background jobs (POST /api/jobs {"kind": ..., ...params}; poll GET /api/jobs/<id>)
# ----------------------------------------------------------------------------
def _parse_rnn(body: dict) -> dict:
    start, end = _years()
    return {"ticker": _ticker(), "model": choice("model", "lstm", ["lstm", "gru"]),
            "epochs": num("epochs", 10, 1, 30, integer=True), "start_year": start, "end_year": end,
            "cost_bps": float(num("cost_bps", 5, 0, 100))}


def _parse_trap(body: dict) -> dict:
    return {"ticker": _ticker(), "epochs": num("epochs", 10, 1, 30, integer=True),
            "window": num("window", 60, 20, 120, integer=True)}


register("rnn_returns", params=_parse_rnn)(neural.rnn_returns)
register("lstm_trap", params=_parse_trap)(neural.lstm_trap)
