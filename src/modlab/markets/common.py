"""Shared data access and helpers for the markets lab."""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd

from stocklab import data as D
from stocklab import evaluate as E
from stocklab import models as M

from ..server.cache import memo

TICKERS = list(D.TICKERS)                     # SPY AAPL MSFT NVDA JPM XOM
TEST_YEARS = list(M.TEST_YEARS)               # 2015 .. 2026
YEAR_MIN, YEAR_MAX = 2005, 2026
TORCH_MISSING = "PyTorch is not installed. Install it with: pip install -e '.[rnn]'"


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def require_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        raise RuntimeError(TORCH_MISSING) from None
    return torch


# ----------------------------------------------------------------------------
# Raw data (memoised: reading a CSV is cheap but not free). Callers must not
# mutate the returned frames.
# ----------------------------------------------------------------------------
@memo(maxsize=16)
def _prices(ticker: str) -> pd.DataFrame:
    return D.load_prices(ticker)


def prices(ticker: str) -> pd.DataFrame:
    return _prices(ticker)[0]


@memo(maxsize=2)
def _vix() -> pd.Series:
    return prices("VIX")["Close"]


def vix() -> pd.Series:
    return _vix()[0]


@memo(maxsize=16)
def _panel(ticker: str, with_vix: bool) -> pd.DataFrame:
    return D.daily_panel(ticker, vix() if with_vix else None)


def panel(ticker: str, with_vix: bool = True) -> pd.DataFrame:
    """stocklab.data.daily_panel: price, log return, realised-variance proxy (+ VIX)."""
    return _panel(ticker, with_vix)[0]


def year_range(start: int, end: int) -> list[int]:
    return list(range(int(start), int(end) + 1))


def last_date() -> pd.Timestamp:
    return max(prices(t).index[-1] for t in TICKERS)


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def iso(dates) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.DatetimeIndex(dates)]


def stride(n: int, max_points: int) -> int:
    return max(1, int(np.ceil(n / max_points)))


def weekly_equity(pnl: dict[str, pd.Series]) -> dict:
    """Cumulative equity curves sampled at the last trading day of each week."""
    eq = pd.DataFrame({k: (1 + v).cumprod() for k, v in pnl.items()})
    eq = eq.ffill().fillna(1.0)
    wk = eq.index.to_period("W-FRI")
    keep = ~pd.Series(wk, index=eq.index).duplicated(keep="last").values
    keep[-1] = True
    sub = eq.loc[keep]
    return {"dates": iso(sub.index), "curves": {k: sub[k].values for k in sub.columns}}


def kupiec_dict(exceed: np.ndarray, alpha: float) -> dict:
    k = E.kupiec(exceed, alpha)
    return {"n": k["n"], "exceed": k["exceed"], "rate": k["rate"], "kupiec_lr": k["lr"], "kupiec_p": k["p"]}
