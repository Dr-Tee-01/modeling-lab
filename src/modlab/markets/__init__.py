"""Markets lab: live, walk-forward forecasting of returns and volatility.

Everything here is computed on request from the raw daily price CSVs in
``data/prices`` using the ``stocklab`` study code (features, models, tests).
Only in-memory memoisation of pure results is used; nothing is precomputed.

Modules
-------
common      data loading, constants, small helpers
garch       fast GARCH(1,1) (variance recursion as an IIR filter via lfilter)
volatility  next-day log realised variance forecasts + Value-at-Risk backtest
returns     next-day return forecasts, significance tests, long/flat backtests
neural      PyTorch jobs: walk-forward LSTM/GRU returns and the "LSTM trap"
stylized    stylised facts of daily returns (ACF, fat tails)
"""
from __future__ import annotations

from .common import TICKERS, TEST_YEARS  # noqa: F401
