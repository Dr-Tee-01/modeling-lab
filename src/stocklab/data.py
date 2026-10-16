"""Loading prices and building leak-free features.

Convention used everywhere: a row dated t holds only information known at the
close of day t. The targets for that row are about day t+1.
"""
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
TICKERS = ["SPY", "AAPL", "MSFT", "NVDA", "JPM", "XOM"]


def load_prices(ticker):
    df = pd.read_csv(ROOT / f"data/prices/{ticker}.csv", index_col="Date", parse_dates=True)
    return df.sort_index()


def daily_panel(ticker, vix=None):
    """Per-ticker frame with log returns and a realized-variance proxy."""
    p = load_prices(ticker)
    adj = p["AdjClose"]
    k = adj / p["Close"]                         # split/dividend adjustment factor, same day
    o, h, l, c = p["Open"] * k, p["High"] * k, p["Low"] * k, adj
    out = pd.DataFrame(index=p.index)
    out["price"] = adj
    out["ret"] = np.log(adj).diff()                                  # close-to-close log return
    overnight = np.log(o / c.shift(1))
    gk = 0.5 * np.log(h / l) ** 2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2   # Garman-Klass
    rv = overnight ** 2 + gk.clip(lower=0)
    floor = rv[rv > 0].quantile(0.01)
    out["rv"] = rv.clip(lower=floor)                                  # daily variance proxy
    out["volume"] = p["Volume"].replace(0, np.nan)
    if vix is not None:
        out["vix"] = vix.reindex(out.index).ffill()
    return out.iloc[1:]


def load_vix():
    return load_prices("VIX")["Close"]


def build_features(d, spy_ret=None):
    """Scale-free features at day t. Returns (X, aux) where aux holds helpers."""
    r = d["ret"]
    vol63 = r.rolling(63).std()
    f = pd.DataFrame(index=d.index)
    for k in range(1, 11):                                            # vol-normalised lags
        f[f"z_lag{k}"] = r.shift(k - 1) / vol63
    for n in (5, 21, 63, 252):                                        # momentum
        f[f"mom{n}"] = r.rolling(n).sum() / (vol63 * np.sqrt(n))
    f["vol5_63"] = np.log(r.rolling(5).std() / vol63)
    f["vol21_63"] = np.log(r.rolling(21).std() / vol63)
    f["rv1_21"] = np.log(d["rv"] / d["rv"].rolling(21).mean())
    for n in (50, 200):
        f[f"ma{n}"] = np.log(d["price"] / d["price"].rolling(n).mean()) / vol63
    up = r.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-r.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi14"] = (up / (up + dn)) * 2 - 1
    f["vlm63"] = np.log(d["volume"] / d["volume"].rolling(63).mean())
    if "vix" in d:
        f["log_vix"] = np.log(d["vix"])
        f["vix_chg"] = np.log(d["vix"]).diff()
        f["vix_rel63"] = np.log(d["vix"] / d["vix"].rolling(63).mean())
    if spy_ret is not None:
        f["spy_z_lag1"] = spy_ret.reindex(d.index) / spy_ret.reindex(d.index).rolling(63).std()
    aux = pd.DataFrame({"vol63": vol63, "y": r.shift(-1)}, index=d.index)
    return f, aux


def har_features(d):
    """HAR-RV inputs (Corsi 2009) on log variance, plus VIX-implied variance."""
    lrv = np.log(d["rv"])
    f = pd.DataFrame(index=d.index)
    f["lrv_d"] = lrv
    f["lrv_w"] = np.log(d["rv"].rolling(5).mean())
    f["lrv_m"] = np.log(d["rv"].rolling(22).mean())
    f["neg_ret"] = np.minimum(d["ret"], 0) ** 2 / d["rv"].rolling(22).mean()   # leverage effect
    if "vix" in d:
        f["lvix2"] = np.log((d["vix"] / 100) ** 2 / 252)
    target = lrv.shift(-1)
    return f, target
