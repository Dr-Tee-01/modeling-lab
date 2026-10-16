"""Forecast evaluation: accuracy, statistical significance, economic value."""
import numpy as np
import pandas as pd
from scipy import stats

COST_BPS = 5          # one-way transaction cost per unit of turnover
DAYS = 252


def r2_oos(y, f, bench=None):
    """Campbell-Thompson out-of-sample R^2 against a benchmark forecast (default 0)."""
    bench = np.zeros_like(y) if bench is None else bench
    return 1 - np.sum((y - f) ** 2) / np.sum((y - bench) ** 2)


def diebold_mariano(y, f1, f2, h=1, lags=5):
    """DM test, squared-error loss, Newey-West variance. Positive stat = f2 better than f1."""
    d = (y - f1) ** 2 - (y - f2) ** 2
    n = len(d)
    dm = d - d.mean()
    gamma = [np.sum(dm[k:] * dm[:n - k]) / n for k in range(lags + 1)]
    var = gamma[0] + 2 * sum((1 - k / (lags + 1)) * gamma[k] for k in range(1, lags + 1))
    stat = d.mean() / np.sqrt(var / n)
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))


def hit_rate(y, f, base=0.5):
    """Share of days with the right direction; one-sided binomial p-value against
    a base rate (use the share of up days: 'always predict up' is the real bar)."""
    m = (y != 0) & (f != 0)
    hits = np.sign(y[m]) == np.sign(f[m])
    k, n = int(hits.sum()), int(m.sum())
    p = stats.binomtest(k, n, base, alternative="greater").pvalue
    return k / n, n, float(p)


def return_scores(df, models):
    """df: rows = (ticker, origin date) with column y and one column per model."""
    out = {}
    y = df["y"].values
    up_share = float((y[y != 0] > 0).mean())          # "always up" hit rate on non-zero days
    for m in models:
        f = df[m].values
        row = {
            "r2_oos": float(r2_oos(y, f)),
            "rmse_pct": float(np.sqrt(np.mean((y - f) ** 2)) * 100),
            "mae_pct": float(np.mean(np.abs(y - f)) * 100),
        }
        if m != "zero":
            hr, n, p = hit_rate(y, f, base=up_share)
            dm, dmp = diebold_mariano(y, np.zeros_like(y), f)
            row.update({"hit_rate": hr, "hit_p": p, "dm_stat": dm, "dm_p": dmp, "n": n})
        out[m] = row
    out["_always_up_hit_rate"] = up_share
    return out


def backtest(y, f, dates, cost_bps=COST_BPS):
    """Long when the forecast is positive, flat otherwise. y are next-day log returns."""
    pos = (f > 0).astype(float)
    simple = np.expm1(y)
    turnover = np.abs(np.diff(np.r_[0.0, pos]))
    pnl = pos * simple - turnover * cost_bps / 1e4
    return pd.Series(pnl, index=dates), pos


def perf(pnl):
    pnl = pd.Series(pnl).dropna()
    eq = (1 + pnl).cumprod()
    years = len(pnl) / DAYS
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = pnl.std() * np.sqrt(DAYS)
    sharpe = pnl.mean() / pnl.std() * np.sqrt(DAYS) if pnl.std() > 0 else np.nan
    dd = (eq / eq.cummax() - 1).min()
    return {"cagr": float(cagr), "vol": float(vol), "sharpe": float(sharpe), "max_dd": float(dd),
            "final_multiple": float(eq.iloc[-1])}


def vol_scores(y_log, f_log, resid_var):
    """y_log, f_log: log realised variance and its forecast. QLIKE on levels with a
    log-normal bias correction estimated in-sample (resid_var)."""
    rv = np.exp(y_log)
    h = np.exp(f_log + 0.5 * resid_var)
    q = rv / h - np.log(rv / h) - 1
    sst = np.sum((y_log - y_log.mean()) ** 2)
    return {
        "r2_log": float(1 - np.sum((y_log - f_log) ** 2) / sst),
        "rmse_log": float(np.sqrt(np.mean((y_log - f_log) ** 2))),
        "qlike": float(np.mean(q)),
        "corr": float(np.corrcoef(y_log, f_log)[0, 1]),
    }


def kupiec(exceed, alpha):
    """Kupiec proportion-of-failures test for VaR exceedances."""
    n, x = len(exceed), int(np.sum(exceed))
    p_hat = x / n
    if x == 0:
        lr = -2 * n * np.log(1 - alpha)
    else:
        lr = -2 * ((n - x) * np.log(1 - alpha) + x * np.log(alpha)
                   - (n - x) * np.log(1 - p_hat) - x * np.log(p_hat))
    return {"n": n, "exceed": x, "rate": p_hat, "lr": float(lr), "p": float(1 - stats.chi2.cdf(lr, 1))}
