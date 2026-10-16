"""Forecasting models and the walk-forward engine.

Walk-forward protocol: for each test year Y, every model is (re)fit on data
strictly before Y, with the last VAL_YEARS of that window held out for model
selection / early stopping, then predicts each day of Y one step ahead.
Rows whose target falls in year Y are never in the training set.
"""
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

SEED = 42
VAL_YEARS = 2
TEST_YEARS = list(range(2015, 2027))


def fold_masks(dates, target_dates, year):
    """train / val / test boolean masks for a test year."""
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year + 1}-01-01")
    val_start = pd.Timestamp(f"{year - VAL_YEARS}-01-01")
    train = np.asarray(target_dates < val_start)
    val = np.asarray((target_dates >= val_start) & (target_dates < start))
    test = np.asarray((dates >= start) & (dates < end) & target_dates.notna())
    return train, val, test


# ----------------------------------------------------------------------------
# Return models (pooled across tickers, target = next-day return / vol63)
# ----------------------------------------------------------------------------
def fit_ridge(X, y, tr, va, alphas=(1, 10, 100, 1000, 10000)):
    sc = StandardScaler().fit(X[tr])
    Xs = sc.transform(X)
    best = None
    for a in alphas:
        m = Ridge(alpha=a).fit(Xs[tr], y[tr])
        e = np.mean((m.predict(Xs[va]) - y[va]) ** 2)
        if best is None or e < best[0]:
            best = (e, a)
    m = Ridge(alpha=best[1]).fit(Xs[tr | va], y[tr | va])
    return m.predict(Xs), {"alpha": best[1]}


def fit_gbm(X, y, tr, va):
    best = None
    for lr, leaves in itertools.product((0.02, 0.05), (7, 15)):
        m = HistGradientBoostingRegressor(
            learning_rate=lr, max_leaf_nodes=leaves, max_iter=400, min_samples_leaf=400,
            l2_regularization=1.0, early_stopping=False, random_state=SEED)
        m.fit(X[tr], y[tr])
        # pick the iteration count on validation (staged predictions)
        errs = [np.mean((p - y[va]) ** 2) for p in m.staged_predict(X[va])]
        k = int(np.argmin(errs))
        if best is None or errs[k] < best[0]:
            best = (errs[k], lr, leaves, k + 1)
    _, lr, leaves, n_iter = best
    m = HistGradientBoostingRegressor(
        learning_rate=lr, max_leaf_nodes=leaves, max_iter=max(n_iter, 1), min_samples_leaf=400,
        l2_regularization=1.0, early_stopping=False, random_state=SEED)
    m.fit(X[tr | va], y[tr | va])
    return m.predict(X), {"lr": lr, "leaves": leaves, "iters": n_iter}


def make_sequences(F, group, win):
    """Stack per-ticker windows: row i -> F[i-win+1 .. i] (same ticker only)."""
    n, k = F.shape
    seq = np.zeros((n, win, k), dtype=np.float32)
    ok = np.zeros(n, dtype=bool)
    for g in np.unique(group):
        idx = np.where(group == g)[0]
        Fg = F[idx]
        for j in range(win - 1, len(idx)):
            seq[idx[j]] = Fg[j - win + 1:j + 1]
            ok[idx[j]] = True
    return seq, ok


def fit_rnn(kind, seq, ok, y, tr, va, epochs=25, hidden=32, lr=1e-3, patience=4):
    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    tr, va = tr & ok, va & ok

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            cell = nn.LSTM if kind == "lstm" else nn.GRU
            self.rnn = cell(seq.shape[2], hidden, batch_first=True)
            self.drop = nn.Dropout(0.2)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(self.drop(out[:, -1])).squeeze(-1)

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    lossf = nn.MSELoss()
    yt = torch.from_numpy(y.astype(np.float32))
    St = torch.from_numpy(seq)
    va_idx = np.where(va)[0]
    tr_idx = np.where(tr)[0]
    best, state, bad, used = np.inf, None, 0, 0
    for ep in range(epochs):
        net.train()
        for b in np.array_split(rng.permutation(tr_idx), max(1, len(tr_idx) // 512)):
            opt.zero_grad()
            loss = lossf(net(St[b]), yt[b])
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            v = lossf(net(St[va_idx]), yt[va_idx]).item()
        if v < best - 1e-5:
            best, bad, used = v, 0, ep + 1
            state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state)
    net.eval()
    pred = np.zeros(len(y))
    with torch.no_grad():
        for b in np.array_split(np.arange(len(y)), max(1, len(y) // 8192)):
            pred[b] = net(St[b]).numpy()
    return pred, {"epochs": used, "val_mse": best}


def arima_forecasts(ret_pct, dates, years=TEST_YEARS, orders=None):
    """Per-ticker ARIMA(p,0,q) on % returns: order chosen once by AIC on pre-test
    data, parameters re-estimated each year, one-step forecasts through the year."""
    dates = pd.DatetimeIndex(dates)
    warnings.filterwarnings("ignore")
    first = dates < pd.Timestamp(f"{years[0]}-01-01")
    if orders is None:
        best = None
        for p, q in itertools.product(range(3), range(3)):
            if p == q == 0:
                continue
            res = ARIMA(ret_pct[first].values, order=(p, 0, q), trend="c").fit()
            if best is None or res.aic < best[0]:
                best = (res.aic, (p, 0, q))
        orders = best[1]
    first = np.asarray(first)
    pred = pd.Series(np.nan, index=dates)                # forecast of ret[t+1] made at t
    for y in years:
        tr = np.asarray(dates < pd.Timestamp(f"{y}-01-01"))
        te = np.asarray((dates >= pd.Timestamp(f"{y}-01-01")) & (dates < pd.Timestamp(f"{y + 1}-01-01")))
        if te.sum() == 0:
            continue
        res = ARIMA(ret_pct[tr].values, order=orders, trend="c").fit()
        res2 = res.append(ret_pct[te].values, refit=False)
        # one-step forecasts for every day from the first test day to one past the last
        f = res2.predict(start=int(tr.sum()), end=int(tr.sum() + te.sum()))
        # f[j] forecasts the return on test day j (j = 0..), made at the previous close
        idx_prev = np.where(tr)[0][-1]
        origin_idx = np.r_[idx_prev, np.where(te)[0]]          # origins: last train day + test days
        pred.iloc[origin_idx] = np.asarray(f)[:len(origin_idx)]
    return pred, orders


# ----------------------------------------------------------------------------
# Volatility models (per ticker unless noted)
# ----------------------------------------------------------------------------
def ewma_var(ret, lam=0.94):
    """RiskMetrics: sigma2[t] is the variance forecast for t+1 made at t."""
    r2 = ret.values ** 2
    s = np.empty_like(r2)
    s[0] = np.nanmean(r2[:63])
    for t in range(1, len(r2)):
        s[t] = lam * s[t - 1] + (1 - lam) * r2[t]
    return pd.Series(s, index=ret.index)


def garch_fit(r):
    """Gaussian GARCH(1,1) with constant mean by maximum likelihood (r in %)."""
    r = np.asarray(r)
    v0 = np.var(r)

    def nll(theta):
        mu, w, a, b = theta
        if w <= 0 or a < 0 or b < 0 or a + b >= 0.9999:
            return 1e12
        e = r - mu
        s2 = np.empty_like(e)
        s2[0] = v0
        for t in range(1, len(e)):
            s2[t] = w + a * e[t - 1] ** 2 + b * s2[t - 1]
        return 0.5 * np.sum(np.log(2 * np.pi * s2) + e ** 2 / s2)

    x0 = [r.mean(), v0 * 0.05, 0.08, 0.9]
    res = minimize(nll, x0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-4})
    return res.x


def garch_filter(r, theta, v0):
    """sigma2_next[t] = variance forecast for t+1 given data through t."""
    mu, w, a, b = theta
    e = np.asarray(r) - mu
    s2 = np.empty_like(e)
    nxt = np.empty_like(e)
    s2[0] = v0
    for t in range(len(e)):
        if t > 0:
            s2[t] = w + a * e[t - 1] ** 2 + b * s2[t - 1]
        nxt[t] = w + a * e[t] ** 2 + b * s2[t]
    return nxt


def garch_forecasts(ret, years=TEST_YEARS, window_years=12):
    """Refit yearly on a rolling window. Returns {year: Series} where each Series
    holds that year's variance forecasts (for t+1, made at t) over the fit window
    and the test year, using parameters estimated on the fit window only."""
    r = ret * 100
    out, params = {}, {}
    for y in years:
        lo = pd.Timestamp(f"{y - window_years}-01-01")
        start = pd.Timestamp(f"{y}-01-01")
        end = pd.Timestamp(f"{y + 1}-01-01")
        tr = (r.index >= lo) & (r.index < start)
        if ((r.index >= start) & (r.index < end)).sum() == 0:
            continue
        th = garch_fit(r[tr].values)
        params[y] = [float(v) for v in th]
        span = (r.index >= lo) & (r.index < end)
        nxt = garch_filter(r[span].values, th, np.var(r[tr].values))
        out[y] = pd.Series(nxt / 1e4, index=r.index[span])
    return out, params
