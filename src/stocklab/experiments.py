"""The three experiments of the project.

1. trap_experiment     the popular "LSTM predicts stock prices" tutorial, and why it fools you
2. returns_experiment  honest next-day return forecasting, walk-forward, six tickers
3. vol_experiment      next-day volatility forecasting and Value-at-Risk
"""
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from . import data as D
from . import evaluate as E
from . import models as M

RET_MODELS = ["zero", "hist_mean", "arima", "ridge", "gbm", "lstm", "gru", "ensemble"]
VOL_MODELS = ["hist21", "ewma", "garch", "har", "har_x", "gbm"]


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:6.1f}s] {msg}", flush=True)


# ============================================================================
# 1. The trap
# ============================================================================
def trap_experiment(ticker="SPY", win=60, epochs=10, seed=M.SEED):
    """Tutorial recipe: MinMax-scale the *price*, 60-day window, LSTM(50), 80/20 split."""
    import torch
    import torch.nn as nn

    price = D.load_prices(ticker)["AdjClose"]
    vals = price.values.reshape(-1, 1)
    split = int(len(vals) * 0.8)
    out = {"dates": price.index, "price": price.values, "split": split}

    def run(scaler):
        torch.manual_seed(seed)
        s = scaler.transform(vals).astype(np.float32).ravel()
        X = np.lib.stride_tricks.sliding_window_view(s[:-1], win)[:, :, None].copy()
        y = s[win:]
        tgt_idx = np.arange(win, len(s))
        tr = tgt_idx < split
        net = nn.LSTM(1, 50, batch_first=True)
        head = nn.Linear(50, 1)
        opt = torch.optim.Adam(list(net.parameters()) + list(head.parameters()), lr=1e-3)
        Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
        g = np.random.default_rng(seed)
        idx = np.where(tr)[0]
        for _ in range(epochs):
            for b in np.array_split(g.permutation(idx), len(idx) // 32):
                opt.zero_grad()
                o, _ = net(Xt[b])
                loss = ((head(o[:, -1]).squeeze(-1) - yt[b]) ** 2).mean()
                loss.backward()
                opt.step()
        with torch.no_grad():
            o, _ = net(Xt)
            p = head(o[:, -1]).squeeze(-1).numpy()
        pred = scaler.inverse_transform(p.reshape(-1, 1)).ravel()
        te = ~tr
        return tgt_idx[te], pred[te]

    # (a) the tutorial: scaler fit on the full series (leaks the test range)
    idx_a, pred_a = run(MinMaxScaler().fit(vals))
    # (b) fixed leak: scaler fit on the training period only
    idx_b, pred_b = run(MinMaxScaler().fit(vals[:split]))

    actual = vals.ravel()[idx_a]
    naive = vals.ravel()[idx_a - 1]                 # "tomorrow = today"

    def scores(p):
        e = actual - p
        return {"rmse": float(np.sqrt(np.mean(e ** 2))),
                "mape_pct": float(np.mean(np.abs(e) / actual) * 100),
                "r2": float(1 - np.sum(e ** 2) / np.sum((actual - actual.mean()) ** 2))}

    chg_actual = actual - naive
    chg_pred = pred_a - naive
    # direction of the predicted move vs the realised move
    hit = float(np.mean(np.sign(chg_pred) == np.sign(chg_actual)))
    # cross-correlation of predicted level changes with actual changes at several lags
    dp = np.diff(pred_a)
    da = np.diff(actual)
    xcorr = {lag: float(np.corrcoef(dp[lag:], da[:len(da) - lag])[0, 1]) for lag in range(0, 6)}
    out.update({
        "test_idx": idx_a, "actual": actual, "naive": naive, "pred_leaky": pred_a, "pred_clean": pred_b,
        "scores": {"lstm_leaky": scores(pred_a), "lstm_clean": scores(pred_b), "naive": scores(naive)},
        "direction_hit": hit,
        "change_corr": float(np.corrcoef(chg_pred, chg_actual)[0, 1]),
        "xcorr": xcorr,
        "train_max": float(vals[:split].max()),
        "test_max": float(vals[split:].max()),
    })
    return out


# ============================================================================
# 2. Returns
# ============================================================================
def build_return_panel():
    vix = D.load_vix()
    spy = D.daily_panel("SPY", vix)
    frames = []
    for t in D.TICKERS:
        d = D.daily_panel(t, vix)
        X, aux = D.build_features(d, spy_ret=None if t == "SPY" else spy["ret"])
        if t == "SPY":
            X["spy_z_lag1"] = X["z_lag1"]
        fr = X.join(aux)
        fr["ticker"] = t
        fr["target_date"] = pd.Series(d.index, index=d.index).shift(-1)
        frames.append(fr)
    P = pd.concat(frames).reset_index().rename(columns={"index": "Date"})
    feat_cols = [c for c in P.columns if c not in ("Date", "ticker", "target_date", "vol63", "y")]
    P = P.dropna(subset=feat_cols + ["vol63"]).sort_values(["ticker", "Date"]).reset_index(drop=True)
    return P, feat_cols


def returns_experiment(win=20):
    P, cols = build_return_panel()
    log(f"return panel: {len(P):,} rows, {len(cols)} features")
    dates = pd.DatetimeIndex(P["Date"])
    tdates = pd.DatetimeIndex(P["target_date"])
    ys = (P["y"] / P["vol63"]).clip(-5, 5).values          # vol-scaled, winsorised target
    ys = np.nan_to_num(ys)

    # standardise with statistics from before the first validation window (leak-free for all folds)
    pre = np.asarray(tdates < pd.Timestamp(f"{M.TEST_YEARS[0] - M.VAL_YEARS}-01-01"))
    sc = StandardScaler().fit(P.loc[pre, cols].values)
    Xs = np.clip(sc.transform(P[cols].values), -5, 5).astype(np.float32)
    seq, ok = M.make_sequences(Xs, P["ticker"].values, win)

    preds = {m: np.full(len(P), np.nan) for m in RET_MODELS}
    info = {m: {} for m in RET_MODELS}
    for year in M.TEST_YEARS:
        tr, va, te = M.fold_masks(dates, tdates, year)
        if te.sum() == 0:
            continue
        has_y = ~np.isnan(P["y"].values)
        tr, va = tr & has_y, va & has_y
        p, i = M.fit_ridge(Xs, ys, tr, va); preds["ridge"][te] = p[te]; info["ridge"][year] = i
        p, i = M.fit_gbm(Xs, ys, tr, va); preds["gbm"][te] = p[te]; info["gbm"][year] = i
        for kind in ("lstm", "gru"):
            p, i = M.fit_rnn(kind, seq, ok, ys, tr, va); preds[kind][te] = p[te]; info[kind][year] = i
        for t in D.TICKERS:
            g = (P["ticker"] == t).values
            preds["hist_mean"][te & g] = P.loc[(tr | va) & g, "y"].mean()
        log(f"fold {year}: train {tr.sum():,}  val {va.sum():,}  test {te.sum():,}")

    vol = P["vol63"].values
    for m in ("ridge", "gbm", "lstm", "gru"):
        preds[m] = preds[m] * vol                               # back to return units
    preds["zero"] = np.zeros(len(P))
    preds["ensemble"] = np.mean([preds[m] for m in ("ridge", "gbm", "lstm", "gru")], axis=0)

    # ARIMA per ticker on the ticker's own return series
    arima_orders = {}
    for t in D.TICKERS:
        d = D.daily_panel(t)
        f, order = M.arima_forecasts(d["ret"] * 100, d.index)
        arima_orders[t] = list(order)
        g = (P["ticker"] == t).values
        preds["arima"][g] = (f / 100).reindex(P.loc[g, "Date"]).values
        log(f"ARIMA{order} {t}")

    test = np.zeros(len(P), dtype=bool)
    for year in M.TEST_YEARS:
        test |= M.fold_masks(dates, tdates, year)[2]
    test &= ~np.isnan(P["y"].values)
    R = P.loc[test, ["Date", "ticker", "y"]].copy()
    for m in RET_MODELS:
        R[m] = preds[m][test]
    R = R.dropna().reset_index(drop=True)

    scores = {"pooled": E.return_scores(R, RET_MODELS)}
    for t in D.TICKERS:
        scores[t] = E.return_scores(R[R.ticker == t], RET_MODELS)

    # economic value: long/flat by model sign, per ticker, plus equal-weight portfolio
    bt, bt_perf = {}, {}
    for t in D.TICKERS:
        Rt = R[R.ticker == t]
        bt[t] = {"buy_hold": pd.Series(np.expm1(Rt["y"].values), index=Rt["Date"])}
        for m in RET_MODELS[1:]:
            bt[t][m] = E.backtest(Rt["y"].values, Rt[m].values, Rt["Date"])[0]
        bt_perf[t] = {k: E.perf(v) for k, v in bt[t].items()}
    ew = {k: pd.concat([bt[t][k] for t in D.TICKERS], axis=1).fillna(0).mean(axis=1)
          for k in bt["SPY"]}
    bt_perf["equal_weight"] = {k: E.perf(v) for k, v in ew.items()}
    bt["equal_weight"] = ew

    # feature importance proxy: correlation of each feature with next-day scaled return (train only)
    corr = {c: float(np.corrcoef(P.loc[pre, c].values, ys[pre])[0, 1]) for c in cols}
    return {"R": R, "scores": scores, "backtests": bt, "bt_perf": bt_perf, "info": info,
            "arima_orders": arima_orders, "feature_corr": corr, "features": cols, "panel_rows": len(P)}


# ============================================================================
# 3. Volatility
# ============================================================================
def vol_experiment():
    vix = D.load_vix()
    rows = []
    garch_by_year, garch_params = {}, {}
    for t in D.TICKERS:
        d = D.daily_panel(t, vix)
        F, target = D.har_features(d)
        fr = F.copy()
        fr["y"] = target
        fr["ret"] = d["ret"]
        fr["ret_next"] = d["ret"].shift(-1)
        fr["hist21"] = np.log(d["rv"].rolling(21).mean())
        fr["ewma"] = np.log(M.ewma_var(d["ret"].fillna(0)))
        garch_by_year[t], garch_params[t] = M.garch_forecasts(d["ret"])
        fr["ticker"] = t
        fr["target_date"] = pd.Series(d.index, index=d.index).shift(-1)
        rows.append(fr)
        log(f"vol features + GARCH {t}")
    V = pd.concat(rows).reset_index().rename(columns={"index": "Date"})
    V = V.dropna(subset=["lrv_d", "lrv_w", "lrv_m", "lvix2", "hist21", "ewma", "neg_ret"]).reset_index(drop=True)
    V = V.sort_values(["ticker", "Date"]).reset_index(drop=True)
    dates, tdates = pd.DatetimeIndex(V["Date"]), pd.DatetimeIndex(V["target_date"])
    har_cols = ["lrv_d", "lrv_w", "lrv_m"]
    harx_cols = har_cols + ["lvix2", "neg_ret"]
    preds = {m: np.full(len(V), np.nan) for m in VOL_MODELS}
    resid_var = {m: np.full(len(V), np.nan) for m in VOL_MODELS}
    fhs_q = {0.05: np.full(len(V), np.nan), 0.01: np.full(len(V), np.nan)}
    y = V["y"].values
    for year in M.TEST_YEARS:
        tr, va, te = M.fold_masks(dates, tdates, year)
        if te.sum() == 0:
            continue
        has = ~np.isnan(y)
        fit = (tr | va) & has
        for t in D.TICKERS:
            gk = (V["ticker"] == t).values
            g_year = np.log(garch_by_year[t][year].reindex(V.loc[gk, "Date"]).values)
            level = {"hist21": V["hist21"].values, "ewma": V["ewma"].values,
                     "garch": np.full(len(V), np.nan)}
            level["garch"][gk] = g_year
            for m in ("hist21", "ewma", "garch"):         # level forecasts: calibrate the log bias
                ok = fit & gk & ~np.isnan(level[m])
                c = np.mean(y[ok] - level[m][ok])
                preds[m][te & gk] = level[m][te & gk] + c
                resid_var[m][te & gk] = np.var(y[ok] - level[m][ok] - c)
            for m, cols in (("har", har_cols), ("har_x", harx_cols)):
                ok = fit & gk
                lr = LinearRegression().fit(V.loc[ok, cols].values, y[ok])
                fitted = lr.predict(V.loc[ok, cols].values)
                preds[m][te & gk] = lr.predict(V.loc[te & gk, cols].values)
                resid_var[m][te & gk] = np.var(y[ok] - fitted)
                if m == "har_x":
                    # filtered historical simulation: empirical quantiles of returns
                    # standardised by the model's own in-sample volatility
                    sig_in = np.sqrt(np.exp(fitted + 0.5 * np.var(y[ok] - fitted)))
                    z = V.loc[ok, "ret_next"].values / sig_in
                    z = z[~np.isnan(z)]
                    for a in (0.05, 0.01):
                        fhs_q[a][te & gk] = np.quantile(z, a)
        # pooled gradient boosting on HAR-X features + ticker-agnostic return features
        cols = harx_cols + ["ret"]
        Xv = V[cols].values
        best = None
        for n_iter in (100, 200, 400):
            m = HistGradientBoostingRegressor(learning_rate=0.05, max_leaf_nodes=15, max_iter=n_iter,
                                              min_samples_leaf=200, random_state=M.SEED)
            m.fit(Xv[tr & has], y[tr & has])
            e = np.mean((m.predict(Xv[va & has]) - y[va & has]) ** 2)
            if best is None or e < best[0]:
                best = (e, n_iter)
        m = HistGradientBoostingRegressor(learning_rate=0.05, max_leaf_nodes=15, max_iter=best[1],
                                          min_samples_leaf=200, random_state=M.SEED).fit(Xv[fit], y[fit])
        preds["gbm"][te] = m.predict(Xv[te])
        resid_var["gbm"][te] = np.var(y[fit] - m.predict(Xv[fit]))
        log(f"vol fold {year}")

    test = np.zeros(len(V), dtype=bool)
    for year in M.TEST_YEARS:
        test |= M.fold_masks(dates, tdates, year)[2]
    test &= ~np.isnan(y)
    W = V.loc[test, ["Date", "ticker", "y", "ret_next", "lvix2"]].copy()
    for m in VOL_MODELS:
        W[m] = preds[m][test]
        W[m + "_rv"] = resid_var[m][test]
    W["fhs_q05"], W["fhs_q01"] = fhs_q[0.05][test], fhs_q[0.01][test]
    n_before = len(W)
    W = W.dropna().reset_index(drop=True)
    assert len(W) >= n_before - 6, f"unexpected NaNs dropped: {n_before - len(W)}"   # only last day per ticker

    scores = {}
    for scope in ["pooled"] + D.TICKERS:
        Wt = W if scope == "pooled" else W[W.ticker == scope]
        scores[scope] = {m: E.vol_scores(Wt["y"].values, Wt[m].values, Wt[m + "_rv"].values) for m in VOL_MODELS}

    # Value-at-Risk backtest (1-day, mean zero, normal quantile), all tickers pooled and SPY alone
    var_res = {}
    for scope in ["SPY", "pooled"]:
        Wt = W if scope == "pooled" else W[W.ticker == "SPY"]
        var_res[scope] = {}
        for m in ("hist21", "ewma", "garch", "har_x", "har_x_fhs"):
            base = "har_x" if m == "har_x_fhs" else m
            sig = np.sqrt(np.exp(Wt[base].values + 0.5 * Wt[base + "_rv"].values))
            var_res[scope][m] = {}
            for a in (0.05, 0.01):
                q = Wt[f"fhs_q{int(a * 100):02d}"].values if m == "har_x_fhs" else stats.norm.ppf(a)
                exc = Wt["ret_next"].values < q * sig
                var_res[scope][m][str(a)] = E.kupiec(exc, a)
    return {"W": W, "scores": scores, "var": var_res, "garch_params": garch_params}
