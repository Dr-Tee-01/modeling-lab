"""PyTorch background jobs (torch is imported lazily, inside the jobs).

rnn_returns  walk-forward LSTM/GRU on stocklab's return features for one ticker
lstm_trap    stocklab.experiments.trap_experiment with per-epoch progress
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from stocklab import models as M

from .common import iso, prices, require_torch, stride
from .returns import design, score_and_backtest

RNN_WINDOW = 20          # as stocklab.experiments.returns_experiment
TRAP_MAX_POINTS = 800


def rnn_returns(params: dict, progress) -> dict:
    require_torch()
    ticker, kind, epochs = params["ticker"], params["model"], params["epochs"]
    start_year, end_year = params["start_year"], params["end_year"]
    cost_bps = params.get("cost_bps", 5.0)
    progress(0.01, "Building features")
    P, cols, dates, tdates, ys, Xs = design(ticker, start_year)
    seq, ok = M.make_sequences(Xs, P["ticker"].values, RNN_WINDOW)
    has = ~np.isnan(P["y"].values)
    folds = []
    for year in range(start_year, end_year + 1):
        tr, va, te = M.fold_masks(dates, tdates, year)
        if te.sum():
            folds.append((year, tr & has, va & has, te))
    pred = np.full(len(P), np.nan)
    test = np.zeros(len(P), dtype=bool)
    epochs_used, val_mse = {}, {}
    vol63 = P["vol63"].values
    for i, (year, tr, va, te) in enumerate(folds):
        progress(0.02 + 0.93 * i / len(folds), f"Training {kind.upper()} for test year {year} ({i + 1}/{len(folds)})")
        with np.errstate(all="ignore"):
            p, info = M.fit_rnn(kind, seq, ok, ys, tr, va, epochs=epochs)
        pred[te] = p[te] * vol63[te]
        test |= te
        epochs_used[year] = info["epochs"]
        val_mse[year] = info["val_mse"]
    progress(0.97, "Scoring and backtesting")
    out = {"ticker": ticker, "model": kind, "epochs": epochs, "window": RNN_WINDOW,
           "years": [f[0] for f in folds]}
    out.update(score_and_backtest(P, {kind: pred}, test & has, [kind], cost_bps))
    out["epochs_used"] = epochs_used
    out["val_mse"] = val_mse
    return out


def lstm_trap(params: dict, progress) -> dict:
    """The tutorial recipe: MinMax-scaled *price*, sliding window, LSTM(50),
    chronological 80/20 split. (a) scaler fit on the full series (leaks the
    test range), (b) scaler fit on the training part only; vs 'tomorrow = today'."""
    torch = require_torch()
    import torch.nn as nn

    ticker, win, epochs, seed = params["ticker"], params["window"], params["epochs"], M.SEED
    price = prices(ticker)["AdjClose"]
    vals = price.values.reshape(-1, 1)
    split = int(len(vals) * 0.8)
    total = 2 * epochs

    def run(scaler, run_idx: int, label: str):
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
        losses = []
        for ep in range(epochs):
            progress((run_idx * epochs + ep) / total, f"{label}: epoch {ep + 1}/{epochs}")
            tot, nb = 0.0, 0
            for b in np.array_split(g.permutation(idx), max(1, len(idx) // 32)):
                opt.zero_grad()
                o, _ = net(Xt[b])
                loss = ((head(o[:, -1]).squeeze(-1) - yt[b]) ** 2).mean()
                loss.backward()
                opt.step()
                tot, nb = tot + loss.item(), nb + 1
            losses.append(tot / nb)
        with torch.no_grad():
            o, _ = net(Xt)
            p = head(o[:, -1]).squeeze(-1).numpy()
        pred = scaler.inverse_transform(p.reshape(-1, 1)).ravel()
        te = ~tr
        return tgt_idx[te], pred[te], losses

    idx_a, pred_a, loss_a = run(MinMaxScaler().fit(vals), 0, "Leaky scaler (fit on all prices)")
    idx_b, pred_b, loss_b = run(MinMaxScaler().fit(vals[:split]), 1, "Clean scaler (fit on training prices)")
    progress(0.99, "Scoring")

    actual = vals.ravel()[idx_a]
    naive = vals.ravel()[idx_a - 1]

    def scores(p):
        e = actual - p
        return {"rmse": float(np.sqrt(np.mean(e ** 2))),
                "mape_pct": float(np.mean(np.abs(e) / actual) * 100),
                "r2": float(1 - np.sum(e ** 2) / np.sum((actual - actual.mean()) ** 2))}

    chg_actual, chg_pred = actual - naive, pred_a - naive
    dp, da = np.diff(pred_a), np.diff(actual)
    xcorr = {lag: float(np.corrcoef(dp[lag:], da[:len(da) - lag])[0, 1]) for lag in range(6)}
    k = stride(len(actual), TRAP_MAX_POINTS)
    dates = price.index[idx_a]
    return {
        "ticker": ticker, "window": win, "epochs": epochs,
        "dates": iso(dates[::k]), "actual": actual[::k], "naive": naive[::k],
        "leaky": pred_a[::k], "clean": pred_b[::k], "stride": k,
        "scores": {"naive": scores(naive), "leaky": scores(pred_a), "clean": scores(pred_b)},
        "direction_hit": float(np.mean(np.sign(chg_pred) == np.sign(chg_actual))),
        "change_corr": float(np.corrcoef(chg_pred, chg_actual)[0, 1]),
        "xcorr": xcorr,
        "train_loss": {"leaky": loss_a, "clean": loss_b},
        "split_date": price.index[split].strftime("%Y-%m-%d"),
        "train_max": float(vals[:split].max()), "test_max": float(vals[split:].max()),
        "n_test": int(len(actual)),
    }
