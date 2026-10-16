"""Matplotlib figures. Every function takes experiment output and saves a PNG."""
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.dates  # noqa: F401  (DateFormatter)
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
from statsmodels.tsa.stattools import acf

from . import data as D

FIG = D.ROOT / "figures"
INK, INK2, INK3, GRID = "#14202b", "#4b5864", "#7d8893", "#e6eaee"
COLORS = {  # fixed identity per model, validated categorical palette
    "zero": "#8c96a0", "hist_mean": "#bcc4cb", "arima": "#e87ba4", "ridge": "#1baf7a",
    "gbm": "#eb6834", "lstm": "#eda100", "gru": "#4a3aa7", "ensemble": "#2a78d6",
    "buy_hold": INK,
    "hist21": "#8c96a0", "ewma": "#1baf7a", "garch": "#e87ba4", "har": "#eb6834", "har_x": "#2a78d6",
}
NAMES = {
    "zero": "Random walk (0)", "hist_mean": "Historical mean", "arima": "ARIMA", "ridge": "Ridge",
    "gbm": "Gradient boosting", "lstm": "LSTM", "gru": "GRU", "ensemble": "Ensemble",
    "buy_hold": "Buy & hold",
    "hist21": "21-day historical", "ewma": "EWMA (RiskMetrics)", "garch": "GARCH(1,1)",
    "har": "HAR-RV", "har_x": "HAR-RV + VIX",
}
TICKER_COLORS = dict(zip(D.TICKERS, ["#14202b", "#2a78d6", "#1baf7a", "#eb6834", "#4a3aa7", "#e87ba4"]))


def style():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight", "savefig.facecolor": "white",
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 10.5,
        "axes.edgecolor": INK3, "axes.labelcolor": INK2, "axes.titlesize": 12.5, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.color": INK3, "ytick.color": INK3, "legend.frameon": False, "lines.linewidth": 1.6,
    })


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png")
    return FIG / f"{name}.png"


def note(ax, text, loc="upper left"):
    x, ha = (0.01, "left") if "left" in loc else (0.99, "right")
    y, va = (0.98, "top") if "upper" in loc else (0.03, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=9.5, color=INK2,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID))


# ----------------------------------------------------------------------------
def fig_growth():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for t in D.TICKERS:
        p = D.load_prices(t)["AdjClose"]
        p = p[p.index >= "2000-01-03"]
        ax.plot(p.index, p / p.iloc[0], color=TICKER_COLORS[t], lw=1.3, label=t)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}" if v >= 1 else f"${v:.2f}"))
    end = D.load_prices("SPY").index[-1]
    ax.axvspan(pd.Timestamp("2015-01-01"), end, color="#2a78d6", alpha=0.07, lw=0)
    ax.set_xlim(pd.Timestamp("2000-01-01"), end + pd.Timedelta(days=60))
    ax.text(pd.Timestamp("2015-03-01"), ax.get_ylim()[0] * 1.4, "walk-forward test period 2015–2026", color=INK2, fontsize=9)
    ax.set_title("Growth of $1 invested in January 2000 (dividend-adjusted, log scale)")
    ax.legend(ncol=6, loc="upper left")
    return save(fig, "fig01_growth")


def fig_trap(T):
    d = T["dates"][T["test_idx"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    ax.plot(d, T["actual"], color=INK, lw=1.4, label="Actual SPY close")
    ax.plot(d, T["pred_leaky"], color=COLORS["lstm"], lw=1.4, label="LSTM, tutorial recipe")
    ax.plot(d, T["pred_clean"], color=COLORS["arima"], lw=1.4, label="Same LSTM, scaler fit on train only")
    ax.axhline(T["train_max"], color=INK3, lw=0.8)
    ax.text(pd.Timestamp("2023-06-01"), T["train_max"] + 8, "highest price seen in training ($%.0f)" % T["train_max"],
            fontsize=8.5, color=INK3, ha="left", va="bottom")
    s = T["scores"]
    note(ax, f"Tutorial LSTM   R² {s['lstm_leaky']['r2']:.3f}   MAPE {s['lstm_leaky']['mape_pct']:.1f}%\n"
             f"Train-only scale R² {s['lstm_clean']['r2']:.3f}   MAPE {s['lstm_clean']['mape_pct']:.1f}%", "lower right")
    ax.set_title("It looks like it works")
    ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax = axes[1]
    # a 40-day window from the first year of the test period, where the tutorial model tracks the price
    k = slice(140, 180)
    ax.plot(d[k], T["actual"][k], color=INK, lw=1.6, marker="o", ms=3, label="Actual")
    ax.plot(d[k], T["pred_leaky"][k], color=COLORS["lstm"], lw=1.6, marker="o", ms=3, label="LSTM")
    ax.plot(d[k], T["naive"][k], color=COLORS["zero"], lw=1.2, ls="-", label="Yesterday's close")
    ax.set_title("Zoom in: a smoothed, delayed copy of the price")
    ax.legend(loc="best", fontsize=9)
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%b %d\n%Y"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    fig.tight_layout()
    return save(fig, "fig02_trap")


def fig_trap_diagnostics(T):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    ax = axes[0]
    labels = ["Yesterday's close", "Tutorial LSTM", "LSTM, no leak"]
    vals = [T["scores"]["naive"]["rmse"], T["scores"]["lstm_leaky"]["rmse"], T["scores"]["lstm_clean"]["rmse"]]
    cols = [COLORS["zero"], COLORS["lstm"], COLORS["arima"]]
    bars = ax.barh(labels, vals, color=cols, height=0.55)
    for b, v in zip(bars, vals):
        ax.text(v + max(vals) * 0.01, b.get_y() + b.get_height() / 2, f"${v:.1f}", va="center", fontsize=10, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE of next-day price, test period (USD)")
    ax.set_title("A one-line baseline beats it 4x")
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    lags = list(T["xcorr"].keys())
    ax.bar(lags, [T["xcorr"][l] for l in lags], color=[COLORS["lstm"] if l == 1 else "#d7dce1" for l in lags], width=0.6)
    ax.axhline(0, color=INK3, lw=0.8)
    ax.set_xticks(lags)
    ax.set_xticklabels(["same day"] + [f"{l} day{'s' if l > 1 else ''} later" for l in lags[1:]], fontsize=9)
    ax.set_ylabel("correlation")
    ax.set_title("Its moves match the market's, one day too late")
    note(ax, f"Direction of next-day move right {T['direction_hit']:.1%} of the time", "upper right")
    fig.tight_layout()
    return save(fig, "fig03_trap_lag")


def fig_stylized():
    r = D.daily_panel("SPY")["ret"].dropna()
    lags = 40
    a1 = acf(r, nlags=lags, fft=True)[1:]
    a2 = acf(r ** 2, nlags=lags, fft=True)[1:]
    a3 = acf(np.abs(r), nlags=lags, fft=True)[1:]
    band = 1.96 / np.sqrt(len(r))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    for ax, a, title, c in ((axes[0], a1, "Returns: almost no memory", COLORS["zero"]),
                            (axes[1], a3, "Size of returns: long memory", COLORS["ensemble"])):
        ax.bar(np.arange(1, lags + 1), a, color=c, width=0.7)
        ax.axhspan(-band, band, color=INK3, alpha=0.12, lw=0)
        ax.axhline(0, color=INK3, lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("lag (trading days)")
    axes[0].set_ylabel("autocorrelation, SPY 1993–2026")
    note(axes[0], "Shaded band: 95% range for pure noise", "upper right")
    fig.tight_layout()
    return save(fig, "fig04_stylized_facts")


def fig_return_leaderboard(Rx):
    S = Rx["scores"]["pooled"]
    models = [m for m in ["hist_mean", "arima", "ridge", "gbm", "lstm", "gru", "ensemble"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    ax = axes[0]
    v = [S[m]["r2_oos"] * 100 for m in models]
    bars = ax.barh([NAMES[m] for m in models], v, color=[COLORS[m] for m in models], height=0.6)
    ax.axvline(0, color=INK, lw=1)
    for b, x, m in zip(bars, v, models):
        sig = "*" if S[m]["dm_p"] < 0.05 else ""
        ax.text(x + (0.02 if x >= 0 else -0.02), b.get_y() + b.get_height() / 2, f"{x:+.2f}%{sig}",
                va="center", ha="left" if x >= 0 else "right", fontsize=9.5, color=INK)
    ax.set_xlim(min(0, min(v)) * 1.5 - 0.05, max(v) * 1.35)
    ax.invert_yaxis()
    ax.set_xlabel("out-of-sample R² vs. random walk (%)")
    ax.set_title("Return forecasts: share of variance explained")
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    hr = [S[m]["hit_rate"] * 100 for m in models]
    bars = ax.barh([NAMES[m] for m in models], hr, color=[COLORS[m] for m in models], height=0.6)
    up = S["_always_up_hit_rate"] * 100
    ax.axvline(50, color=INK3, lw=1)
    ax.axvline(up, color=INK, lw=1.2)
    right = max(hr + [up]) + 1.6
    for b, x in zip(bars, hr):
        ax.text(right, b.get_y() + b.get_height() / 2, f"{x:.1f}%", va="center", ha="right", fontsize=9.5, color=INK)
    ax.set_xlim(48, right)
    ax.set_xlabel(f"direction correct, % of days   (black line: always predict 'up' = {up:.1f}%)")
    ax.invert_yaxis()
    ax.set_yticklabels([])
    ax.set_title("Direction accuracy")
    ax.grid(axis="y", visible=False)
    fig.text(0.01, -0.03, "Pooled over 6 tickers, test years 2015–2026, walk-forward with yearly refits. "
             "* = Diebold-Mariano p < 0.05 against the random walk.", fontsize=9, color=INK3)
    fig.tight_layout()
    return save(fig, "fig05_return_leaderboard")


def fig_r2_heatmap(Rx):
    models = ["hist_mean", "arima", "ridge", "gbm", "lstm", "gru", "ensemble"]
    M = np.array([[Rx["scores"][t][m]["r2_oos"] * 100 for m in models] for t in D.TICKERS])
    lim = np.nanmax(np.abs(M))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([NAMES[m] for m in models])
    ax.set_yticks(range(len(D.TICKERS)))
    ax.set_yticklabels(D.TICKERS)
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=9.5,
                    color="white" if abs(M[i, j]) > lim * 0.6 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("out-of-sample R² (%)", color=INK2)
    ax.set_title("Out-of-sample R² by ticker (blue = beats random walk, red = worse)")
    return save(fig, "fig06_r2_by_ticker")


def fig_backtest(Rx):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
    for ax, scope, title in ((axes[0], "SPY", "SPY: long when the model says up, else cash"),
                             (axes[1], "equal_weight", "Six-stock equal-weight portfolio")):
        bt = Rx["backtests"][scope]
        for k in ["buy_hold", "hist_mean", "arima", "ridge", "gbm", "lstm", "gru", "ensemble"]:
            if k == "hist_mean":
                continue
            eq = (1 + bt[k]).cumprod()
            ax.plot(eq.index, eq.values, color=COLORS[k], lw=2.2 if k == "buy_hold" else 1.3,
                    label=f"{NAMES[k]}  (Sharpe {Rx['bt_perf'][scope][k]['sharpe']:.2f})")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}x" if f"{v:g}"[0] in "25" else ""))
        ax.set_title(title)
        ax.legend(fontsize=8.5, loc="upper left")
    fig.text(0.01, -0.03, "Includes 5 bp per trade. Cash earns 0. Strategy = sign of the model's next-day forecast.",
             fontsize=9, color=INK3)
    fig.tight_layout()
    return save(fig, "fig07_backtest")


def fig_vol_timeseries(Vx):
    W = Vx["W"]
    S = W[W.ticker == "SPY"].set_index("Date")
    ann = lambda lv: np.sqrt(np.exp(lv) * 252) * 100
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    real = ann(S["y"]).rolling(5).mean()
    ax.plot(S.index, real, color="#c9d0d6", lw=1.0, label="Realized (5-day avg)")
    ax.plot(S.index, ann(S["har_x"]), color=COLORS["har_x"], lw=1.2, label="HAR-RV + VIX forecast")
    ax.plot(S.index, ann(S["garch"]), color=COLORS["garch"], lw=1.0, label="GARCH(1,1) forecast")
    ax.set_ylabel("annualized volatility (%)")
    ax.set_title("SPY next-day volatility: forecast vs realized")
    ax.legend(loc="upper left")
    ax = axes[1]
    z = S.loc["2020-02-01":"2020-06-30"]
    ax.plot(z.index, ann(z["y"]), color="#9aa4ad", lw=1, marker="o", ms=2.5, label="Realized (daily)")
    ax.plot(z.index, ann(z["har_x"]), color=COLORS["har_x"], lw=1.6, label="HAR-RV + VIX")
    ax.plot(z.index, ann(z["garch"]), color=COLORS["garch"], lw=1.4, label="GARCH(1,1)")
    ax.plot(z.index, ann(z["hist21"]), color=COLORS["hist21"], lw=1.4, label="21-day historical")
    ax.set_title("COVID crash, Feb–Jun 2020")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    return save(fig, "fig08_vol_forecast")


def fig_vol_leaderboard(Vx):
    models = ["hist21", "ewma", "garch", "har", "har_x", "gbm"]
    S = Vx["scores"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    ax = axes[0]
    v = [S["pooled"][m]["r2_log"] * 100 for m in models]
    cols = [("#4a3aa7" if m == "gbm" else COLORS[m]) for m in models]
    bars = ax.barh([NAMES[m] for m in models], v, color=cols, height=0.6)
    for b, x in zip(bars, v):
        ax.text(x + 0.5, b.get_y() + b.get_height() / 2, f"{x:.1f}%", va="center", fontsize=9.5, color=INK)
    for i, m in enumerate(models):
        xs = [S[t][m]["r2_log"] * 100 for t in D.TICKERS]
        ax.scatter(xs, [i] * len(xs), s=14, color=INK, zorder=3, alpha=0.6)
    ax.invert_yaxis()
    ax.set_xlim(0, max(v) * 1.25)
    ax.set_xlabel("out-of-sample R² of log variance (%)   bars: pooled, dots: each ticker")
    ax.set_title("Volatility forecasts: real, consistent skill")
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    q = [S["pooled"][m]["qlike"] for m in models]
    bars = ax.barh(models, q, color=cols, height=0.6)
    for b, x in zip(bars, q):
        ax.text(x + max(q) * 0.01, b.get_y() + b.get_height() / 2, f"{x:.3f}", va="center", fontsize=9.5, color=INK)
    ax.invert_yaxis()
    ax.set_yticklabels([])
    ax.set_xlabel("QLIKE loss (lower is better)")
    ax.set_title("Robust loss for variance forecasts")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return save(fig, "fig09_vol_leaderboard")


def fig_var(Vx):
    from scipy import stats
    W = Vx["W"]
    S = W[W.ticker == "SPY"].set_index("Date")
    sig = np.sqrt(np.exp(S["har_x"] + 0.5 * S["har_x_rv"]))
    var_norm = stats.norm.ppf(0.01) * sig
    var_fhs = S["fhs_q01"] * sig
    ret = S["ret_next"]
    exc_n, exc_f = ret < var_norm, ret < var_fhs
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(S.index, ret * 100, color="#c9d0d6", lw=0.7, label="SPY next-day return")
    ax.plot(S.index, var_norm * 100, color=COLORS["garch"], lw=1.0, label="99% VaR, normal tails")
    ax.plot(S.index, var_fhs * 100, color=COLORS["har_x"], lw=1.2, label="99% VaR, empirical tails (FHS)")
    ax.scatter(S.index[exc_n & ~exc_f], ret[exc_n & ~exc_f] * 100, facecolors="white", edgecolors=COLORS["garch"],
               s=18, zorder=3, label="breach of normal VaR only")
    ax.scatter(S.index[exc_f], ret[exc_f] * 100, color="#d03b3b", s=16, zorder=4, label="breach of both")
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
    rn, rf = Vx["var"]["SPY"]["har_x"]["0.01"], Vx["var"]["SPY"]["har_x_fhs"]["0.01"]
    note(ax, f"Target breach rate 1.00%\nNormal tails: {rn['rate']:.2%}  (Kupiec p = {rn['p']:.3f})\n"
             f"Empirical tails: {rf['rate']:.2%}  (Kupiec p = {rf['p']:.2f})", "lower left")
    ax.set_title("1-day 99% Value-at-Risk for SPY from the HAR-RV + VIX volatility forecast, 2015–2026")
    ax.legend(loc="lower right", ncol=2, fontsize=8.5)
    return save(fig, "fig10_var_backtest")
