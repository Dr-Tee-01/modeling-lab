"""The six MC312 spread models, solved with the shared core.

Every function is pure: parameters in, arrays and summary numbers out. Where a
closed form exists it is returned next to the numerical solution, so the API
(and the tests) can show the solver agrees with the theory.
"""
from __future__ import annotations

import numpy as np

from ..core import fitting, montecarlo, ode

_trapz = getattr(np, "trapezoid", None) or np.trapz

# Influenza in an English boarding school, January 1978 (BMJ): boys in bed, days 1-14.
FLU_1978 = np.array([3, 8, 26, 76, 225, 298, 258, 233, 189, 128, 68, 29, 14, 4], dtype=float)
FLU_N = 763


# --------------------------------------------------------------------- Lab 2: dosing
def dosing(model: str = "one", dose_mg: float = 100, tau_h: float = 8, half_life_h: float = 22,
           absorbed: float = 0.12, ka: float = 0.6, load: float = 1.0, mec: float = 10, mtc: float = 20,
           volume_ml: float = 3000, t_end: float = 240) -> dict:
    """Repeated oral dosing. ``model='one'``: dose enters the blood at once and is
    cleared at rate k = ln2 / t½ (exact solution). ``model='two'``: dose goes to the
    gut, moves to the blood at rate ka, and is cleared at rate k (RK4 between doses)."""
    D = dose_mg * 1000 * absorbed / volume_ml          # µg/mL added to the blood per dose
    k = np.log(2) / half_life_h
    p = np.exp(-k * tau_h)                             # fraction left when the next dose arrives
    h = 0.05
    doses = np.arange(0, t_end, tau_h)
    amount = np.where(np.arange(len(doses)) == 0, D * load, D)

    if model == "one":
        t = np.linspace(0, t_end, int(round(t_end / h)) + 1)
        lag = t[:, None] - doses[None, :]
        blood = (amount * np.exp(-k * np.clip(lag, 0, None)) * (lag >= -1e-9)).sum(axis=1)
        gut = None
    else:
        def f(_, y):
            return np.array([-ka * y[0], ka * y[0] - k * y[1]])
        dose_steps = {int(round(d / h)): a for d, a in zip(doses, amount)}

        def give(tt, y):                       # a dose lands in the gut at each dosing time
            i = int(round(tt / h))
            return y + np.array([dose_steps[i], 0.0]) if i in dose_steps else y
        t, out = ode.rk4(f, [dose_steps.get(0, 0.0), 0.0], t_end, h, events=give)
        gut, blood = out[:, 0], out[:, 1]

    last = (t >= doses[-1] - tau_h) & (t < doses[-1]) if len(doses) > 1 else t >= 0
    if model == "one":
        peak, trough = D / (1 - p), D * p / (1 - p)     # steady-state plateau, closed form
    else:
        peak, trough = float(blood[last].max()), float(blood[last].min())
    reach = np.nonzero(blood >= mec)[0]
    mid = (blood[1:] + blood[:-1]) / 2
    inside = float(((mid >= mec) & (mid <= mtc)).mean())
    status = "too_high" if peak > mtc else "too_low" if trough < mec else "in_window"
    step = max(1, len(t) // 1200)
    return {
        "t": t[::step], "blood": blood[::step], "gut": None if gut is None else gut[::step],
        "per_dose": D, "k": k, "carryover": p,
        "steady_peak": peak, "steady_trough": trough,
        "first_effective_h": float(t[reach[0]]) if len(reach) else None,
        "share_in_window": inside, "status": status,
        "loading_multiplier": 1 / (1 - p), "loading_dose_mg": dose_mg / (1 - p),
        "hours_to_90pct": float(np.log(10) / k),
        "solver": "exact" if model == "one" else "rk4",
    }


# --------------------------------------------------------------------- Lab 4: Bass
def bass(p: float = 0.03, q: float = 0.38, market: float = 100, years: float = 25) -> dict:
    """dN/dt = (p + q N/M)(M - N), solved by RK4 and checked against the closed form."""
    h = 0.05
    t, Y = ode.rk4(lambda _, y: np.array([(p + q * y[0] / market) * (market - y[0])]), [0.0], years, h)
    N = Y[:, 0]
    e = np.exp(-(p + q) * t)
    exact = market * (1 - e) / (1 + (q / p) * e)
    innov, imit = p * (market - N), q * N / market * (market - N)
    t_peak = np.log(q / p) / (p + q) if q > p else 0.0
    step = max(1, len(t) // 600)
    return {
        "t": t[::step], "adopters": N[::step], "exact": exact[::step],
        "sales_innovators": innov[::step], "sales_imitators": imit[::step],
        "peak_year": float(t_peak), "peak_sales": float(market * (p + q) ** 2 / (4 * q)) if q > p else float(p * market),
        "max_solver_error": float(np.abs(N - exact).max()),
        "share_innovators": float(_trapz(innov, t) / max(_trapz(innov + imit, t), 1e-12)),
    }


# --------------------------------------------------------------------- Labs 5-6: SIR
def sir(beta: float = 1.66, alpha: float = 0.5, vaccinate_pct: float = 0, lockdown: bool = False,
        lock_start: float = 3, lock_days: float = 10, lock_cut_pct: float = 50, days: float = 20,
        n: int = FLU_N) -> dict:
    """SIR for the 1978 school with optional daily vaccination and a lockdown window."""
    v = vaccinate_pct / 100
    cut = 1 - lock_cut_pct / 100

    def b(t):
        return beta * cut if lockdown and lock_start <= t < lock_start + lock_days else beta

    def f(t, y):
        S, I = y[0], y[1]
        inf = b(t) * S * I / n
        vac = v * S
        return np.array([-inf - vac, inf - alpha * I, alpha * I, vac, inf])

    h = 0.01
    t, Y = ode.rk4(f, [n - 1, 1, 0, 0, 1], days, h)
    I = Y[:, 1]
    model_days = ode.sample(t, Y, np.arange(1, len(FLU_1978) + 1))[:, 1]
    pk = int(I.argmax())
    step = 10
    return {
        "t": t[::step], "S": Y[::step, 0], "I": I[::step], "R": Y[::step, 2], "V": Y[::step, 3],
        "data": {"day": list(range(1, len(FLU_1978) + 1)), "in_bed": FLU_1978},
        "model_at_data_days": model_days,
        "rmse": float(np.sqrt(np.mean((model_days - FLU_1978) ** 2))),
        "r0": beta / alpha, "peak": float(I[pk]), "peak_day": float(t[pk]),
        "ever_infected": float(Y[-1, 4]), "share_infected": float(Y[-1, 4] / n),
    }


def sir_fit(n: int = FLU_N) -> dict:
    """Least-squares fit of beta and alpha to the 14 daily bed counts."""
    h, days = 0.05, np.arange(1, len(FLU_1978) + 1)

    def resid(x):
        beta, alpha = x
        t, Y = ode.rk4(lambda _, y: np.array([-beta * y[0] * y[1] / n, beta * y[0] * y[1] / n - alpha * y[1]]),
                       [n - 1, 1], float(days[-1]), h)
        return ode.sample(t, Y, days)[:, 1] - FLU_1978

    fit = fitting.least_squares(resid, [1.5, 0.5], [0.1, 0.05], [5.0, 2.0])
    beta, alpha = fit["x"]
    return {"beta": beta, "alpha": alpha, "r0": beta / alpha, "rmse": fit["rmse"],
            "evaluations": fit["evaluations"], "method": "trust-region least squares on RK4 solutions"}


# --------------------------------------------------------------------- Labs 7-8: rumour
def rumour(beta: float = 0.3, gamma: float = 0.1, alpha: float = 0.0, newcomers: float = 0.0, days: float = 100) -> dict:
    """Ignorants i, spreaders s, stiflers r (people out of 1,000 at the start).
    Spreading at beta*s*i/N; a spreader stops when meeting s or r at gamma*(s+r)/N;
    an education campaign moves ignorants to stiflers at rate alpha; newcomers join as ignorants."""
    def f(_, y):
        N = y[0] + y[1] + y[2]
        sp = beta * y[1] * y[0] / N
        sf = gamma * (y[1] + y[2]) / N * y[1]
        return np.array([newcomers - sp - alpha * y[0], sp - sf, sf + alpha * y[0], sp])

    t, Y = ode.rk4(f, [990, 10, 0, 10], days, 0.05)
    c, i0 = beta / gamma, 0.99
    R0 = 1 + c
    pure = alpha == 0 and newcomers == 0
    s_max = 1 - (1 / c) * (1 + np.log(i0) - i0 + np.log(R0))
    r_inf = 1.0
    for _ in range(500):
        r_inf = 1 - np.exp(-R0 * r_inf)
    drift = None
    if pure:   # conserved along exact trajectories when alpha = newcomers = 0
        Q = Y[:, 1] / 1000 - np.log(Y[:, 0] / 1000) / c + (1 / c + 1) * Y[:, 0] / 1000
        drift = float(np.abs(Q - Q[0]).max())
    pk = int(Y[:, 1].argmax())
    total = 1000 + newcomers * days
    step = 10
    return {
        "t": t[::step], "ignorant": Y[::step, 0], "spreader": Y[::step, 1], "stifler": Y[::step, 2],
        "peak_spreaders": float(Y[pk, 1]), "peak_day": float(t[pk]),
        "reach": float(Y[-1, 3] / total),
        "theory": {"s_max": float(s_max * 1000) if pure else None, "r_inf": float(r_inf) if pure else None, "R0": R0},
        "conserved_drift": drift,
    }


# --------------------------------------------------------------------- Lab 9: Monte Carlo
def decay(lam: float = 0.05, runs: int = 100, n0: int = 1000, t_max: int = 100, seed: int = 9) -> dict:
    """Each atom gets an exact random decay time; averages are compared with N0 e^(-lam t)."""
    g = montecarlo.rng(seed)
    times = np.maximum(1, np.ceil(-np.log(1 - g.random((runs, n0))) / lam))   # step at which each atom decays
    grid = np.arange(t_max + 1)
    srt = np.sort(times, axis=1)
    alive = n0 - np.stack([np.searchsorted(row, grid, side="right") for row in srt])   # (runs, t_max+1)
    mean, sd = alive.mean(axis=0), alive.std(axis=0)
    theory = n0 * np.exp(-lam * grid)
    half = np.nonzero(mean <= n0 / 2)[0]
    i_half = min(t_max, int(round(np.log(2) / lam)))
    return {
        "t": grid, "mean": mean, "sd": sd, "theory": theory, "runs_shown": alive[:20],
        "half_life_measured": float(half[0]) if len(half) else None, "half_life_theory": float(np.log(2) / lam),
        "sd_at_half_life": float(sd[i_half]),
        "sd_theory_at_half_life": float(np.sqrt(n0 * np.exp(-lam * i_half) * (1 - np.exp(-lam * i_half)))),
        "max_gap_to_theory": float(np.abs(mean - theory).max()),
        "first_run_decay_times": times[0].astype(int),
    }


def coins(n: int = 5000, sequences: int = 6, seed: int = 9) -> dict:
    """Running share of heads in several sequences of fair coin tosses."""
    g = montecarlo.rng(seed)
    heads = np.cumsum(g.random((sequences, n)) < 0.5, axis=1)
    k = np.arange(1, n + 1)
    idx = np.unique(np.r_[np.arange(min(n, 200)), np.linspace(0, n - 1, 800).astype(int)])
    return {"n": idx + 1, "share": heads[:, idx] / k[idx], "band": 1.96 * 0.5 / np.sqrt(k[idx]),
            "final": heads[:, -1] / n}


# --------------------------------------------------------------------- Lab 10: random walk
def walk(p: float = 0.5, steps: int = 200, walkers: int = 2000, seed: int = 10) -> dict:
    """1-D walks of +-1 steps; mean and variance against n(2p-1) and 4np(1-p)."""
    g = montecarlo.rng(seed)
    pos = np.concatenate([np.zeros((walkers, 1)), np.cumsum(np.where(g.random((walkers, steps)) < p, 1, -1), axis=1)], axis=1)
    n = np.arange(steps + 1)
    final = pos[:, -1]
    vals, counts = np.unique(final, return_counts=True)
    mu_t, var_t = steps * (2 * p - 1), 4 * steps * p * (1 - p)
    normal = None
    if var_t > 0:   # the normal approximation, spread over the lattice spacing of 2
        normal = walkers * 2 * np.exp(-((vals - mu_t) ** 2) / (2 * var_t)) / np.sqrt(2 * np.pi * var_t)
    return {
        "n": n, "mean": pos.mean(axis=0), "var": pos.var(axis=0),
        "mean_theory": n * (2 * p - 1), "var_theory": 4 * n * p * (1 - p),
        "paths": pos[:40], "final_values": vals, "final_counts": counts, "normal_expected": normal,
    }


def walk2d(walkers: int = 250, steps: int = 500, seed: int = 11) -> dict:
    """2-D lattice walks with 8 neighbour moves; mean squared distance grows as 1.5 n."""
    g = montecarlo.rng(seed)
    moves = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]])
    xy = np.concatenate([np.zeros((walkers, 1, 2)), np.cumsum(moves[g.integers(0, 8, (walkers, steps))], axis=1)], axis=1)
    r2 = (xy**2).sum(axis=2).mean(axis=0)
    return {"paths": xy[:12], "msd": r2, "msd_theory": 1.5 * np.arange(steps + 1),
            "final": xy[:, -1]}
