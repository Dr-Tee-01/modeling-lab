"""Explainers for the Spread section (MC312 labs). Figures reuse modlab.spread.models."""
from __future__ import annotations

import functools
import inspect
import math

import numpy as np

from ..core import ode
from ..spread import models as M
from .registry import explainer, figure

S_, I_, R_ = "s1", "s2", "s3"


# ===================================================================== SIR: figures
@functools.lru_cache(maxsize=64)
def _sir(beta: float, alpha: float, lockdown: bool = False, lock_start: float = 3, lock_days: float = 10,
         lock_cut: float = 50, vacc: float = 0) -> dict:
    return M.sir(beta=beta, alpha=alpha, lockdown=lockdown, lock_start=lock_start, lock_days=lock_days,
                 lock_cut_pct=lock_cut, vaccinate_pct=vacc)


@figure("sir-compartments")
def _fig_compartments(p):
    """The three boxes on a chosen day, with arrow widths set by the two flows."""
    day = p.num("day", 4, 0, 20)
    r = _sir(1.66, 0.5)
    t = np.asarray(r["t"])
    i = int(np.argmin(np.abs(t - day)))
    S, I, R = float(r["S"][i]), float(r["I"][i]), float(r["R"][i])
    n = r["population"]
    infect, recover = 1.66 * S * I / n, 0.5 * I
    w = lambda v: 1.5 + 10 * min(1.0, v / 160)
    return {
        "kind": "flow", "height": 230,
        "nodes": [
            {"id": "S", "x": 10, "y": 70, "w": 130, "h": 90, "title": "Susceptible", "value": f"{S:.0f}",
             "sub": "can catch it", "fill": S / n, "fill_color": S_},
            {"id": "I", "x": 235, "y": 70, "w": 130, "h": 90, "title": "Infected", "value": f"{I:.0f}",
             "sub": "in bed", "fill": I / n, "fill_color": I_, "hot": True},
            {"id": "R", "x": 460, "y": 70, "w": 130, "h": 90, "title": "Recovered", "value": f"{R:.0f}",
             "sub": "immune", "fill": R / n, "fill_color": R_},
        ],
        "edges": [
            {"from": "S", "to": "I", "label": f"{infect:.0f} / day", "below": "β S I / N", "width": w(infect), "flow": True, "color": I_},
            {"from": "I", "to": "R", "label": f"{recover:.0f} / day", "below": "α I", "width": w(recover), "flow": True, "color": R_},
        ],
        "texts": [{"x": 300, "y": 30, "text": f"Day {t[i]:.1f} of {n} boys", "cls": "n-title"}],
        "notes": [f"On day {t[i]:.1f}, {infect:.0f} boys a day fall ill and {recover:.0f} a day recover.",
                  "The infection arrow is widest when there are many of both susceptible and infected boys."],
    }


@figure("sir-curves")
def _fig_curves(p):
    beta, alpha = p.num("beta", 1.66, 0.2, 4), p.num("alpha", 0.5, 0.1, 1.5)
    r = _sir(beta, alpha)
    return {
        "kind": "lines", "x": r["t"], "xlabel": "Day", "ylabel": "Boys", "ymin": 0, "ymax": r["population"],
        "series": [{"name": "Susceptible", "color": S_, "y": r["S"]},
                   {"name": "Infected", "color": I_, "y": r["I"], "area": True},
                   {"name": "Recovered", "color": R_, "y": r["R"]}],
        "markers": [{"x": r["peak_day"], "label": f"peak {r['peak']:.0f}"}],
        "notes": [f"R₀ = β/α = {beta:.2f}/{alpha:.2f} = {r['r0']:.2f}.",
                  f"The peak is {r['peak']:.0f} boys on day {r['peak_day']:.1f}; {100 * r['share_infected']:.0f}% of the school is infected in the end."
                  if r["r0"] > 1 else "With R₀ below 1 each case causes less than one more: the outbreak dies out at once."],
    }


@figure("sir-rt")
def _fig_rt(p):
    """Effective reproduction number R_t = R0 * S/N against the infected curve."""
    r = _sir(1.66, 0.5)
    S, n = np.asarray(r["S"]), r["population"]
    rt = r["r0"] * S / n
    cross = float(np.asarray(r["t"])[np.argmax(rt < 1)])
    return {
        "kind": "lines", "x": r["t"], "xlabel": "Day", "ylabel": "Reproduction number", "ymin": 0,
        "series": [{"name": "Rₜ = R₀ · S/N", "color": "accent", "y": rt},
                   {"name": "Infected (scaled to fit)", "color": I_, "y": np.asarray(r["I"]) / r["peak"] * r["r0"], "dash": True, "area": True}],
        "markers": [{"x": cross, "label": "Rₜ = 1"}],
        "notes": [f"Rₜ falls through 1 on day {cross:.1f}, exactly when the infected curve peaks (day {r['peak_day']:.1f}).",
                  f"At that moment {S[np.argmax(rt < 1)]:.0f} boys ({100 / r['r0']:.0f}% of the school, 1/R₀) are still susceptible."],
    }


def _final_size(r0: float) -> float:
    """Solve z = 1 - exp(-R0 z) for the share eventually infected (0 when R0 <= 1)."""
    if r0 <= 1:
        return 0.0
    z = 0.9
    for _ in range(100):
        z = 1 - np.exp(-r0 * z)
    return float(z)


@figure("sir-final-size")
def _fig_final(p):
    r0s = np.linspace(0.5, 6, 111)
    z = [_final_size(v) for v in r0s]
    school = _sir(1.66, 0.5)
    herd = [max(0.0, 1 - 1 / v) for v in r0s]
    return {
        "kind": "lines", "x": r0s, "xlabel": "R₀", "ylabel": "Share of the population", "ymin": 0, "ymax": 1,
        "series": [{"name": "Ever infected (final size)", "color": I_, "y": z, "area": True},
                   {"name": "Herd-immunity threshold 1 − 1/R₀", "color": R_, "y": herd, "dash": True}],
        "points": [{"x": school["r0"], "y": school["share_infected"], "color": "accent", "label": "1978 school"}],
        "markers": [{"x": 1, "label": "R₀ = 1"}],
        "notes": [f"The school's R₀ of {school['r0']:.2f} predicts {100 * _final_size(school['r0']):.0f}% infected; "
                  f"the simulation gives {100 * school['share_infected']:.0f}%.",
                  "Final size always overshoots the herd-immunity threshold: infections keep happening after the peak."],
    }


@figure("sir-fit")
def _fig_fit(p):
    fit = M.sir_fit()
    r = _sir(round(fit["beta"], 4), round(fit["alpha"], 4))
    guess = _sir(1.0, 0.5)
    days = list(range(1, len(M.FLU_1978) + 1))
    return {
        "kind": "lines", "x": r["t"], "xlabel": "Day", "ylabel": "Boys in bed", "ymin": 0,
        "series": [{"name": "Best fit", "color": I_, "y": r["I"]},
                   {"name": "First guess β = 1.0", "color": "neutral", "y": guess["I"], "dash": True}],
        "points": [{"x": d, "y": float(v), "color": "accent"} for d, v in zip(days, M.FLU_1978)],
        "notes": [f"The fit finds β = {fit['beta']:.2f} and α = {fit['alpha']:.2f} (R₀ = {fit['r0']:.2f}) "
                  f"after {fit['evaluations']} model solves.",
                  f"Typical miss: {fit['rmse']:.0f} boys per day (root-mean-square error over {len(days)} days)."],
    }


@functools.lru_cache(maxsize=1)
def _solver_errors():
    beta, alpha, n = 1.66, 0.5, M.FLU_N

    def f(_, y):
        inf = beta * y[0] * y[1] / n
        return np.array([-inf, inf - alpha * y[1]])

    y0, T = [n - 1.0, 1.0], 8.0
    _, ref = ode.rk4(f, y0, T, 0.001)
    exact = ref[-1, 1]
    hs = [0.8, 0.4, 0.2, 0.1, 0.05, 0.025]
    eul, rk = [], []
    for h in hs:
        y = np.array(y0)
        for k in range(int(round(T / h))):
            y = y + h * f(k * h, y)
        eul.append(abs(y[1] - exact))
        rk.append(abs(ode.rk4(f, y0, T, h)[1][-1, 1] - exact))
    return hs, eul, rk, exact


@figure("sir-solver")
def _fig_solver(p):
    hs, eul, rk, exact = _solver_errors()
    return {
        "kind": "lines", "x": hs, "xlog": True, "ylog": True, "xlabel": "Step size h (days)", "ylabel": "Error in I on day 8 (boys)",
        "series": [{"name": "Euler (first order)", "color": "warn", "y": eul},
                   {"name": "RK4 (fourth order)", "color": "accent", "y": rk}],
        "notes": [f"Halving h cuts Euler's error about {eul[-2] / eul[-1]:.1f}× but RK4's about {rk[-3] / rk[-2]:.0f}× (theory: 2× and 16×).",
                  f"With h = 0.1 day, RK4 is off by {rk[3]:.2g} boys; Euler by {eul[3]:.0f}."],
    }


@figure("sir-lockdown")
def _fig_lockdown(p):
    start, days, cut = p.num("lock_start", 3, 0, 12), p.num("lock_days", 6, 1, 15), p.num("lock_cut", 60, 0, 95)
    base, ld = _sir(1.66, 0.5), _sir(1.66, 0.5, True, start, days, cut)
    saved = base["ever_infected"] - ld["ever_infected"]
    return {
        "kind": "lines", "x": base["t"], "xlabel": "Day", "ylabel": "Boys in bed", "ymin": 0,
        "series": [{"name": "No lockdown", "color": "neutral", "y": base["I"], "dash": True},
                   {"name": "With lockdown", "color": I_, "y": ld["I"], "area": True}],
        "bands": [{"x0": start, "x1": start + days, "label": "lockdown"}],
        "notes": [f"Peak {base['peak']:.0f} → {ld['peak']:.0f} boys; {max(0, saved):.0f} fewer boys ever ill.",
                  "The peak now comes after the lockdown ends: with most boys still susceptible, Rₜ jumps back above 1 and the outbreak resumes."
                  if ld["peak_day"] > start + days else
                  "The peak comes before or during the lockdown, so the cut in contacts shortens the tail of the outbreak."],
    }


# ===================================================================== SIR: text
_FIT = M.sir_fit()

explainer("spread/sir", {
    "title": "How an epidemic rises and falls: the SIR model",
    "kicker": "Modeling Lab · MC312 Labs 5–6",
    "summary": f"In {M.FLU['month']} {M.FLU['year']} flu reached {M.FLU['place']} of {M.FLU_N} pupils, starting from a single case. "
               f"Within a week {int(M.FLU_1978.max())} were in bed on the same morning. The SIR model explains that outbreak "
               "with two numbers, and shows why every epidemic must peak, how big it gets, and what a lockdown can and cannot do.",
    "sections": [
        {"id": "story", "title": "The outbreak", "blocks": [
            {"type": "p", "text": "Doctors at the school counted how many boys were <b>in bed</b> each morning for two weeks. "
                                  f"The numbers climbed from {int(M.FLU_1978[0])} to {int(M.FLU_1978.max())} and then fell almost as quickly. "
                                  "The model treats it as an outbreak left to run its course: no closures and no vaccination."},
            {"type": "cards", "items": [
                {"title": "One closed population", "text": f"{M.FLU_N} boys, nobody arriving or leaving: the model's simplest setting."},
                {"title": "One infectious boy", "text": "The outbreak starts from a single case on day 0."},
                {"title": "Daily counts", "text": f"{len(M.FLU_1978)} mornings of data, enough to estimate how contagious the flu was."},
            ]},
            {"type": "p", "text": "The question the model answers: <i>given how fast the flu spreads and how long boys stay ill, "
                                  "what does the whole outbreak look like?</i>"},
        ]},
        {"id": "boxes", "title": "Three boxes, two arrows", "blocks": [
            {"type": "p", "text": "Every boy is in exactly one of three groups. <b>Susceptible</b> boys (S) can catch the flu, "
                                  "<b>infected</b> boys (I) have it and pass it on, and <b>recovered</b> boys (R) are immune. "
                                  "People only move one way: from S to I when they catch it, from I to R when they get better."},
            {"type": "figure", "figure": "sir-compartments", "title": "The school on one day",
             "caption": "Move the day. The boxes fill with each group's share of the school; the arrows carry today's flows.",
             "controls": [{"k": "day", "label": "Day", "min": 0, "max": 20, "step": 0.5, "value": 4}]},
            {"type": "callout", "tone": "info", "title": "Why the infection arrow changes width",
             "text": "New infections need a meeting between a susceptible boy and an infected one. Early on there are almost no "
                     "infected boys; late on there are almost no susceptible ones. The arrow is widest in between."},
        ]},
        {"id": "equations", "title": "The equations, term by term", "blocks": [
            {"type": "eq", "text": "dS/dt = − β S I / N\ndI/dt =   β S I / N − α I\ndR/dt =   α I",
             "where": [["S, I, R", "number of boys in each group (S + I + R = N)"],
                       ["N", f"the whole school, {M.FLU_N} boys"],
                       ["β", "contacts per day that would pass the flu on, if the other boy were susceptible"],
                       ["S/N", "the chance that a contact is with a susceptible boy"],
                       ["α", "recovery rate per day; 1/α is the average number of days ill"]]},
            {"type": "steps", "title": "One day of bookkeeping", "items": [
                "Each infected boy makes β flu-passing contacts a day.",
                "A share S/N of those contacts are with boys who can still catch it, so I infected boys cause β·I·S/N new cases.",
                "Those boys leave S and enter I: the same term appears with a minus sign in dS/dt and a plus sign in dI/dt.",
                "Meanwhile a share α of the infected recover today and move to R.",
                "Add the three equations: the total change is zero, so nobody is lost or created.",
            ]},
            {"type": "figure", "figure": "sir-curves", "title": "The whole outbreak",
             "caption": "Change β and α. The infected curve rises, peaks and falls; recovered boys pile up as an S-shaped curve.",
             "controls": [{"k": "beta", "label": "β, contacts per day", "min": 0.2, "max": 4, "step": 0.02, "value": 1.66},
                          {"k": "alpha", "label": "α, recoveries per day", "min": 0.1, "max": 1.5, "step": 0.01, "value": 0.5}]},
        ]},
        {"id": "r0", "title": "R₀ and why every outbreak peaks", "blocks": [
            {"type": "p", "text": "At the start nearly everyone is susceptible, so one infected boy causes β new cases a day for "
                                  "1/α days: <b>R₀ = β/α</b> new cases in total. This is the <b>basic reproduction number</b>. "
                                  "If R₀ is above 1 the outbreak grows; below 1 it fizzles out."},
            {"type": "p", "text": "As the outbreak runs, fewer contacts land on susceptible boys. The <b>effective</b> reproduction "
                                  "number is Rₜ = R₀ · S/N. The moment it falls below 1, each case is replaced by less than one "
                                  "new case and the number in bed starts to fall: that moment <i>is</i> the peak."},
            {"type": "figure", "figure": "sir-rt", "title": "Rₜ crosses 1 at the peak",
             "caption": "Rₜ (solid) shrinks as susceptible boys run out. The dashed curve is the infected count, scaled to fit."},
            {"type": "callout", "tone": "tip", "title": "Herd immunity",
             "text": "Setting Rₜ = 1 gives S/N = 1/R₀. Once more than 1 − 1/R₀ of a population is immune, an outbreak cannot grow. "
                     f"For the school (fitted R₀ ≈ {_FIT['r0']:.1f}) that is about {100 * (1 - 1 / _FIT['r0']):.0f}%."},
            {"type": "quiz", "q": "If β doubles and α doubles too, what happens to R₀?",
             "options": ["It doubles", "It stays the same", "It halves"], "answer": 1,
             "why": "R₀ = β/α, so doubling both leaves it unchanged. The outbreak has the same shape and size; it just runs twice as fast."},
        ]},
        {"id": "size", "title": "How big does it get?", "blocks": [
            {"type": "p", "text": "Dividing dS/dt by dR/dt removes time and gives a relation between the start and the end of the "
                                  "outbreak. If z is the share of the population ever infected:"},
            {"type": "eq", "text": "z = 1 − e^(−R₀ z)", "where": [["z", "final share ever infected"], ["R₀", "basic reproduction number"]]},
            {"type": "figure", "figure": "sir-final-size", "title": "Final size against R₀",
             "caption": "The equation has a non-zero solution only when R₀ > 1, and the share infected rises steeply just above it."},
            {"type": "p", "text": "The final size is always larger than the herd-immunity threshold. After the peak there are still "
                                  "many infected boys, and they keep infecting others while the curve comes down: epidemiologists "
                                  "call this <b>overshoot</b>."},
        ]},
        {"id": "fit", "title": "Fitting β and α to the 1978 counts", "blocks": [
            {"type": "p", "text": "The model has two unknowns. The server finds the pair that makes the infected curve pass as close "
                                  "as possible to the 14 daily counts, using <b>least squares</b>."},
            {"type": "steps", "items": [
                "Guess β and α.",
                "Solve the model with RK4 and read off I on days 1 to 14.",
                "Square the 14 differences from the data and add them up.",
                "Adjust β and α in the direction that lowers that total (a trust-region step), and repeat until it stops falling.",
            ]},
            {"type": "figure", "figure": "sir-fit", "title": "Best fit against the data",
             "caption": "Dots are the doctors' counts. A first guess (dashed) is far too slow; the fitted curve passes through the data."},
            {"type": "callout", "tone": "warn", "title": "What the fit assumes",
             "text": "Everyone mixes equally, everyone is ill for the same average time, and being ill means being in bed. "
                     "Real schools have dormitories and friendship groups, so the fitted β is an average over all of them."},
            {"type": "try", "label": "Open the model at the best fit", "params": {"beta": round(_FIT["beta"], 2), "alpha": round(_FIT["alpha"], 2)}},
        ]},
        {"id": "solver", "title": "How the server solves it", "blocks": [
            {"type": "p", "text": "The SIR equations have no formula for I(t), so the server steps forward in time. "
                                  "<b>Euler's method</b> uses the slope at the start of each step. <b>Runge–Kutta 4</b> (RK4) "
                                  "samples the slope four times per step and averages them, which is far more accurate for the same step."},
            {"type": "figure", "figure": "sir-solver", "title": "Error against step size",
             "caption": "Log–log axes: Euler's line has slope 1, RK4's slope 4. The live model uses RK4 with h = 0.01 day."},
        ]},
        {"id": "control", "title": "Lockdowns and vaccination", "blocks": [
            {"type": "p", "text": "A lockdown cuts β for a while; vaccination moves boys straight from S to R. Both lower Rₜ. "
                                  "Timing matters as much as strength."},
            {"type": "figure", "figure": "sir-lockdown", "title": "A lockdown window",
             "caption": "Pick when it starts, how long it lasts and how much it cuts contacts.",
             "controls": [{"k": "lock_start", "label": "Starts on day", "min": 0, "max": 12, "step": 0.5, "value": 3},
                          {"k": "lock_days", "label": "Lasts (days)", "min": 1, "max": 15, "step": 0.5, "value": 6},
                          {"k": "lock_cut", "label": "Cuts contacts by", "min": 0, "max": 95, "step": 5, "value": 60, "unit": "%"}]},
            {"type": "compare", "items": [
                {"title": "Early and long", "points": ["Rₜ held below 1 while most boys are susceptible", "The outbreak can burn out", "Fewest boys ever ill"]},
                {"title": "Late or short", "points": ["Most of the damage is already done", "The curve rebounds when it ends", "Mainly delays the peak"]},
            ]},
            {"type": "quiz", "q": "A lockdown halves contacts from day 2 to day 5, then ends. What is the most likely outcome?",
             "options": ["The outbreak is over", "The peak comes later but is almost as large", "Twice as many boys fall ill"],
             "answer": 1, "why": "Three days is not long enough for Rₜ to stay below 1 until the infected run out. When contacts "
                                 "return, most boys are still susceptible and the outbreak resumes."},
            {"type": "try", "label": "Try an early lockdown in the live model", "params": {"lockdown": True, "lock_start": 2, "lock_days": 10, "lock_cut": 60}},
        ]},
    ],
    "takeaways": [
        "SIR tracks three groups and two flows: infection (β S I / N) and recovery (α I).",
        "R₀ = β/α decides whether an outbreak grows at all.",
        "The peak comes exactly when Rₜ = R₀ · S/N falls to 1.",
        "The final size solves z = 1 − e^(−R₀ z) and always overshoots herd immunity.",
        f"Fitting the 1978 counts gives β ≈ {_FIT['beta']:.2f} and α ≈ {_FIT['alpha']:.2f}, so R₀ ≈ {_FIT['r0']:.1f} "
        f"and about {1 / _FIT['alpha']:.1f} days in bed.",
    ],
    "glossary": [
        ["Compartment", "A group of people in the same state (susceptible, infected, recovered)."],
        ["R₀", "Average number of people one case infects in a fully susceptible population."],
        ["Rₜ", "The same number at time t, after some people are immune."],
        ["Herd immunity", "Enough immunity that Rₜ < 1 even with normal contact."],
        ["Least squares", "Choosing parameters that minimise the sum of squared errors against data."],
        ["RK4", "A fourth-order Runge–Kutta solver: four slope samples per time step."],
    ],
}, order=3)


# ===================================================================== shared helpers
def _defaults(fn) -> dict:
    """A model function's default arguments: the settings each live view starts from, quoted in the prose."""
    return {k: v.default for k, v in inspect.signature(fn).parameters.items() if v.default is not inspect.Parameter.empty}


def _at(t, y, x) -> float:
    """Linear interpolation of a model curve at time x."""
    return float(np.interp(x, np.asarray(t, float), np.asarray(y, float)))


def _pct(v, d=0) -> str:
    return f"{100 * v:.{d}f}%"


def _first(t, mask):
    """First time at which a boolean condition holds, or None."""
    idx = np.nonzero(np.asarray(mask))[0]
    return float(np.asarray(t)[idx[0]]) if len(idx) else None


def _hour(v) -> str:
    return "never" if v is None else f"hour {v:.0f}"


# ===================================================================== Dosing (Lab 2): figures
DD = _defaults(M.dosing)
MEC, MTC = DD["mec"], DD["mtc"]
V_L = M.DOSING_VOLUME_ML / 1000
_STATUS = {"in_window": "settles inside the window", "too_high": "has peaks above the safe maximum",
           "too_low": "has troughs below the effective level"}


@functools.lru_cache(maxsize=512)
def _dose(model: str = "one", dose: float = DD["dose_mg"], tau: float = DD["tau_h"], half: float = DD["half_life_h"],
          ka: float = DD["ka"], load: float = DD["load"], t_end: float = M.DOSING_HOURS) -> dict:
    return M.dosing(model=model, dose_mg=dose, tau_h=tau, half_life_h=half, absorbed=DD["absorbed"], ka=ka,
                    load=load, mec=MEC, mtc=MTC, t_end=t_end)


def _window(n: int) -> list[dict]:
    return [{"name": f"MTC, safe maximum ({MTC:g} µg/mL)", "color": "bad", "y": [MTC] * n, "dash": True, "thin": True},
            {"name": f"MEC, minimum effective ({MEC:g} µg/mL)", "color": "warn", "y": [MEC] * n, "dash": True, "thin": True}]


@figure("dosing-single")
def _fig_dose_single(p):
    """One dose on its own: an exponential decay, halving every half-life."""
    half = p.num("half_life", DD["half_life_h"], 2, 72)
    T = float(6 * np.ceil(5.5 * half / 6))
    r = _dose(tau=T, half=half, t_end=T)            # the second dose would arrive after the horizon
    t, c, D = r["t"], r["blood"], r["per_dose"]
    y5 = _at(t, c, 5 * half)
    tail = (f"On its own the dose never reaches the minimum effective level of {MEC:g} µg/mL: the drug only works once doses overlap. "
            if D < MEC else "")
    return {
        "kind": "lines", "x": t, "xlabel": "Hours after the dose", "ylabel": "µg/mL", "ymin": 0,
        "series": [{"name": "Blood level after one dose", "color": "s1", "y": c, "area": True}],
        "markers": [{"x": n * half, "label": ("t½" if n == 1 else f"{n}t½")} for n in range(1, 6)],
        "points": [{"x": n * half, "y": _at(t, c, n * half), "color": "accent"} for n in range(1, 6)],
        "notes": [f"One {DD['dose_mg']:g} mg dose raises the level by D = {D:.1f} µg/mL. With t½ = {half:g} h the body clears "
                  f"{_pct(1 - math.exp(-r['k']), 1)} of what is left every hour.",
                  tail + f"After five half-lives ({5 * half:g} h) {y5:.2f} µg/mL is left, {_pct(y5 / D)} of the start."],
    }


@figure("dosing-flow")
def _fig_dose_flow(p):
    """The gut → blood model at a chosen hour: tank levels and the two flows."""
    hour = p.num("hour", 9, 0, 48)
    r = _dose("two")
    t = np.asarray(r["t"])
    i = int(np.argmin(np.abs(t - hour)))
    gut, blood = np.asarray(r["gut"]), np.asarray(r["blood"])
    G, B, k, ka = float(gut[i]), float(blood[i]), r["k"], DD["ka"]
    inflow, out = ka * G, k * B
    vmax = float((ka * gut).max())
    w = lambda v: 1.5 + 9 * min(1.0, v / vmax)
    return {
        "kind": "flow", "height": 230,
        "nodes": [
            {"id": "D", "x": 10, "y": 80, "w": 95, "h": 80, "title": "Dose", "value": f"{DD['dose_mg']:g} mg",
             "sub": f"every {DD['tau_h']:g} h"},
            {"id": "G", "x": 165, "y": 80, "w": 115, "h": 80, "title": "Gut", "value": f"{G:.2f}", "sub": "µg/mL equiv.",
             "fill": G / float(gut.max()), "fill_color": "s4"},
            {"id": "B", "x": 345, "y": 70, "w": 125, "h": 100, "title": "Blood", "value": f"{B:.1f} µg/mL",
             "sub": f"window {MEC:g}–{MTC:g}", "fill": B / (MTC * 1.3), "fill_color": "s1", "hot": True},
            {"id": "C", "x": 525, "y": 80, "w": 70, "h": 80, "title": "Cleared", "sub": f"t½ {DD['half_life_h']:g} h"},
        ],
        "edges": [
            {"from": "D", "to": "G", "label": _pct(DD["absorbed"]), "below": "absorbed", "width": 3},
            {"from": "G", "to": "B", "label": f"{inflow:.2f}", "below": "k₁ G", "width": w(inflow), "flow": True, "color": "s4"},
            {"from": "B", "to": "C", "label": f"{out:.2f}", "below": "k B", "width": w(out), "flow": True, "color": "s1"},
        ],
        "texts": [{"x": 300, "y": 30, "text": f"Hour {t[i]:.1f} · flows in µg/mL per hour", "cls": "n-title"}],
        "notes": [f"At hour {t[i]:.1f} the gut passes {inflow:.2f} µg/mL an hour into the blood and the body clears {out:.2f}, "
                  f"so the blood level is {'rising' if inflow > out else 'falling'}.",
                  f"Over the {r['hours']:.0f} h run the level settles between {r['steady_trough']:.1f} and {r['steady_peak']:.1f} µg/mL, "
                  "where the drug cleared in one interval equals one dose."],
    }


@figure("dosing-buildup")
def _fig_dose_buildup(p):
    dose, tau = p.num("dose", DD["dose_mg"], 25, 400), p.num("tau", DD["tau_h"], 2, 24)
    r = _dose(dose=dose, tau=tau)
    n = len(r["t"])
    status = {"in_window": f"That plateau sits inside the window, and the level spends {_pct(r['share_in_window'])} of the "
                           f"{r['horizon_days']:.0f} days between {MEC:g} and {MTC:g} µg/mL.",
              "too_high": f"The plateau peak is above the safe maximum of {MTC:g} µg/mL: lower the dose or give it in smaller, more frequent parts.",
              "too_low": f"The trough is below the minimum effective level of {MEC:g} µg/mL: raise the dose or shorten the interval."}[r["status"]]
    return {
        "kind": "lines", "x": r["t"], "xlabel": "Hours", "ylabel": "µg/mL", "ymin": 0,
        "series": [{"name": "Blood level", "color": "s1", "y": r["blood"], "area": True}] + _window(n),
        "markers": [{"x": r["hours_to_90pct"], "label": "90% of plateau"}],
        "notes": [f"Each dose has decayed to {_pct(r['carryover'])} of itself when the next arrives (p = e^(−kτ)), "
                  f"so the level settles between {r['steady_trough']:.1f} and {r['steady_peak']:.1f} µg/mL.", status],
    }


@figure("dosing-geometric")
def _fig_dose_geometric(p):
    """Peak just after each dose, split into the new dose and what is left of earlier ones."""
    r = _dose()
    t, c, D, pr = np.asarray(r["t"]), np.asarray(r["blood"]), r["per_dose"], r["carryover"]
    n = 12
    peaks = [float(c[int(np.argmin(np.abs(t - j * DD["tau_h"])))]) for j in range(n)]
    left = [pk - D for pk in peaks]
    return {
        "kind": "bars", "labels": [str(j + 1) for j in range(n)], "xlabel": "Dose number", "ylabel": "µg/mL just after the dose",
        "series": [{"name": "Left from earlier doses", "color": "s3", "values": left},
                   {"name": "The new dose D", "color": "s1", "values": [D] * n}],
        "notes": [f"The peak climbs {peaks[0]:.1f}, {peaks[1]:.1f}, {peaks[2]:.1f}, {peaks[3]:.1f} … µg/mL. Each rise is "
                  f"p = {pr:.2f} times the one before, so the climb slows and stops.",
                  f"After dose {n}, {left[-1]:.1f} µg/mL is left over from earlier doses, close to its limit "
                  f"D·p/(1 − p) = {r['steady_trough']:.1f}."],
    }


@figure("dosing-loading")
def _fig_dose_loading(p):
    load = p.num("load", round(float(_dose()["loading_multiplier"]), 1), 1, 8)
    T = 96.0
    base, ld = _dose(t_end=T), _dose(load=load, t_end=T)
    n, tau, mult = len(base["t"]), DD["tau_h"], float(base["loading_multiplier"])
    t = np.asarray(ld["t"])
    peak1 = float(np.asarray(ld["blood"])[t < tau].max())
    if peak1 > MTC:
        second = (f"Too large: the first peak reaches {peak1:.1f} µg/mL, above the safe maximum of {MTC:g}. "
                  f"The ideal multiplier is 1/(1 − p) = {mult:.2f}.")
    elif load < mult - 0.15:
        second = f"Below the ideal multiplier 1/(1 − p) = {mult:.2f}, the level still has to climb towards the plateau."
    else:
        second = f"Close to the ideal multiplier 1/(1 − p) = {mult:.2f}: the first dose fills the body to the plateau peak at once."
    return {
        "kind": "lines", "x": base["t"], "xlabel": "Hours", "ylabel": "µg/mL", "ymin": 0,
        "series": [{"name": "Ordinary first dose", "color": "neutral", "y": base["blood"], "dash": True},
                   {"name": f"First dose ×{load:.1f}", "color": "s1", "y": ld["blood"], "area": True}] + _window(n),
        "notes": [(f"With an ordinary first dose the drug is effective from {_hour(base['first_effective_h'])}."
                   if load < 1.05 else
                   f"With a first dose {load:.1f}× as large ({load * DD['dose_mg']:.0f} mg) the drug is effective from "
                   f"{_hour(ld['first_effective_h'])}; with an ordinary first dose, from {_hour(base['first_effective_h'])}."), second],
    }


_MAP_DOSES = np.arange(25, 401, 25)
_MAP_TAUS = np.arange(2, 25, 2)


@figure("dosing-map")
def _fig_dose_map(p):
    """Every dose × interval pair on a grid, coloured by where its plateau sits."""
    half = p.num("half_life", DD["half_life_h"], 4, 72)
    code = {"too_low": 1, "in_window": 2, "too_high": 3}
    cells, good, rows_ok = [], 0, 0
    mark = (int(np.argmin(np.abs(_MAP_TAUS - DD["tau_h"]))), int(np.argmin(np.abs(_MAP_DOSES - DD["dose_mg"]))))
    for i, tau in enumerate(_MAP_TAUS):
        row = [_dose(dose=float(d), tau=float(tau), half=half, t_end=48.0)["status"] for d in _MAP_DOSES]
        good += row.count("in_window")
        rows_ok += "in_window" in row
        cells += [4 if (i, j) == mark else code[s] for j, s in enumerate(row)]
    here = _dose(half=half)["status"]
    tau_max = half * math.log2(MTC / MEC)
    step = int(_MAP_DOSES[1] - _MAP_DOSES[0])
    rows = ((f"No dose works with an interval longer than t½ · log₂(MTC/MEC) = {tau_max:.1f} h: below that row every regimen is "
             f"too low or too high. {rows_ok} of {len(_MAP_TAUS)} rows have at least one dose that works.")
            if tau_max < _MAP_TAUS[-1] else
            (f"Every interval on the grid is below t½ · log₂(MTC/MEC) = {tau_max:.1f} h, so each could work with the right dose; "
             f"{rows_ok} of {len(_MAP_TAUS)} rows have one among the {step} mg steps shown."))
    return {
        "kind": "grid", "cols": len(_MAP_DOSES), "rows": len(_MAP_TAUS), "cells": cells, "palette": "cat",
        "colors": ["neutral", "warn", "good", "bad", "accent"], "height": 330,
        "xlabel": "Dose (mg)", "ylabel": "Interval (h)",
        "xticks": [[j, f"{d:g}"] for j, d in enumerate(_MAP_DOSES) if j % 3 == 1],
        "yticks": [[i, f"{t:g}"] for i, t in enumerate(_MAP_TAUS) if i % 2 == 0],
        "legend": [{"name": "Plateau too low", "color": "warn"}, {"name": "Inside the window", "color": "good"},
                   {"name": "Too high", "color": "bad"}, {"name": "Default regimen", "color": "accent"}],
        "notes": [f"With t½ = {half:g} h, {good} of {len(cells)} regimens on the grid settle inside the window. The highlighted square is the "
                  f"default, {DD['dose_mg']:g} mg every {DD['tau_h']:g} h, which {_STATUS[here]}.", rows],
    }


@figure("dosing-gut")
def _fig_dose_gut(p):
    ka = p.num("ka", DD["ka"], 0.05, 3)
    one, two = _dose(), _dose("two", ka=ka)
    t = np.asarray(one["t"])
    y2 = np.interp(t, np.asarray(two["t"]), np.asarray(two["blood"]))
    return {
        "kind": "lines", "x": t, "xlabel": "Hours", "ylabel": "µg/mL", "ymin": 0,
        "series": [{"name": "Straight into the blood", "color": "neutral", "y": one["blood"], "dash": True},
                   {"name": f"Through the gut, k₁ = {ka:.2f}/h", "color": "s1", "y": y2, "area": True}] + _window(len(t)),
        "notes": [f"Through the gut the last interval swings between {two['steady_trough']:.1f} and {two['steady_peak']:.1f} µg/mL; "
                  f"entering the blood at once, between {one['steady_trough']:.1f} and {one['steady_peak']:.1f}.",
                  f"Effective from {_hour(two['first_effective_h'])} through the gut, against {_hour(one['first_effective_h'])} "
                  "when the dose enters the blood at once."],
    }


# ===================================================================== Dosing: text
_D0 = _dose()
_K0, _P0, _DPER = float(_D0["k"]), float(_D0["carryover"]), float(_D0["per_dose"])
_TAU_MAX = DD["half_life_h"] * math.log2(MTC / MEC)
_D2X = _dose(dose=2 * DD["dose_mg"])
_DAILY = DD["dose_mg"] * 24 / DD["tau_h"]          # the same daily amount, given once a day
_ONCE = _dose(dose=_DAILY, tau=24.0)
_ONCE_OK = _TAU_MAX >= 24

explainer("spread/dosing", {
    "title": "How repeated doses build up to a steady level: drug dosing and the therapeutic window",
    "kicker": "Modeling Lab · MC312 Lab 2",
    "summary": f"A drug taken every {DD['tau_h']:g} hours has not left the body when the next dose arrives, so each dose lands on "
               "what is left of the earlier ones. The level climbs, then settles on a plateau. The model shows how dose, interval "
               "and half-life set that plateau, how long it takes to get there, and when a larger first dose gets there at once.",
    "sections": [
        {"id": "window", "title": "The dosing question", "blocks": [
            {"type": "p", "text": "A drug only helps inside a band of concentrations in the blood. Below the <b>minimum effective "
                                  "concentration</b> (MEC) it does nothing; above the <b>maximum safe concentration</b> (MTC) it "
                                  "is toxic. The band between them is the <b>therapeutic window</b>. The prescriber controls two "
                                  "things, the size of each dose and the time between doses, and the body does the rest."},
            {"type": "cards", "items": [
                {"title": "The window", "text": f"The live view starts with MEC = {MEC:g} and MTC = {MTC:g} µg/mL."},
                {"title": "The drug", "text": f"A half-life of {DD['half_life_h']:g} h, with {_pct(DD['absorbed'])} of each dose "
                                              f"reaching {V_L:g} L of blood (the view's phenytoin preset)."},
                {"title": "The regimen", "text": f"{DD['dose_mg']:g} mg every {DD['tau_h']:g} h: the two numbers a prescriber chooses."},
            ]},
            {"type": "p", "text": "Start with one dose on its own. It raises the level by a fixed amount, then the body clears it, "
                                  "half of what is left every half-life."},
            {"type": "figure", "figure": "dosing-single", "title": "One dose, decaying",
             "caption": "Move the half-life. The dots mark one, two, three, four and five half-lives; each halves the level again.",
             "controls": [{"k": "half_life", "label": "Half-life t½", "min": 2, "max": 72, "step": 1, "value": DD["half_life_h"], "unit": "h"}]},
            {"type": "p", "text": "The question the model answers: <i>if doses keep arriving before the last one has gone, where does "
                                  "the level end up, and how should dose and interval be chosen to keep it inside the window?</i>"},
        ]},
        {"id": "flow", "title": "Where a dose goes", "blocks": [
            {"type": "p", "text": "In the simplest model, the <b>one-compartment</b> model, the absorbed share of each tablet enters "
                                  "the blood at once and is cleared from there. A more realistic version adds the <b>gut</b>: the "
                                  "dose waits there and passes into the blood at a rate k₁, so the level rises over a few hours "
                                  "instead of jumping."},
            {"type": "figure", "figure": "dosing-flow", "title": "Gut, blood and clearance, hour by hour",
             "caption": "Move the hour. The tanks fill with the current amounts; arrow widths carry the two flows. A new dose lands "
                        f"in the gut every {DD['tau_h']:g} h.",
             "controls": [{"k": "hour", "label": "Hour", "min": 0, "max": 48, "step": 0.5, "value": 9, "unit": "h"}]},
            {"type": "callout", "tone": "info", "title": "Why clearance catches up",
             "text": "The body clears a fixed <i>share</i> of the drug per hour, so the more there is, the faster it goes. As doses "
                     "pile up the out-flow grows until, over one interval, it removes exactly one dose. That balance is the plateau."},
        ]},
        {"id": "equations", "title": "The equations, term by term", "blocks": [
            {"type": "eq", "text": "dC/dt = − k C              between doses\nC  →  C + D               at each dose, t = 0, τ, 2τ, …\n"
                                   "k = ln 2 / t½,     D = dose × f / V",
             "where": [["C", "concentration in the blood, µg/mL"],
                       ["k", "elimination rate per hour; a share k of the drug is cleared each hour (for small k)"],
                       ["t½", "half-life: the time for C to halve when no dose arrives"],
                       ["τ", "dosing interval, hours"],
                       ["D", "rise in concentration from one dose"],
                       ["f, V", f"fraction absorbed, and the blood volume ({V_L:g} L in the model)"]]},
            {"type": "steps", "title": "One interval of bookkeeping", "items": [
                f"A dose arrives and the level jumps by D = {DD['dose_mg']:g} mg × {DD['absorbed']:g} / {V_L:g} L = {_DPER:.1f} µg/mL.",
                f"Between doses the level decays exponentially: k = ln 2 / {DD['half_life_h']:g} h = {_K0:.4f} per hour.",
                f"When the next dose is due, a share p = e^(−kτ) = e^(−{_K0:.4f} × {DD['tau_h']:g}) = {_P0:.3f} of the level is left.",
                "The new dose adds D on top of that remainder, and the cycle repeats.",
            ]},
            {"type": "p", "text": "Because the decay is proportional to the level, the model is <b>linear</b>: doses do not interact, "
                                  "and the level at any time is just the sum of what is left of every dose so far."},
        ]},
        {"id": "plateau", "title": "Why the level levels off", "blocks": [
            {"type": "p", "text": "Just after dose n the level is the new dose plus what remains of each earlier one: D, plus D·p, "
                                  "plus D·p², and so on. That is a geometric series, and because p is below 1 it has a finite sum."},
            {"type": "eq", "text": "peak after n doses  = D (1 + p + p² + … + pⁿ⁻¹) = D (1 − pⁿ) / (1 − p)\n"
                                   "plateau peak        = D / (1 − p)\nplateau trough      = D p / (1 − p)",
             "where": [["p", "e^(−kτ), the share of the level left after one interval"], ["D", "rise from one dose"]]},
            {"type": "figure", "figure": "dosing-geometric", "title": "Each peak, split in two",
             "caption": "Blue: the new dose, the same every time. Green: what is left of all earlier doses, growing towards its limit."},
            {"type": "figure", "figure": "dosing-buildup", "title": "The level over ten days",
             "caption": "Change the dose and the interval. The level climbs in a saw-tooth and settles; the dashed lines are the window.",
             "controls": [{"k": "dose", "label": "Dose", "min": 25, "max": 400, "step": 5, "value": DD["dose_mg"], "unit": "mg"},
                          {"k": "tau", "label": "Interval τ", "min": 2, "max": 24, "step": 1, "value": DD["tau_h"], "unit": "h"}]},
            {"type": "quiz", "q": f"The interval stays at {DD['tau_h']:g} h and the dose doubles. What happens to the plateau?",
             "options": ["It doubles", "It rises, but by less than double, because more drug is cleared", "It stays the same"],
             "answer": 0, "why": "The plateau peak D/(1 − p) is proportional to D, and p does not depend on the dose. "
                                 f"For this drug the peak goes from {_D0['steady_peak']:.1f} to {_D2X['steady_peak']:.1f} µg/mL."},
        ]},
        {"id": "time", "title": "How long it takes, and loading doses", "blocks": [
            {"type": "p", "text": "The gap between the current level and the plateau shrinks by the factor p every interval, which is "
                                  "the same as e^(−kt) over time t. Reaching 90% of the plateau therefore takes ln 10 / k "
                                  f"= {_D0['hours_to_90pct']:.0f} h, whatever the dose and the interval."},
            {"type": "callout", "tone": "tip", "title": "Rule of thumb",
             "text": f"ln 10 / ln 2 = {math.log2(10):.2f}, so reaching 90% of the plateau takes about {math.log2(10):.1f} half-lives. "
                     f"For a half-life of {DD['half_life_h']:g} h that is {_D0['hours_to_90pct'] / 24:.1f} days of ineffective treatment."},
            {"type": "p", "text": "A <b>loading dose</b> skips the wait. If the first dose is 1/(1 − p) times the usual one, it puts "
                                  "the level straight onto the plateau peak, and ordinary doses then keep it there. "
                                  f"Here that multiplier is {_D0['loading_multiplier']:.2f}: a first dose of {_D0['loading_dose_mg']:.0f} mg."},
            {"type": "figure", "figure": "dosing-loading", "title": "With and without a loading dose",
             "caption": "Change the size of the first dose. Too small and the level still climbs; too large and it overshoots the window.",
             "controls": [{"k": "load", "label": "First dose ×", "min": 1, "max": 8, "step": 0.1,
                           "value": round(float(_D0["loading_multiplier"]), 1), "unit": "×"}]},
            {"type": "try", "label": "Open the model with the loading dose",
             "params": {"model": "one", "load": round(float(_D0["loading_multiplier"]), 1)}},
        ]},
        {"id": "map", "title": "Which regimens work", "blocks": [
            {"type": "p", "text": "The window sets two conditions: the plateau peak D/(1 − p) must stay below MTC, and the trough "
                                  "D p/(1 − p) must stay above MEC. Dividing one by the other removes the dose: the ratio of peak to "
                                  "trough is 1/p = e^(kτ), and it must not exceed MTC/MEC."},
            {"type": "eq", "text": "peak / trough = e^(kτ) ≤ MTC / MEC\nτ ≤ t½ · log₂(MTC / MEC)",
             "where": [["τ", "dosing interval"], ["t½", "half-life"], ["MTC / MEC", "width of the window as a ratio"]]},
            {"type": "p", "text": f"For this drug and window the longest workable interval is {_TAU_MAX:.1f} h. Any longer and the level "
                                  "swings too far between doses: a dose large enough to keep the trough effective makes the peak toxic."},
            {"type": "figure", "figure": "dosing-map", "title": "Every regimen at once",
             "caption": "Each square is one regimen: a dose (across) given every so many hours (down). A plateau that is too "
                        "low has troughs below MEC; one that is too high has peaks above MTC. Change the half-life.",
             "controls": [{"k": "half_life", "label": "Half-life t½", "min": 4, "max": 72, "step": 1, "value": DD["half_life_h"], "unit": "h"}]},
            {"type": "compare", "items": [
                {"title": "Large, infrequent doses", "points": ["Easy to keep to", "Big swing between peak and trough",
                                                                "Fail once τ exceeds t½ · log₂(MTC/MEC)"]},
                {"title": "Small, frequent doses", "points": ["Smooth level close to the average", "Same plateau for the same daily amount",
                                                              "Harder for patients to keep to"]},
            ]},
            {"type": "quiz", "q": f"Could any once-a-day dose keep this drug (t½ = {DD['half_life_h']:g} h) inside the window?",
             "options": ["Yes, if the dose is chosen carefully", "No: with τ = 24 h the swing e^(kτ) is wider than the window",
                         "Only with a loading dose"],
             "answer": 0 if _ONCE_OK else 1,
             "why": f"The swing ratio is e^(k · 24) = {math.exp(_K0 * 24):.2f} against a window ratio MTC/MEC = {MTC / MEC:g}. "
                    + ("It fits, so some dose works." if _ONCE_OK else
                       f"It does not fit, whatever the dose. The same daily amount once a day ({_DAILY:.0f} mg) "
                       f"{_STATUS[_ONCE['status']]}.")},
            {"type": "try", "label": "Try the same daily amount once a day", "params": {"model": "one", "dose": round(_DAILY), "tau": 24, "load": 1},
             "note": f"{_DAILY:.0f} mg every 24 h: the regimen {_STATUS[_ONCE['status']]}."},
        ]},
        {"id": "gut", "title": "Absorption through the gut", "blocks": [
            {"type": "p", "text": "With the gut in the model each dose first lands in the gut (G) and moves into the blood (B) at "
                                  "rate k₁. The blood now gains drug gradually and loses it as before."},
            {"type": "eq", "text": "dG/dt = − k₁ G\ndB/dt =   k₁ G − k B",
             "where": [["G", "drug waiting in the gut (as a blood concentration equivalent)"], ["B", "concentration in the blood"],
                       ["k₁", "rate at which the gut passes the drug on, per hour"], ["k", "elimination rate, as before"]]},
            {"type": "figure", "figure": "dosing-gut", "title": "Gut absorption smooths the saw-tooth",
             "caption": "Change k₁. Slow absorption lowers and delays each peak; fast absorption approaches the one-compartment curve.",
             "controls": [{"k": "ka", "label": "Gut → blood rate k₁", "min": 0.05, "max": 3, "step": 0.05, "value": DD["ka"], "unit": "/h"}]},
            {"type": "p", "text": "The average level on the plateau is the same either way: every dose is eventually absorbed and "
                                  "cleared at the same rate k. What the gut changes is the <b>shape</b> within each interval, "
                                  "and so whether the peaks and troughs stay inside the window."},
            {"type": "try", "label": "Open the gut → blood model", "params": {"model": "two", "ka": DD["ka"], "load": 1}},
        ]},
        {"id": "solver", "title": "How the server computes it", "blocks": [
            {"type": "p", "text": "The one-compartment model has an exact solution: the level is a sum of decaying exponentials, "
                                  "C(t) = Σ D e^(−k(t − nτ)) over the doses given so far. The server evaluates that sum on a grid "
                                  f"every {M.DOSING_H:g} h over {M.DOSING_HOURS} h, and takes the plateau peak and trough from the "
                                  "closed forms above."},
            {"type": "p", "text": "The gut model has no such simple sum, so the server integrates it with fourth-order Runge–Kutta "
                                  f"(RK4) at the same {M.DOSING_H:g} h step, and adds each dose to the gut as an instant jump at the "
                                  "end of the step where it falls. Its peak and trough are read from the last dosing interval."},
            {"type": "callout", "tone": "warn", "title": "What the model leaves out",
             "text": "Clearance is assumed proportional to the level (a constant k). For some drugs clearance saturates at higher "
                     "levels, and then the level rises faster than the model predicts when the dose goes up. The model also assumes "
                     "one well-mixed blood volume, the same share absorbed every time, every dose taken on time, and one window "
                     "for every patient."},
        ]},
    ],
    "takeaways": [
        "Each dose decays to p = e^(−kτ) of itself by the next one, so levels build up to a plateau instead of growing forever.",
        "The plateau peak is D/(1 − p) and the trough D·p/(1 − p); both are proportional to the dose.",
        f"Reaching 90% of the plateau takes about {math.log2(10):.1f} half-lives, whatever the dose.",
        "A first dose 1/(1 − p) times the usual one starts the patient on the plateau.",
        f"Peak over trough is e^(kτ), so the interval may not exceed t½ · log₂(MTC/MEC): {_TAU_MAX:.1f} h for this drug.",
    ],
    "glossary": [
        ["Therapeutic window", "The range of blood levels between the minimum effective (MEC) and maximum safe (MTC) concentrations."],
        ["Half-life", "Time for the body to clear half of the drug when no more is given."],
        ["Elimination rate k", "ln 2 divided by the half-life; the proportional rate at which the drug is cleared."],
        ["Plateau (steady state)", "The repeating pattern the level settles into, when one interval clears exactly one dose."],
        ["Loading dose", "A larger first dose that reaches the plateau at once."],
        ["Compartment", "A well-mixed part of the body treated as one tank, such as the gut or the blood."],
        ["Geometric series", "A sum 1 + p + p² + …; for p below 1 it approaches 1/(1 − p)."],
    ],
}, order=1)


# ===================================================================== Bass (Lab 4): figures
BD = _defaults(M.bass)
_BASS_CASES = (("Typical product", BD["p"], BD["q"], "s1"), ("Word-of-mouth hit", 0.01, 0.7, "s3"),
               ("Advertising-led", 0.1, 0.1, "s2"))          # the live view's three presets


@functools.lru_cache(maxsize=512)
def _bass(p: float = BD["p"], q: float = BD["q"], years: float = BD["years"]) -> dict:
    return M.bass(p=p, q=q, market=BD["market"], years=years)


@figure("bass-flow")
def _fig_bass_flow(p):
    """The market at a chosen year: who has bought, and the two flows of new buyers."""
    year = p.num("year", 3, 0, BD["years"])
    r = _bass()
    t = np.asarray(r["t"])
    i = int(np.argmin(np.abs(t - year)))
    m = BD["market"]
    N, a, w_ = float(r["adopters"][i]), float(r["sales_innovators"][i]), float(r["sales_imitators"][i])
    vmax = float(max(np.max(r["sales_innovators"]), np.max(r["sales_imitators"])))
    w = lambda v: 1.5 + 9 * min(1.0, v / vmax)
    return {
        "kind": "flow", "height": 240,
        "nodes": [
            {"id": "P", "x": 20, "y": 80, "w": 150, "h": 84, "title": "Not yet bought", "value": f"{m - N:.1f}M",
             "sub": "M − N", "fill": (m - N) / m, "fill_color": "s1"},
            {"id": "A", "x": 430, "y": 80, "w": 150, "h": 84, "title": "Adopters", "value": f"{N:.1f}M",
             "sub": "N", "fill": N / m, "fill_color": "s3", "hot": True},
        ],
        "edges": [
            {"from": "P", "to": "A", "label": f"advertising  {a:.2f}M / year", "width": w(a), "flow": True, "color": "s2", "curve": [0, -64]},
            {"from": "P", "to": "A", "label": f"word of mouth  {w_:.2f}M / year", "width": w(w_), "flow": True, "color": "s3", "curve": [0, 64]},
        ],
        "texts": [{"x": 300, "y": 26, "text": f"Year {t[i]:.1f} of a {m:g}M market", "cls": "n-title"},
                  {"x": 300, "y": 226, "text": "advertising p (M − N)   ·   word of mouth q (N/M)(M − N)"}],
        "notes": [f"In year {t[i]:.1f}, {a:.2f}M people a year buy because of advertising and {w_:.2f}M through word of mouth; "
                  f"{N:.1f}M of {m:g}M own one.",
                  "Word of mouth now brings more buyers than advertising: it grows with the number of owners, while advertising "
                  "only shrinks as the pool of non-owners empties." if w_ > a else
                  "Advertising still brings more buyers than word of mouth: word of mouth needs owners to talk, and there are few so far."],
    }


@figure("bass-sales")
def _fig_bass_sales(p):
    pp, q = p.num("p", BD["p"], 0.001, 0.15), p.num("q", BD["q"], 0, 1)
    r = _bass(pp, q)
    innov, imit = np.asarray(r["sales_innovators"]), np.asarray(r["sales_imitators"])
    peak = (f"Sales peak in year {r['peak_year']:.1f} at {r['peak_sales']:.1f}M a year: t* = ln(q/p)/(p + q) = "
            f"ln({q / pp:.1f}) / {pp + q:.3f}." if r["word_of_mouth_takes_over"] else
            f"With q ≤ p word of mouth never catches up: sales are highest at launch ({r['peak_sales']:.1f}M a year) and fall from there.")
    return {
        "kind": "lines", "x": r["t"], "xlabel": "Year", "ylabel": "Sales (M / year)", "ymin": 0,
        "series": [{"name": "All sales", "color": "s1", "y": innov + imit, "area": True},
                   {"name": "Because of advertising", "color": "s2", "y": innov},
                   {"name": "Through word of mouth", "color": "s3", "y": imit}],
        "markers": [{"x": r["peak_year"], "label": "peak"}] if r["word_of_mouth_takes_over"] else [],
        "notes": [peak, (f"Over {r['years']:.0f} years all buyers came through advertising" if r["share_innovators"] > 0.995 else
                         f"Over {r['years']:.0f} years {_pct(r['share_innovators'])} of buyers came through advertising and the rest "
                         "through word of mouth") + f"; {_pct(r['final_share'])} of the market has bought."],
    }


@figure("bass-adoption")
def _fig_bass_adoption(p):
    """Share of the market that owns one, for four strengths of word of mouth."""
    pp = p.num("p", BD["p"], 0.005, 0.15)
    qs = (0.0, BD["q"] / 2, BD["q"], 2 * BD["q"])
    runs = [_bass(pp, q) for q in qs]
    t = runs[0]["t"]
    halves = [_first(r["t"], np.asarray(r["adopters"]) >= BD["market"] / 2) for r in runs]
    never = f"not within {BD['years']:g} years"
    said = ", ".join(f"{never if h is None else f'{h:.1f} years'} at q = {q:.2f}" for h, q in zip(halves, qs))
    return {
        "kind": "lines", "x": t, "xlabel": "Year", "ylabel": "Share of the market that owns one", "ymin": 0, "ymax": 1,
        "series": [{"name": f"q = {q:.2f}", "color": c, "y": np.asarray(r["adopters"]) / BD["market"], "dash": q == 0}
                   for q, r, c in zip(qs, runs, ("neutral", "s1", "s2", "s3"))],
        "points": [{"x": h, "y": 0.5, "color": "accent"} for h in halves if h is not None],
        "notes": [f"Time to reach half the market with p = {pp:.3f}: {said}.",
                  f"With q = 0 the curve is M(1 − e^(−pt)), steepest at launch. With q above p it is an S, steepest when "
                  f"N/M = (q − p)/(2q): {_pct(max(0.0, (BD['q'] - pp) / (2 * BD['q'])))} for q = {BD['q']:.2f}."],
    }


@figure("bass-peak-map")
def _fig_bass_peak_map(p):
    """Peak timing against peak size, for a range of q at three values of p."""
    ps, qs = (0.01, BD["p"], 0.1), np.round(np.linspace(0.05, 1, 20), 3)
    lines, points = [], []
    for pp, c in zip(ps, ("s1", "s2", "s3")):
        rs = [(q, _bass(pp, float(q), 2.0)) for q in qs if q > pp]
        xs, ys = [r["peak_year"] for _, r in rs], [r["peak_sales"] for _, r in rs]
        lines.append({"x": xs, "y": ys, "color": c, "name": f"p = {pp:g}"})
        points += [{"x": x, "y": y, "color": c, "r": 3} for x, y in zip(xs, ys)]
    d = _bass(BD["p"], BD["q"], 2.0)
    points.append({"x": d["peak_year"], "y": d["peak_sales"], "color": "accent", "r": 6, "label": "default"})
    mid = [(float(q), _bass(BD["p"], float(q), 2.0)["peak_year"]) for q in qs if q > BD["p"]]
    q_late, t_late = max(mid, key=lambda v: v[1])
    lo, hi = _bass(ps[0], BD["q"], 2.0), _bass(ps[-1], BD["q"], 2.0)
    return {
        "kind": "scatter", "points": points, "lines": lines, "xlabel": "Year of peak sales", "ylabel": "Peak sales (M / year)",
        "notes": [f"For p = {BD['p']:g} the peak comes latest (year {t_late:.1f}) at q = {q_late:.2f}; stronger word of mouth than "
                  "that makes the peak both earlier and higher.",
                  f"At q = {BD['q']:g}, raising p from {ps[0]:g} to {ps[-1]:g} moves the peak from year {lo['peak_year']:.1f} to "
                  f"{hi['peak_year']:.1f} and from {lo['peak_sales']:.1f}M to {hi['peak_sales']:.1f}M a year."],
    }


@figure("bass-share")
def _fig_bass_share(p):
    """Share of each year's buyers who came through advertising, for the three presets."""
    series, said = [], []
    t = None
    for name, pp, q, c in _BASS_CASES:
        r = _bass(pp, q)
        innov, imit = np.asarray(r["sales_innovators"]), np.asarray(r["sales_imitators"])
        share = innov / np.maximum(innov + imit, 1e-12)
        t = r["t"]
        series.append({"name": f"{name} (p = {pp:g}, q = {q:g})", "color": c, "y": share})
        cross = _first(t, share < 0.5)
        said.append(f"{name.lower()}: {'never' if cross is None else f'from year {cross:.1f}'}")
    return {
        "kind": "lines", "x": t, "xlabel": "Year", "ylabel": "Share of this year's buyers from advertising", "ymin": 0, "ymax": 1,
        "series": series,
        "notes": ["Word of mouth brings more buyers than advertising once N/M passes p/q. " + "; ".join(said).capitalize() + ".",
                  f"At launch every buyer comes through advertising, since nobody owns one yet; by year {BD['years']:g}, "
                  f"{_pct(1 - series[0]['y'][-1])} of the typical product's few remaining buyers come through word of mouth."],
    }


@functools.lru_cache(maxsize=1)
def _bass_errors():
    pp, q, m, T = BD["p"], BD["q"], BD["market"], BD["years"]
    f = lambda _, y: np.array([(pp + q * y[0] / m) * (m - y[0])])
    exact = lambda t: m * (1 - np.exp(-(pp + q) * t)) / (1 + (q / pp) * np.exp(-(pp + q) * t))
    hs = [2.5, 1.25, 0.5, 0.25, 0.1, 0.05]
    eul, rk = [], []
    for h in hs:
        n = int(round(T / h))
        y, worst = np.array([0.0]), 0.0
        for k in range(n):
            y = y + h * f(k * h, y)
            worst = max(worst, abs(y[0] - exact((k + 1) * h)))
        eul.append(worst)
        t, Y = ode.rk4(f, [0.0], T, h)
        rk.append(float(np.abs(Y[:, 0] - exact(t)).max()))
    return hs, eul, rk


@figure("bass-solver")
def _fig_bass_solver(p):
    hs, eul, rk = _bass_errors()
    i = hs.index(0.5)
    return {
        "kind": "lines", "x": hs, "xlog": True, "ylog": True, "xlabel": "Step size h (years)",
        "ylabel": "Largest error in N (millions)",
        "series": [{"name": "Euler (first order)", "color": "warn", "y": eul},
                   {"name": "RK4 (fourth order)", "color": "accent", "y": rk}],
        "notes": [f"Halving h from {hs[i]:g} to {hs[i + 1]:g} years cuts Euler's error {eul[i] / eul[i + 1]:.1f}× and RK4's "
                  f"{rk[i] / rk[i + 1]:.0f}× (theory: 2× and 16×).",
                  f"At the live model's step h = {M.BASS_H:g} year, RK4's largest error is {_bass()['max_solver_error']:.1e} "
                  "million people."],
    }


# ===================================================================== Bass: text
_B0 = _bass()
_B_HIT, _B_ADS = _bass(*_BASS_CASES[1][1:3]), _bass(*_BASS_CASES[2][1:3])

explainer("spread/bass", {
    "title": "Why new products sell slowly, then fast, then stop: the Bass diffusion model",
    "kicker": "Modeling Lab · MC312 Lab 4",
    "summary": "People buy a new product for one of two reasons: something outside the market persuades them, or someone they know "
               "already owns one. The Bass model puts both into one equation. It explains the S-shaped curve of adoption, "
               "predicts when sales peak, and shows why word of mouth needs early buyers before it can do anything.",
    "sections": [
        {"id": "question", "title": "The adoption question", "blocks": [
            {"type": "p", "text": "Plot how many people own a new product against time and the curve is usually an S: a slow start, a "
                                  "steep middle, then a long flat top as the market fills up. Yearly sales rise, peak and fall. "
                                  "The model asks what produces that shape and what sets its timing."},
            {"type": "cards", "items": [
                {"title": "A fixed market M", "text": f"Everyone who could ever buy one: {BD['market']:g} million people in the live view. "
                                                      "Each buys once."},
                {"title": "Innovators, rate p", "text": "Buy because of something outside the market: advertising, coverage, a shop "
                                                        "window. They do not need anyone else to own one."},
                {"title": "Imitators, rate q", "text": "Buy because people they meet already own one. The more owners there are, "
                                                       "the stronger the pull."},
            ]},
            {"type": "p", "text": "The question: <i>given p, q and M, when do sales peak, how high, and how long until the market is "
                                  "saturated?</i>"},
        ]},
        {"id": "flow", "title": "Two reasons to buy", "blocks": [
            {"type": "p", "text": "There are only two groups: people who have bought (N) and people who have not (M − N). Each year "
                                  "some of the second group move to the first, along two routes."},
            {"type": "figure", "figure": "bass-flow", "title": "The market in a chosen year",
             "caption": "Move the year. The boxes fill with each group's share of the market; the arrows carry this year's buyers "
                        "by route.",
             "controls": [{"k": "year", "label": "Year", "min": 0, "max": BD["years"], "step": 0.5, "value": 3}]},
            {"type": "callout", "tone": "info", "title": "Why word of mouth starts at zero",
             "text": "At launch nobody owns the product, so there is nobody to imitate. However large q is, the first buyers must "
                     "come through advertising. Once they exist, each one adds to the pull on the rest."},
        ]},
        {"id": "equations", "title": "The equation, term by term", "blocks": [
            {"type": "eq", "text": "dN/dt = p (M − N)  +  q (N/M)(M − N)\n      = (p + q N/M)(M − N)",
             "where": [["N(t)", "people who have bought by time t (millions)"], ["M", "market size; M − N have not bought yet"],
                       ["p", "share of non-owners who buy each year because of outside influence"],
                       ["q", "strength of word of mouth; q·N/M is the extra buying rate when a share N/M already owns one"]]},
            {"type": "steps", "title": "One year of bookkeeping", "items": [
                "Count the people who have not bought yet: M − N.",
                "A share p of them buy because of advertising. This term is largest at launch and shrinks as the pool empties.",
                "A further share q·N/M buy because they know owners. This term starts at zero and grows with N.",
                "Add the two: that is this year's sales. N grows by that amount; M − N shrinks by the same.",
            ]},
            {"type": "figure", "figure": "bass-sales", "title": "Sales per year, split by reason",
             "caption": "Change p and q. Advertising sales fall steadily from launch; word-of-mouth sales rise, peak and fall.",
             "controls": [{"k": "p", "label": "Innovation p", "min": 0.001, "max": 0.15, "step": 0.001, "value": BD["p"]},
                          {"k": "q", "label": "Imitation q", "min": 0, "max": 1, "step": 0.01, "value": BD["q"]}]},
        ]},
        {"id": "shape", "title": "Where the S comes from", "blocks": [
            {"type": "p", "text": "Without word of mouth (q = 0) every year a fixed share p of the remaining non-owners buys, so "
                                  "adoption approaches M like a decaying exponential: fastest at the start. Word of mouth adds a "
                                  "term that is small when N is small, so the curve starts slowly, speeds up as owners multiply, "
                                  "and slows again when few non-owners are left."},
            {"type": "figure", "figure": "bass-adoption", "title": "Adoption for four strengths of word of mouth",
             "caption": "Change p. The dots mark when each curve reaches half the market.",
             "controls": [{"k": "p", "label": "Innovation p", "min": 0.005, "max": 0.15, "step": 0.005, "value": BD["p"]}]},
            {"type": "quiz", "q": "Advertising stops entirely (p becomes 0) before launch. What happens?",
             "options": ["Word of mouth sells the product more slowly", "Nobody ever buys it", "Sales start later but reach the same peak"],
             "answer": 1, "why": "With p = 0 and N = 0, both terms of dN/dt are zero: nobody owns one, so nobody can recommend it. "
                                 "Word of mouth can only amplify buying that something else has started."},
        ]},
        {"id": "peak", "title": "When sales peak, and how high", "blocks": [
            {"type": "p", "text": "The equation can be solved exactly. Sales are highest when the slope of the S is steepest, which "
                                  "gives the timing and size of the peak in closed form."},
            {"type": "eq", "text": "N(t) = M (1 − e^(−(p+q)t)) / (1 + (q/p) e^(−(p+q)t))\n"
                                   "t*   = ln(q/p) / (p + q)            (only when q > p)\n"
                                   "peak sales = M (p + q)² / (4q),    N(t*) = M (q − p) / (2q)",
             "where": [["t*", "year of peak sales"], ["N(t*)", "owners at the peak"]]},
            {"type": "p", "text": f"For the live view's default (p = {BD['p']:g}, q = {BD['q']:g}) the peak comes in year "
                                  f"{_B0['peak_year']:.1f} at {_B0['peak_sales']:.1f}M sales a year, when "
                                  f"{_pct((BD['q'] - BD['p']) / (2 * BD['q']))} of the market owns one. Sales peak "
                                  "<b>before</b> half the market has bought: the fall begins while most people are still to come."},
            {"type": "figure", "figure": "bass-peak-map", "title": "Timing against size of the peak",
             "caption": "Each line runs over q from low (left end near year 0) to high, for one value of p. The large dot is the default."},
            {"type": "quiz", "q": "Word of mouth gets stronger (q rises) while p stays fixed. What happens to the peak once q is well above p?",
             "options": ["Later and higher", "Earlier and higher", "Earlier and lower"], "answer": 1,
             "why": "Peak sales M(p + q)²/(4q) rise with q whenever q > p. The timing ln(q/p)/(p + q) first rises with q and then "
                    "falls, because a stronger pull fills the market faster."},
        ]},
        {"id": "who", "title": "Who buys when", "blocks": [
            {"type": "p", "text": "The share of each year's buyers who came through advertising starts at 100% and falls as owners "
                                  "accumulate. The two routes bring equal numbers when q·N/M = p, that is when N/M = p/q."},
            {"type": "figure", "figure": "bass-share", "title": "Advertising's share of each year's buyers",
             "caption": "The three presets of the live view. Below the 0.5 line, word of mouth brings most buyers."},
            {"type": "compare", "items": [
                {"title": "Word-of-mouth hit", "points": [f"p = {_BASS_CASES[1][1]:g}, q = {_BASS_CASES[1][2]:g}: peak in year {_B_HIT['peak_year']:.1f}",
                                                          f"{_pct(_B_HIT['share_innovators'])} of buyers through advertising",
                                                          "Slow start, then a sharp, high peak"]},
                {"title": "Advertising-led", "points": [f"p = q = {_BASS_CASES[2][1]:g}: no peak after launch",
                                                        f"{_pct(_B_ADS['share_innovators'])} of buyers through advertising",
                                                        f"Sales highest at launch, {_B_ADS['peak_sales']:.0f}M a year, then falling"]},
            ]},
            {"type": "try", "label": "Open a word-of-mouth hit", "params": {"p": _BASS_CASES[1][1], "q": _BASS_CASES[1][2]}},
            {"type": "try", "label": "Open an advertising-led product", "params": {"p": _BASS_CASES[2][1], "q": _BASS_CASES[2][2]}},
        ]},
        {"id": "solver", "title": "How the server computes it", "blocks": [
            {"type": "p", "text": f"The server integrates dN/dt with fourth-order Runge–Kutta (RK4) at a step of {M.BASS_H:g} year and "
                                  "compares the result with the closed form at every step. The two agree to many decimal places, "
                                  "which checks the solver on a problem where the exact answer is known."},
            {"type": "figure", "figure": "bass-solver", "title": "Error against step size",
             "caption": "Log–log axes. Euler's error falls in proportion to h, RK4's in proportion to h⁴."},
            {"type": "callout", "tone": "warn", "title": "What the model assumes",
             "text": "A fixed market that does not grow, one purchase per person and no repeat sales, constant p and q, and "
                     "everyone equally likely to hear from every owner. Price, competition and supply do not appear; in data they "
                     "show up as p and q that change over time."},
        ]},
    ],
    "takeaways": [
        "Buyers come from two routes: outside influence p(M − N) and word of mouth q(N/M)(M − N).",
        "Word of mouth needs owners, so early sales always come from advertising.",
        "With q > p adoption is an S-curve and sales peak at t* = ln(q/p)/(p + q).",
        f"Sales peak when N/M = (q − p)/(2q), before half the market has bought ({_pct((BD['q'] - BD['p']) / (2 * BD['q']))} for the default).",
        f"RK4 matches the closed form to {_B0['max_solver_error']:.0e} million people at h = {M.BASS_H:g} year.",
    ],
    "glossary": [
        ["Diffusion of innovations", "The spread of a new product or idea through a population over time."],
        ["Innovation coefficient p", "Yearly buying rate among non-owners due to outside influence."],
        ["Imitation coefficient q", "Strength of word of mouth; multiplied by the share of owners."],
        ["Market potential M", "Everyone who will eventually buy."],
        ["S-curve", "Cumulative adoption: slow, then fast, then saturating."],
        ["Closed form", "An exact formula for the solution, here N(t), used to check the numerical solver."],
    ],
}, order=2)


# ===================================================================== Rumour (Labs 7-8): figures
RD = _defaults(M.rumour)
_RUM_N0 = M.RUMOUR_N0


@functools.lru_cache(maxsize=512)
def _rum(beta: float = RD["beta"], gamma: float = RD["gamma"], alpha: float = RD["alpha"],
         newcomers: float = RD["newcomers"], days: float = M.RUMOUR_DAYS) -> dict:
    return M.rumour(beta=beta, gamma=gamma, alpha=alpha, newcomers=newcomers, days=days)


@figure("rumour-flow")
def _fig_rumour_flow(p):
    """The town on a chosen day: three groups, and the two kinds of meeting that move people between them."""
    day = p.num("day", 10, 0, 60)
    r = _rum()
    t = np.asarray(r["t"])
    i = int(np.argmin(np.abs(t - day)))
    ig, sp, st = (np.asarray(r[k]) for k in ("ignorant", "spreader", "stifler"))
    N = ig + sp + st
    tell, stop = RD["beta"] * sp * ig / N, RD["gamma"] * sp * (sp + st) / N
    vmax = float(tell.max())
    w = lambda v: 1.5 + 9 * min(1.0, v / vmax)
    a, b, c, n = float(ig[i]), float(sp[i]), float(st[i]), float(N[i])
    return {
        "kind": "flow", "height": 230,
        "nodes": [
            {"id": "I", "x": 20, "y": 70, "w": 150, "h": 90, "title": "Never heard", "value": f"{a:.0f}", "sub": "i",
             "fill": a / n, "fill_color": "s1"},
            {"id": "S", "x": 225, "y": 70, "w": 150, "h": 90, "title": "Spreading", "value": f"{b:.0f}", "sub": "s",
             "fill": b / n, "fill_color": "s2", "hot": True},
            {"id": "R", "x": 430, "y": 70, "w": 150, "h": 90, "title": "Stopped", "value": f"{c:.0f}", "sub": "r",
             "fill": c / n, "fill_color": "s3"},
        ],
        "edges": [
            {"from": "I", "to": "S", "label": f"{tell[i]:.0f} / day", "below": "β s i / N", "width": w(tell[i]), "flow": True, "color": "s2"},
            {"from": "S", "to": "R", "label": f"{stop[i]:.0f} / day", "below": "γ s (s + r) / N", "width": w(stop[i]), "flow": True, "color": "s3"},
        ],
        "texts": [{"x": 300, "y": 30, "text": f"Day {t[i]:.0f} in a town of {n:.0f}", "cls": "n-title"}],
        "notes": [f"On day {t[i]:.0f}, {tell[i]:.0f} people a day hear the rumour and {stop[i]:.0f} spreaders a day give up.",
                  "More are told than give up, so the number spreading is still growing." if tell[i] > stop[i] else
                  "More give up than are told, so the number spreading is falling, and with no newcomers it never rises again."],
    }


@figure("rumour-curves")
def _fig_rumour_curves(p):
    beta, gamma = p.num("beta", RD["beta"], 0.05, 1), p.num("gamma", RD["gamma"], 0.02, 0.5)
    r = _rum(beta, gamma)
    th, n = r["theory"], len(r["t"])
    series = [{"name": "Never heard", "color": "s1", "y": r["ignorant"]},
              {"name": "Spreading", "color": "s2", "y": r["spreader"], "area": True},
              {"name": "Stopped", "color": "s3", "y": r["stifler"]},
              {"name": "Closed-form peak", "color": "neutral", "y": [th["s_max"]] * n, "dash": True, "thin": True}]
    still = float(np.asarray(r["spreader"])[-1])
    end = (f"By day {r['days']:.0f}, {_pct(r['reach'])} have heard it (closed form {_pct(th['r_inf'])}) and "
           f"{float(np.asarray(r['ignorant'])[-1]):.0f} people never have.")
    if still >= 1:
        end += f" {still:.0f} are still spreading, so the rumour has not finished."
    return {
        "kind": "lines", "x": r["t"], "xlabel": "Day", "ylabel": "People", "ymin": 0, "ymax": r["population"],
        "series": series, "markers": [{"x": r["peak_day"], "label": f"peak {r['peak_spreaders']:.0f}"}],
        "notes": [f"c = β/γ = {beta / gamma:.2f}, so R₀ = 1 + c = {th['R0']:.2f}. Spreaders peak at {r['peak_spreaders']:.0f} on day "
                  f"{r['peak_day']:.0f}; the closed form predicts {th['s_max']:.0f}.", end],
    }


@figure("rumour-balance")
def _fig_rumour_balance(p):
    """Per-spreader rates: telling (β i/N) against giving up (γ (s + r)/N)."""
    r = _rum()
    ig, sp, st = (np.asarray(r[k]) for k in ("ignorant", "spreader", "stifler"))
    N, t = ig + sp + st, np.asarray(r["t"])
    tell, stop = RD["beta"] * ig / N, RD["gamma"] * (sp + st) / N
    j = int(np.argmax(tell < stop))
    R0 = r["theory"]["R0"]
    return {
        "kind": "lines", "x": t, "xlabel": "Day", "ylabel": "Per spreader, per day",
        "series": [{"name": "Tells someone new: β i / N", "color": "s2", "y": tell},
                   {"name": "Gives up: γ (s + r) / N", "color": "s3", "y": stop},
                   {"name": "Spreaders (scaled to fit)", "color": "neutral", "y": sp / r["peak_spreaders"] * RD["beta"],
                    "dash": True, "area": True}],
        "markers": [{"x": float(t[j]), "label": "rates equal"}],
        "notes": [f"The two rates cross on day {t[j]:.1f}; spreaders peak on day {r['peak_day']:.1f}.",
                  f"At that moment {_pct(ig[j] / N[j])} of people have not heard it, matching 1/R₀ = 1/(1 + β/γ) = {_pct(1 / R0)}."],
    }


def _reach_theory(c: float) -> float:
    """Share who ever hear the rumour when almost everyone starts out not having heard it: r = 1 − e^(−(1 + c) r)."""
    return _final_size(1 + c)


@functools.lru_cache(maxsize=1)
def _reach_sims():
    g = 0.5                                    # fast stifling, so even the slowest rumour finishes within the horizon
    return [(c, _rum(c * g, g, days=150.0 if c < 0.5 else float(M.RUMOUR_DAYS))["reach"]) for c in (0.25, 0.5, 1, 2, 4)]


@figure("rumour-reach")
def _fig_rumour_reach(p):
    cs = np.round(np.linspace(0.05, 6, 120), 4)
    rum, epi = [_reach_theory(c) for c in cs], [_final_size(c) for c in cs]
    sims = _reach_sims()
    c0 = RD["beta"] / RD["gamma"]
    gap = max(abs(s - _reach_theory(c)) for c, s in sims)
    return {
        "kind": "lines", "x": cs, "xlabel": "c = β / γ", "ylabel": "Share who ever hear it", "ymin": 0, "ymax": 1,
        "series": [{"name": "Rumour: r = 1 − e^(−(1 + c) r)", "color": "s2", "y": rum, "area": True},
                   {"name": "Epidemic with R₀ = c (SIR final size)", "color": "neutral", "y": epi, "dash": True}],
        "points": [{"x": c, "y": s, "color": "accent"} for c, s in sims]
                  + [{"x": c0, "y": _rum()["reach"], "color": "s1", "label": "default"}],
        "notes": [f"There is no threshold: even at c = {cs[0]:g} the rumour reaches {_pct(rum[0])}, where an epidemic with R₀ = "
                  f"{cs[0]:g} would not start. With β = γ (c = 1) it reaches {_pct(_reach_theory(1))}.",
                  f"Dots are full simulations, which start with {M.RUMOUR_S0} of {_RUM_N0} people spreading; they sit within "
                  f"{100 * gap:.2g} percentage points of the curve."],
    }


@figure("rumour-check")
def _fig_rumour_check(p):
    r = _rum()
    th = r["theory"]
    return {
        "kind": "stats",
        "items": [{"label": "Peak spreaders, RK4", "value": f"{r['peak_spreaders']:.1f}", "note": f"closed form {th['s_max']:.1f}", "tone": "good"},
                  {"label": "Ever heard it, RK4", "value": _pct(r["reach"], 1), "note": f"closed form {_pct(th['r_inf'], 1)}"},
                  {"label": "Drift of the conserved quantity", "value": f"{r['conserved_drift']:.1e}", "note": f"over {r['days']:.0f} days",
                   "tone": "good"},
                  {"label": "R₀ = 1 + β/γ", "value": f"{th['R0']:.2f}", "note": f"β = {RD['beta']:g}, γ = {RD['gamma']:g}"}],
        "notes": [f"The solver's peak differs from the closed form by {abs(r['peak_spreaders'] - th['s_max']):.2g} people.",
                  f"Reach differs by {100 * abs(r['reach'] - th['r_inf']):.2g} percentage points: the closed form assumes almost "
                  f"nobody has heard it at the start, while the simulation starts with {M.RUMOUR_S0} spreaders."],
    }


@figure("rumour-campaign")
def _fig_rumour_campaign(p):
    alpha, new = p.num("alpha", 0.05, 0, 0.1), p.num("newcomers", 0, 0, 10)
    base, r = _rum(), _rum(alpha=alpha, newcomers=new)
    if alpha == 0 and new == 0:
        second = "With no campaign and no newcomers this is the plain model."
    elif new == 0:
        second = (f"The campaign moves {_pct(alpha, 1)} of those who have not heard it to 'stopped' each day, so spreaders run "
                  "out of people to tell sooner.")
    else:
        second = (f"{new:g} newcomers a day keep supplying people who have not heard it: on day {r['days']:.0f}, "
                  f"{float(np.asarray(r['spreader'])[-1]):.1f} people are still spreading, against "
                  f"{float(np.asarray(base['spreader'])[-1]):.1f} with no intervention.")
    return {
        "kind": "lines", "x": r["t"], "xlabel": "Day", "ylabel": "People", "ymin": 0,
        "series": [{"name": "Spreading, no intervention", "color": "neutral", "y": base["spreader"], "dash": True},
                   {"name": "Spreading, with these settings", "color": "s2", "y": r["spreader"], "area": True},
                   {"name": "Never heard, with these settings", "color": "s1", "y": r["ignorant"], "thin": True}],
        "notes": [f"Peak spreaders {base['peak_spreaders']:.0f} → {r['peak_spreaders']:.0f}; share who ever hear it "
                  f"{_pct(base['reach'])} → {_pct(r['reach'])}.", second],
    }


# ===================================================================== Rumour: text
_R0 = _rum()
_RC = RD["beta"] / RD["gamma"]
_R_NEVER = float(np.asarray(_R0["ignorant"])[-1])
_R_SLOW = _rum(RD["beta"], min(0.5, 3 * RD["gamma"]))
_R_EDU = _rum(alpha=0.05)

explainer("spread/rumour", {
    "title": "How a rumour spreads, and why it stops before everyone hears it",
    "kicker": "Modeling Lab · MC312 Labs 7–8",
    "summary": "A rumour passes from person to person like an infection, but nobody recovers from knowing it. Spreaders stop when "
               "they keep meeting people who already know. That one change to the SIR model means a rumour always takes off, "
               f"yet always dies out with some people never told: in the model's town of {_RUM_N0:,}, {_R_NEVER:.0f} never hear it.",
    "sections": [
        {"id": "question", "title": "A rumour is not a disease", "blocks": [
            {"type": "p", "text": "In an epidemic, people stop being infectious after a while whatever happens around them. A rumour "
                                  "is different: people keep passing it on until they find that the people they meet have already "
                                  "heard it. Then the story is stale and they stop. Stopping depends on how far the rumour has "
                                  "spread, not on time."},
            {"type": "cards", "items": [
                {"title": "Never heard (i)", "text": f"Everyone at the start except {M.RUMOUR_S0} people: {_RUM_N0 - M.RUMOUR_S0:,} of {_RUM_N0:,}."},
                {"title": "Spreading (s)", "text": "Know the rumour and tell it to everyone who has not heard it."},
                {"title": "Stopped (r)", "text": "Know it but no longer pass it on. Rumour models call them stiflers."},
            ]},
            {"type": "p", "text": "The question: <i>how many people eventually hear the rumour, how many are spreading it at the peak, "
                                  "and what would an education campaign change?</i>"},
        ]},
        {"id": "flow", "title": "Three groups, two kinds of meeting", "blocks": [
            {"type": "p", "text": "Everyone mixes at random. When a spreader meets someone who has not heard, that person becomes a "
                                  "spreader. When a spreader meets someone who already knows (another spreader or a stifler), the "
                                  "spreader stops."},
            {"type": "figure", "figure": "rumour-flow", "title": "The town on one day",
             "caption": "Move the day. The boxes fill with each group's share; arrow widths carry today's flows.",
             "controls": [{"k": "day", "label": "Day", "min": 0, "max": 60, "step": 1, "value": 10}]},
            {"type": "callout", "tone": "info", "title": "Why the stopping arrow grows",
             "text": "Early on almost everyone a spreader meets is new to the rumour, so nobody stops. As knowers pile up, each "
                     "spreader is more and more likely to meet one. The rumour creates the very people who end it."},
        ]},
        {"id": "equations", "title": "The equations, term by term", "blocks": [
            {"type": "eq", "text": "di/dt = − β s i / N\nds/dt =   β s i / N − γ s (s + r) / N\ndr/dt =   γ s (s + r) / N",
             "where": [["i, s, r", "people who have never heard it, are spreading it, have stopped (i + s + r = N)"],
                       ["β", "rate at which a spreader tells someone, if that person has not heard"],
                       ["γ", "rate at which a spreader gives up, if the person met already knows"],
                       ["i/N, (s + r)/N", "chance that a meeting is with someone who has not heard, or has"]]},
            {"type": "steps", "title": "One day of bookkeeping", "items": [
                "Each spreader has meetings at a steady rate.",
                "A share i/N of them are with people who have not heard: β s i / N people become spreaders.",
                "A share (s + r)/N are with people who know: γ s (s + r) / N spreaders stop.",
                "Compare with SIR: the recovery term α I is replaced by a term that grows as the rumour spreads.",
                "Add the three equations: the total is unchanged, so nobody is lost or created.",
            ]},
        ]},
        {"id": "curves", "title": "The whole rumour", "blocks": [
            {"type": "p", "text": "Only the ratio <b>c = β/γ</b> decides how far the rumour gets; β on its own sets how fast. The curves "
                                  "look like an epidemic: spreaders rise, peak and fall, the never-heard group shrinks, and the "
                                  "stopped group grows."},
            {"type": "figure", "figure": "rumour-curves", "title": f"{M.RUMOUR_DAYS} days of a rumour",
             "caption": "Change β and γ. The dashed line is the peak predicted by the closed form below.",
             "controls": [{"k": "beta", "label": "Spreading β", "min": 0.05, "max": 1, "step": 0.01, "value": RD["beta"]},
                          {"k": "gamma", "label": "Stifling γ", "min": 0.02, "max": 0.5, "step": 0.01, "value": RD["gamma"]}]},
        ]},
        {"id": "peak", "title": "Why it peaks, and why there is no threshold", "blocks": [
            {"type": "p", "text": "Per spreader, the rate of telling someone new is β i/N and the rate of giving up is γ (s + r)/N = "
                                  "γ (1 − i/N). Spreaders increase while the first is larger. The two are equal when "
                                  "i/N = 1/(1 + c), so the peak comes when the never-heard share has fallen to 1/(1 + c)."},
            {"type": "figure", "figure": "rumour-balance", "title": "Telling against giving up",
             "caption": "Per-spreader rates for the default rumour. The dashed curve is the number spreading, scaled to fit."},
            {"type": "callout", "tone": "tip", "title": "R₀ for a rumour",
             "text": "In SIR the peak comes when S/N = 1/R₀. Here it comes when i/N = 1/(1 + c), so 1 + c plays the part of R₀. "
                     "Since c is positive, R₀ = 1 + c is always above 1: as long as anyone is spreading at all, a rumour started "
                     f"among people who have not heard it always grows at first. For the default, R₀ = {_R0['theory']['R0']:.2f}."},
            {"type": "quiz", "q": "Spreaders give up three times as readily (γ triples). Does the rumour still take off?",
             "options": ["No, it dies out at once", "Yes, but it reaches fewer people", "Yes, and it reaches the same number, only faster"],
             "answer": 1, "why": f"R₀ = 1 + β/γ falls from {_R0['theory']['R0']:.2f} to {_R_SLOW['theory']['R0']:.2f} but stays above 1, "
                                 f"so the rumour still grows. It reaches {_pct(_R_SLOW['reach'])} instead of {_pct(_R0['reach'])}."},
        ]},
        {"id": "size", "title": "How far it gets: a conserved quantity", "blocks": [
            {"type": "p", "text": "Dividing ds/dt by di/dt removes time. Using fractions of the town and s + r = 1 − i, the result can "
                                  "be integrated exactly, and it gives a quantity that stays constant along every trajectory:"},
            {"type": "eq", "text": "Q = s − (1/c) ln i + (1 + 1/c) i      (constant)\n"
                                   "s_max = 1 − (1/c)(1 + ln i₀ − i₀ + ln(1 + c))\n"
                                   "r∞ = 1 − e^(−(1 + c) r∞)",
             "where": [["i, s", "fractions of the town never heard and spreading"], ["i₀", "fraction who have not heard at the start"],
                       ["s_max", "largest fraction spreading at once, reached when i = 1/(1 + c)"],
                       ["r∞", "share who ever hear it, when almost nobody has at the start"]]},
            {"type": "p", "text": "The last equation is the SIR final-size equation with R₀ replaced by 1 + c. It has a positive "
                                  "solution for every c, and the solution is always below 1: some people never hear the rumour, "
                                  "because by the time it could reach them every spreader has stopped."},
            {"type": "figure", "figure": "rumour-reach", "title": "Reach against c",
             "caption": "The rumour's reach (solid) against an epidemic's final size with R₀ = c (dashed), with full simulations as dots."},
        ]},
        {"id": "solver", "title": "How the server computes it", "blocks": [
            {"type": "p", "text": f"The server integrates the three equations with fourth-order Runge–Kutta (RK4), step {M.RUMOUR_H:g} "
                                  f"day, for {M.RUMOUR_DAYS} days. Without a campaign or newcomers it also tracks the quantity Q above: "
                                  "the exact equations keep it constant, so any change in it is numerical error."},
            {"type": "figure", "figure": "rumour-check", "title": "The solver against the closed forms",
             "caption": "Default settings. Peak and reach from RK4, next to the formulas."},
        ]},
        {"id": "control", "title": "Education campaigns and newcomers", "blocks": [
            {"type": "p", "text": "An <b>education campaign</b> teaches people to ignore the rumour before they hear it, moving them "
                                  "straight from 'never heard' to 'stopped' at rate α. <b>Newcomers</b> join the town at Λ a day, "
                                  "all of them never having heard it."},
            {"type": "eq", "text": "di/dt = − β s i / N − α i + Λ\ndr/dt =   γ s (s + r) / N + α i",
             "where": [["α", "share of the never-heard reached by the campaign each day"], ["Λ", "newcomers per day"]]},
            {"type": "figure", "figure": "rumour-campaign", "title": "A campaign and a growing town",
             "caption": "Change the campaign and the inflow of newcomers. The dashed curve is the rumour with neither.",
             "controls": [{"k": "alpha", "label": "Campaign α", "min": 0, "max": 0.1, "step": 0.005, "value": 0.05, "unit": "/day"},
                          {"k": "newcomers", "label": "Newcomers Λ", "min": 0, "max": 10, "step": 1, "value": 0, "unit": "/day"}]},
            {"type": "compare", "items": [
                {"title": "Campaign", "points": ["Removes listeners before the rumour reaches them",
                                                 "Also adds stiflers, who make spreaders stop sooner",
                                                 f"α = 0.05 cuts reach from {_pct(_R0['reach'])} to {_pct(_R_EDU['reach'])}"]},
                {"title": "Newcomers", "points": ["Keep supplying people who have not heard", "Spreaders never quite run out",
                                                  "The conserved quantity no longer holds, so there is no closed form"]},
            ]},
            {"type": "quiz", "q": "Why does a campaign that reaches only a few per cent of people a day cut the rumour's reach so much?",
             "options": ["It converts spreaders back", "It removes listeners and adds stiflers at the same time",
                         "It slows down every meeting"], "answer": 1,
             "why": "Each person the campaign reaches is one fewer person to tell and one more knower for spreaders to meet. "
                    "Both effects push the per-spreader balance towards giving up."},
            {"type": "try", "label": "Open the education campaign", "params": {"beta": RD["beta"], "gamma": RD["gamma"], "alpha": 0.05, "newcomers": 0}},
            {"type": "try", "label": "Open a growing town", "params": {"beta": RD["beta"], "gamma": RD["gamma"], "alpha": 0, "newcomers": 5}},
        ]},
        {"id": "limits", "title": "What the model assumes", "blocks": [
            {"type": "callout", "tone": "warn", "title": "Assumptions",
             "text": "Everyone meets everyone else equally often; a spreader stops after one meeting with a knower; nobody forgets "
                     "the rumour or starts spreading it again; and a stifler never passes it on. Real rumours travel through networks "
                     "of friends, where clusters can hear it again and again while other groups are never reached."},
            {"type": "compare", "items": [
                {"title": "SIR epidemic", "points": ["Recovery at a fixed rate α", "Threshold: no outbreak if R₀ = β/α ≤ 1",
                                                     "Final size z = 1 − e^(−R₀ z)"]},
                {"title": "Rumour", "points": ["Stopping depends on meeting knowers", "No threshold: R₀ = 1 + β/γ > 1 always",
                                               "Reach r∞ = 1 − e^(−(1 + c) r∞)"]},
            ]},
        ]},
    ],
    "takeaways": [
        "A rumour is SIR with a twist: spreaders stop when they meet people who already know, at rate γ (s + r)/N.",
        "Only c = β/γ decides how far it gets; spreaders peak when the never-heard share falls to 1/(1 + c).",
        "R₀ = 1 + c is always above 1, so there is no threshold: a rumour always takes off.",
        f"It never reaches everyone: r∞ = 1 − e^(−(1 + c) r∞), {_pct(_R0['theory']['r_inf'])} for the default.",
        "A conserved quantity gives the peak in closed form and checks the RK4 solver.",
    ],
    "glossary": [
        ["Ignorant", "Someone who has not heard the rumour."],
        ["Spreader", "Someone who knows the rumour and passes it on."],
        ["Stifler", "Someone who knows the rumour but no longer passes it on."],
        ["c = β/γ", "Ratio of the telling rate to the giving-up rate; sets the rumour's reach."],
        ["Conserved quantity", "A combination of the variables that stays constant as the system evolves."],
        ["Threshold", "A value of R₀ below which an outbreak cannot grow; rumours have none."],
    ],
}, order=4)


# ===================================================================== Monte Carlo (Lab 9): figures
CD = _defaults(M.decay)
KD = _defaults(M.coins)
_N0, _TMAX = M.DECAY_N0, M.DECAY_T_MAX
_SERIES = ("s1", "s2", "s3", "s4")


@functools.lru_cache(maxsize=256)
def _decay(lam: float = CD["lam"], runs: int = CD["runs"]) -> dict:
    return M.decay(lam=lam, runs=runs)


@functools.lru_cache(maxsize=256)
def _coins(n: int = KD["n"]) -> dict:
    return M.coins(n=n)


@figure("chance-atoms")
def _fig_chance_atoms(p):
    """The first run's atoms at a chosen moment: still there or decayed."""
    r = _decay()
    t = p.num("t", r["half_life_step"], 0, 60, integer=True)
    times = np.asarray(r["first_run_decay_times"])
    alive = int((times > t).sum())
    cols = 50
    expected = _N0 * math.exp(-CD["lam"] * t)
    second = (f"This is the half-life, ln 2/λ ≈ {r['half_life_theory']:.1f} s, rounded to whole seconds: about half are left."
              if t == r["half_life_step"] else
              ("No atom has decayed yet." if t == 0 else
               f"{_N0 - alive} have decayed. Which ones is pure chance, so the orange squares form no pattern."))
    return {
        "kind": "grid", "cols": cols, "rows": int(np.ceil(_N0 / cols)), "palette": "cat", "colors": ["neutral", "s1", "s2"],
        "cells": [1 if T > t else 2 for T in times], "height": 300,
        "notes": [f"At t = {t} s, {alive} of {_N0} atoms in this run are left; N₀e^(−λt) predicts {expected:.0f}.", second],
    }


@figure("chance-times")
def _fig_chance_times(p):
    """Histogram of the first run's decay times against the exponential law."""
    r = _decay()
    times, lam = np.asarray(r["first_run_decay_times"]), CD["lam"]
    edges = list(range(0, _TMAX + 1, 10))
    counts = [int(((times > a) & (times <= b)).sum()) for a, b in zip(edges, edges[1:])] + [int((times > _TMAX).sum())]
    expect = [_N0 * (math.exp(-lam * a) - math.exp(-lam * b)) for a, b in zip(edges, edges[1:])] + [_N0 * math.exp(-lam * _TMAX)]
    labels = [f"{a + 1}–{b}" for a, b in zip(edges, edges[1:])] + [f">{_TMAX}"]
    return {
        "kind": "bars", "labels": labels, "xlabel": "Decay time (s)", "ylabel": "Atoms",
        "series": [{"name": "First run", "color": "s2", "values": counts},
                   {"name": "Expected, N₀(e^(−λa) − e^(−λb))", "color": "neutral", "values": expect}],
        "notes": [f"The first ten seconds hold {counts[0]} atoms (expected {expect[0]:.0f}); each later bin expects "
                  f"e^(−10λ) = {math.exp(-10 * lam):.2f} times as many as the one before.",
                  f"{counts[-1]} atoms outlast the whole {_TMAX} s run (expected {expect[-1]:.1f})."],
    }


@figure("chance-runs")
def _fig_chance_runs(p):
    lam, runs = p.num("lam", CD["lam"], 0.01, 0.3), p.num("runs", CD["runs"], 10, 500, integer=True)
    r = _decay(lam, runs)
    th, one = np.asarray(r["theory"]), np.asarray(r["runs_shown"])
    hl = (f"The mean halves at t = {r['half_life_measured']:.0f} s, against ln 2/λ = {r['half_life_theory']:.1f} s."
          if r["half_life_measured"] is not None else f"The mean has not halved within {_TMAX} s.")
    return {
        "kind": "lines", "x": r["t"], "xlabel": "Seconds", "ylabel": "Atoms left", "ymin": 0, "ymax": _N0,
        "series": [{"name": "One run", "color": "neutral", "y": one[0], "thin": True},
                   {"name": "Another run", "color": "s4", "y": one[1], "thin": True},
                   {"name": f"Mean of {runs} runs", "color": "s1", "y": r["mean"]},
                   {"name": "N₀e^(−λt)", "color": "s2", "y": th, "dash": True}],
        "markers": [{"x": r["half_life_theory"], "label": "t½"}] if r["half_life_theory"] <= _TMAX else [],
        "notes": [hl, f"The mean of {runs} runs never strays more than {r['max_gap_to_theory']:.1f} atoms from N₀e^(−λt); "
                      f"a single run strays up to {float(np.abs(one[0] - th).max()):.1f}."],
    }


@figure("chance-spread")
def _fig_chance_spread(p):
    """How far single runs scatter around the mean, against the binomial formula."""
    r = _decay()
    t, lam = np.asarray(r["t"]), CD["lam"]
    q = np.exp(-lam * t)
    th = np.sqrt(_N0 * q * (1 - q))
    i = r["half_life_step"]
    mean = np.asarray(r["mean"])
    return {
        "kind": "lines", "x": t, "xlabel": "Seconds", "ylabel": "Standard deviation across runs (atoms)", "ymin": 0,
        "series": [{"name": f"Simulated, {CD['runs']} runs", "color": "s1", "y": r["sd"]},
                   {"name": "Binomial √(N₀ q (1 − q)), q = e^(−λt)", "color": "s2", "y": th, "dash": True}],
        "markers": [{"x": i, "label": "half-life"}],
        "notes": [f"The spread is largest near the half-life, where each atom's survival is an even chance: theory ±"
                  f"{r['sd_theory_at_half_life']:.1f} atoms, simulation ±{r['sd_at_half_life']:.1f}.",
                  f"Relative to the count, single runs get noisier as atoms run out: ±{_pct(r['sd'][i] / mean[i])} at the half-life, "
                  f"±{_pct(r['sd'][-1] / mean[-1])} at t = {_TMAX} s."],
    }


_CONV_RUNS = (10, 30, 100, 300, 1000)


@figure("chance-convergence")
def _fig_chance_convergence(p):
    gaps = [_decay(runs=n)["max_gap_to_theory"] for n in _CONV_RUNS]
    ref = [gaps[0] * math.sqrt(_CONV_RUNS[0] / n) for n in _CONV_RUNS]
    return {
        "kind": "lines", "x": list(_CONV_RUNS), "xlog": True, "ylog": True, "xlabel": "Runs averaged",
        "ylabel": "Worst gap between the mean and N₀e^(−λt) (atoms)",
        "series": [{"name": "Simulated", "color": "s1", "y": gaps},
                   {"name": "Falling like 1/√runs", "color": "neutral", "y": ref, "dash": True}],
        "points": [{"x": n, "y": g, "color": "s1"} for n, g in zip(_CONV_RUNS, gaps)],
        "notes": [f"From {_CONV_RUNS[0]} to {_CONV_RUNS[-1]} runs the worst gap falls from {gaps[0]:.1f} to {gaps[-1]:.2f} atoms, "
                  f"{gaps[0] / gaps[-1]:.1f}× smaller; the 1/√runs law predicts {math.sqrt(_CONV_RUNS[-1] / _CONV_RUNS[0]):.0f}×.",
                  "Each point is one set of random draws, so it wobbles about the dashed line rather than sitting on it."],
    }


@figure("chance-coins")
def _fig_chance_coins(p):
    n = p.num("n", KD["n"], 100, 50000, integer=True)
    c = _coins(n)
    share = np.asarray(c["share"])
    final, band = np.asarray(c["final"]), float(np.asarray(c["band"])[-1])
    inside = int((np.abs(final - c["p_heads"]) <= band).sum())
    return {
        "kind": "lines", "x": c["n"], "xlog": True, "xlabel": "Tosses (log scale)", "ylabel": "Share of heads", "ymin": 0, "ymax": 1,
        "series": [{"name": f"Sequence {k + 1}", "color": _SERIES[k % 4], "y": share[k], "thin": True} for k in range(len(share))]
                  + [{"name": "95% band, upper", "color": "neutral", "y": c["band_hi"], "dash": True},
                     {"name": "95% band, lower", "color": "neutral", "y": c["band_lo"], "dash": True}],
        "notes": [f"After {n:,} tosses the {c['sequences']} sequences sit between {_pct(c['final_min'], 1)} and "
                  f"{_pct(c['final_max'], 1)} heads.",
                  f"Here the band reaches {100 * band:.1f} percentage points either side of {_pct(c['p_heads'])}, and {inside} of "
                  f"{c['sequences']} sequences end inside it."],
    }


@figure("chance-check")
def _fig_chance_check(p):
    r, c = _decay(), _coins()
    return {
        "kind": "stats",
        "items": [{"label": "Half-life, measured", "value": f"{r['half_life_measured']:.0f} s",
                   "note": f"ln 2 / λ = {r['half_life_theory']:.1f} s", "tone": "good"},
                  {"label": "Spread at the half-life", "value": f"±{r['sd_at_half_life']:.1f}",
                   "note": f"binomial ±{r['sd_theory_at_half_life']:.1f}"},
                  {"label": "Worst gap, mean against theory", "value": f"{r['max_gap_to_theory']:.1f}", "note": f"atoms, {CD['runs']} runs"},
                  {"label": "Coins after the last toss", "value": f"{_pct(c['final_min'], 1)}–{_pct(c['final_max'], 1)}",
                   "note": f"{c['sequences']} sequences of {KD['n']:,}"}],
        "notes": [f"All of this comes from seed {CD['seed']}: the same seed always draws the same numbers, so every figure on this "
                  "page can be reproduced exactly.",
                  f"One experiment draws {_N0:,} uniform numbers per run, {_N0 * CD['runs']:,} for the {CD['runs']} runs averaged here."],
    }


# ===================================================================== Monte Carlo: text
_C0 = _decay()
_LAM = CD["lam"]
_P_SEC = 1 - math.exp(-_LAM)
_FAST = 0.2                                    # the live view's 'Fast decay' preset

explainer("spread/chance", {
    "title": "How randomness averages out: Monte Carlo decay and coin tosses",
    "kicker": "Modeling Lab · MC312 Lab 9",
    "summary": f"Give each of {_N0:,} atoms its own random moment to decay and nothing in the code says 'exponential', yet the "
               "count of survivors follows N₀e^(−λt). Monte Carlo simulation turns a rule for one random event into predictions "
               f"for the whole, and the same averaging explains why the share of heads in a long run of tosses settles on {M.COIN_P:g}.",
    "sections": [
        {"id": "question", "title": "One rule per atom", "blocks": [
            {"type": "p", "text": "A radioactive atom does not age. In every second it has the same small chance of decaying, "
                                  "whatever has happened before. Nobody can say <i>when</i> a given atom will go, but the rule is "
                                  "exact. Monte Carlo simulation takes such a rule, plays it out with random numbers many times, "
                                  "and reads off the average and the scatter."},
            {"type": "cards", "items": [
                {"title": "Many atoms", "text": f"{_N0:,} atoms per run, each decaying independently of the others."},
                {"title": "One rate", "text": f"Decay constant λ = {_LAM:g} per second in the live view's default."},
                {"title": "Many runs", "text": f"The whole experiment repeated {CD['runs']} times with fresh random numbers, then averaged."},
            ]},
            {"type": "p", "text": "The question: <i>why does the average of many random runs trace a smooth curve, how far does a "
                                  "single run stray from it, and how many runs are enough?</i>"},
        ]},
        {"id": "one-run", "title": "One run of the experiment", "blocks": [
            {"type": "p", "text": "Here is the first run, one square per atom. Move the clock and watch atoms disappear at random."},
            {"type": "figure", "figure": "chance-atoms", "title": f"{_N0:,} atoms, one run",
             "caption": "Blue: still there. Orange: decayed by time t. The slider starts at the half-life.",
             "controls": [{"k": "t", "label": "Time t", "min": 0, "max": 60, "step": 1, "value": int(_C0["half_life_step"]), "unit": "s"}]},
            {"type": "callout", "tone": "info", "title": "Random but predictable",
             "text": "No square knows about any other. Yet with a thousand of them, the share left at any moment is close to "
                     "e^(−λt), because each atom is a separate draw and their errors mostly cancel."},
        ]},
        {"id": "rule", "title": "Turning a random number into a decay time", "blocks": [
            {"type": "p", "text": "If the chance of decaying is the same in every instant, the chance of an atom surviving to time t "
                                  "is e^(−λt). The server inverts that: it draws a uniform random number u and solves e^(−λT) = 1 − u "
                                  "for the decay time T, then rounds up to the next whole second."},
            {"type": "eq", "text": "P(atom still there at t) = e^(−λt)\nT = ⌈ −ln(1 − u) / λ ⌉,     u uniform on [0, 1)",
             "where": [["λ", "decay constant, per second"], ["u", "a uniform random number, one per atom"],
                       ["T", "the second in which that atom decays"], ["⌈ ⌉", "round up to a whole second"]]},
            {"type": "steps", "title": "Why the formula works", "items": [
                "Small values of u are as likely as large ones.",
                "−ln(1 − u) is near 0 for most u, and large only when u is close to 1.",
                "So most decay times are short and a few are long, in exactly the proportions e^(−λt) requires.",
                "Rounding up groups the continuous times into whole seconds, the time step the experiment counts in.",
            ]},
            {"type": "figure", "figure": "chance-times", "title": "When the first run's atoms decayed",
             "caption": "Decay times in ten-second bins, against the count the exponential law expects."},
            {"type": "quiz", "q": "An atom has survived 50 s. What is its chance of decaying in the next second, compared with a fresh atom's?",
             "options": ["Higher: it is overdue", "The same", "Lower: it has proved stable"], "answer": 1,
             "why": f"The rule has no memory. Every atom still present has the same chance, 1 − e^(−λ) = {_pct(_P_SEC, 1)} per second "
                    f"for λ = {_LAM:g}, however long it has already lasted."},
        ]},
        {"id": "many", "title": "Many runs and their average", "blocks": [
            {"type": "p", "text": "Each run gives a jagged staircase. The average over many runs is smooth and lies on the theory, "
                                  "because averaging cancels the independent errors of the runs."},
            {"type": "eq", "text": "E[N(t)] = N₀ e^(−λt)\nt½ = ln 2 / λ",
             "where": [["N(t)", "atoms left at time t in one run"], ["N₀", f"atoms at the start, {_N0:,}"], ["t½", "half-life"]]},
            {"type": "figure", "figure": "chance-runs", "title": "Single runs, their mean and the theory",
             "caption": "Change λ and the number of runs. Single runs wander; the mean follows the dashed curve.",
             "controls": [{"k": "lam", "label": "Decay constant λ", "min": 0.01, "max": 0.3, "step": 0.005, "value": _LAM, "unit": "/s"},
                          {"k": "runs", "label": "Runs averaged", "min": 10, "max": 500, "step": 10, "value": CD["runs"]}]},
            {"type": "try", "label": "Open the model with only ten runs", "params": {"runs": 10, "lam": _LAM},
             "note": "The average is visibly rough: compare it with the default."},
        ]},
        {"id": "spread", "title": "How far single runs stray", "blocks": [
            {"type": "p", "text": "At time t each atom has survived with probability q = e^(−λt), independently of the others, so the "
                                  "number left in one run is a <b>binomial</b> count. Its standard deviation has a formula."},
            {"type": "eq", "text": "sd[N(t)] = √(N₀ q (1 − q)),     q = e^(−λt)",
             "where": [["q", "chance that one atom is still there at t"], ["N₀ q", "the expected count"]]},
            {"type": "figure", "figure": "chance-spread", "title": "Scatter across runs, over time",
             "caption": "The standard deviation across runs at each second, against the binomial formula."},
            {"type": "callout", "tone": "tip", "title": "Absolute and relative error",
             "text": "The scatter in atoms is largest at the half-life, but the scatter as a share of the count keeps growing: "
                     "late in the run few atoms are left, and each one that goes is a large part of the total."},
        ]},
        {"id": "averaging", "title": "More runs, smaller error", "blocks": [
            {"type": "p", "text": "Averaging R independent runs divides the standard deviation by √R. To halve the error of a Monte "
                                  "Carlo estimate you need four times as many runs; to gain one more decimal place, a hundred times."},
            {"type": "figure", "figure": "chance-convergence", "title": "Error against number of runs",
             "caption": "Log–log axes. The dashed line has slope −½, the 1/√runs law."},
            {"type": "quiz", "q": f"With {CD['runs']} runs the mean is off by at most {_C0['max_gap_to_theory']:.1f} atoms. "
                                  "How many runs would halve that, on average?",
             "options": [f"{2 * CD['runs']}", f"{4 * CD['runs']}", f"{10 * CD['runs']}"], "answer": 1,
             "why": "The error falls like 1/√R, so halving it needs √R to double, which means four times as many runs."},
        ]},
        {"id": "coins", "title": "The same law with coins", "blocks": [
            {"type": "p", "text": f"Tossing a fair coin is the simplest random experiment. After n tosses the share of heads is an "
                                  f"average of n independent 0s and 1s, so it settles on {M.COIN_P:g} at the same 1/√n rate: the "
                                  "<b>law of large numbers</b>."},
            {"type": "eq", "text": f"share of heads after n tosses ≈ {M.COIN_P:g} ± 1.96 √({M.COIN_P:g} · {1 - M.COIN_P:g} / n)   (95% band)",
             "where": [["n", "tosses so far"], ["1.96", "the multiplier for a 95% band under the normal approximation"]]},
            {"type": "figure", "figure": "chance-coins", "title": "Share of heads as tosses add up",
             "caption": "Change the number of tosses. The x-axis is logarithmic: each step to the right is ten times as many tosses.",
             "controls": [{"k": "n", "label": "Tosses", "min": 100, "max": 50000, "step": 100, "value": KD["n"]}]},
            {"type": "compare", "items": [
                {"title": "One run, few events", "points": ["Large relative scatter", "The rule is hard to see", "Early coin shares swing widely"]},
                {"title": "Many runs, many events", "points": ["Scatter shrinks like 1/√n", "The average reveals the rule",
                                                               "Coin shares settle into the band"]},
            ]},
        ]},
        {"id": "server", "title": "How the server computes it, and what it assumes", "blocks": [
            {"type": "steps", "items": [
                f"Seed a random generator (seed {CD['seed']} by default), so the same settings always give the same numbers.",
                f"Draw {_N0:,} × runs uniform numbers and turn each into a decay second with the formula above.",
                "Sort each run's decay times and count how many atoms remain at every whole second.",
                "Average across runs, take the standard deviation, and compare both with the formulas.",
            ]},
            {"type": "figure", "figure": "chance-check", "title": "The default experiment in numbers",
             "caption": "Measured values next to the formulas they should match."},
            {"type": "try", "label": "Open fast decay", "params": {"lam": _FAST, "runs": CD["runs"]},
             "note": f"λ = {_FAST:g}: a half-life of {math.log(2) / _FAST:.1f} s, and by t = {_TMAX // 5} s only "
                     f"{_pct(math.exp(-_FAST * (_TMAX // 5)), 1)} of the atoms are expected to remain."},
            {"type": "callout", "tone": "warn", "title": "Assumptions",
             "text": "Atoms decay independently, time is counted in whole seconds, and the random numbers come from a "
                     "pseudo-random generator: a deterministic sequence that behaves like chance for this purpose. The live view's "
                     "'Draw new random numbers' button changes the seed, which changes every run but not the averages' agreement "
                     "with theory."},
        ]},
    ],
    "takeaways": [
        "Monte Carlo simulation plays out a rule for single random events many times and reads off averages.",
        "Decay times come from one uniform number each: T = ⌈−ln(1 − u)/λ⌉.",
        "The average of many runs follows N₀e^(−λt); a single run scatters by the binomial √(N₀ q (1 − q)).",
        "Averaging R runs cuts the error like 1/√R: four times the runs for half the error.",
        f"The same law of large numbers makes the share of heads settle on {M.COIN_P:g}.",
    ],
    "glossary": [
        ["Monte Carlo", "Estimating a quantity by simulating the random process many times and averaging."],
        ["Decay constant λ", "Rate of decay per atom; the chance of decaying in a short time dt is about λ dt."],
        ["Half-life", "Time for half of the atoms to decay, ln 2 / λ."],
        ["Binomial distribution", "The count of successes among independent trials with the same probability."],
        ["Law of large numbers", "Averages of many independent draws settle on the expected value."],
        ["Seed", "The starting value of a random generator; the same seed gives the same sequence."],
        ["Memoryless", "The chance of an event in the next moment does not depend on how long one has waited."],
    ],
}, order=5)


# ===================================================================== Random walk (Lab 10): figures
WD = _defaults(M.walk)
W2 = _defaults(M.walk2d)
_WITHIN2 = math.erf(2 / math.sqrt(2))                 # normal probability of lying within 2 sd of the mean
_MOVES8 = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
_MSQ8 = sum(dx * dx + dy * dy for dx, dy in _MOVES8) / len(_MOVES8)


@functools.lru_cache(maxsize=256)
def _walk(p: float = WD["p"], steps: int = WD["steps"], walkers: int = WD["walkers"]) -> dict:
    return M.walk(p=p, steps=steps, walkers=walkers)


@functools.lru_cache(maxsize=64)
def _walk2d(steps: int = W2["steps"]) -> dict:
    return M.walk2d(steps=steps)


def _inside2(r) -> float:
    """Share of walkers whose final position is within two theoretical standard deviations of the theoretical mean."""
    m = r["summary"]
    vals, cnt = np.asarray(r["final_values"]), np.asarray(r["final_counts"])
    ok = np.abs(vals - m["mean_theory"]) <= 2 * m["sd_theory"] + 1e-9
    return float(cnt[ok].sum() / cnt.sum())


@figure("walk-step")
def _fig_walk_step(p):
    """One step from position x: right with probability p, left with 1 − p."""
    pr = p.num("p", WD["p"], 0, 1)
    r = _walk(pr)
    m = r["summary"]
    mu, var = 2 * pr - 1, 4 * pr * (1 - pr)
    return {
        "kind": "flow", "height": 220,
        "nodes": [{"id": "L", "x": 30, "y": 90, "w": 130, "h": 70, "title": "x − 1", "sub": "one step left"},
                  {"id": "X", "x": 235, "y": 90, "w": 130, "h": 70, "title": "x", "sub": "now", "hot": True},
                  {"id": "R", "x": 440, "y": 90, "w": 130, "h": 70, "title": "x + 1", "sub": "one step right"}],
        "edges": [{"from": "X", "to": "L", "label": f"1 − p = {1 - pr:.2f}", "width": 1.5 + 9 * (1 - pr), "flow": True, "color": "s2"},
                  {"from": "X", "to": "R", "label": f"p = {pr:.2f}", "width": 1.5 + 9 * pr, "flow": True, "color": "s1"}],
        "texts": [{"x": 300, "y": 34, "text": f"mean step 2p − 1 = {mu:+.2f}   ·   variance of a step 4p(1 − p) = {var:.2f}", "cls": "n-title"}],
        "notes": [f"After {m['steps']} steps the {WD['walkers']:,} simulated walkers average {m['mean']:+.1f} (theory "
                  f"{m['mean_theory']:+.1f}), with variance {m['var']:.1f} (theory {m['var_theory']:.1f}).",
                  "Every walker makes the same move, so there is no randomness at all." if var == 0 else
                  f"A single step has variance 4p(1 − p) = {var:.2f}" + (", the largest possible: a fair coin spreads walkers fastest."
                                                                         if pr == 0.5 else
                                                                         ", below a fair coin's 1, so these walkers spread more slowly.")],
    }


@figure("walk-paths")
def _fig_walk_paths(p):
    pr = p.num("p", WD["p"], 0, 1)
    r = _walk(pr)
    n, m = np.asarray(r["n"], float), r["summary"]
    pts = lambda y: np.column_stack([n, np.asarray(y, float)]).tolist()
    paths = [{"points": pts(row), "color": "neutral", "width": 1} for row in np.asarray(r["paths"])]
    paths += [{"points": pts(r["band_hi"]), "color": "s2", "dash": True, "width": 1.5, "name": "Theory ± 2 sd"},
              {"points": pts(r["band_lo"]), "color": "s2", "dash": True, "width": 1.5},
              {"points": pts(r["mean"]), "color": "s1", "width": 2.5, "name": f"Mean of all {WD['walkers']:,} walkers"}]
    return {
        "kind": "paths", "paths": paths, "equal": False, "xlabel": "Step", "ylabel": "Position", "height": 300,
        "notes": [f"After {m['steps']} steps the mean is {m['mean']:+.2f} (theory n(2p − 1) = {m['mean_theory']:+.2f}) and the "
                  f"variance {m['var']:.1f} (theory 4np(1 − p) = {m['var_theory']:.1f}).",
                  f"{_pct(_inside2(r), 1)} of the {WD['walkers']:,} walkers end inside the dashed ±2 sd lines."],
    }


@figure("walk-sqrt")
def _fig_walk_sqrt(p):
    """Typical distance from the start grows like √n."""
    steps = 4 * WD["steps"]
    r = _walk(steps=steps)
    n = np.asarray(r["n"])
    sd, th = np.sqrt(np.asarray(r["var"])), np.sqrt(np.asarray(r["var_theory"]))
    a, b = steps // 4, steps
    return {
        "kind": "lines", "x": n, "xlabel": "Step n", "ylabel": "Typical distance from the start", "ymin": 0,
        "series": [{"name": "Simulated: √(variance)", "color": "s1", "y": sd},
                   {"name": "Theory: √n", "color": "s2", "y": th, "dash": True}],
        "markers": [{"x": a, "label": f"n = {a}"}, {"x": b, "label": f"n = {b}"}],
        "notes": [f"After {a} steps the typical distance is {sd[a]:.1f}; after {b} it is {sd[b]:.1f}. Four times the steps, "
                  f"{sd[b] / sd[a]:.2f}× the distance.",
                  f"Theory says √n: {math.sqrt(a):.1f} and {math.sqrt(b):.1f}. A walker who always stepped the same way would be {b} away."],
    }


@figure("walk-hist")
def _fig_walk_hist(p):
    walkers = p.num("walkers", WD["walkers"], 100, 5000, integer=True)
    r = _walk(walkers=walkers)
    vals, cnt, nrm = np.asarray(r["final_values"], float), np.asarray(r["final_counts"], float), np.asarray(r["normal_expected"])
    j = int(np.argmax(np.abs(cnt - nrm)))
    m = r["summary"]
    return {
        "kind": "lines", "x": vals, "xlabel": f"Position after {m['steps']} steps", "ylabel": "Walkers", "ymin": 0,
        "series": [{"name": f"{walkers:,} simulated walkers", "color": "s1", "y": cnt, "area": True},
                   {"name": "Normal approximation", "color": "s2", "y": nrm, "dash": True}],
        "notes": [f"{_pct(_inside2(r), 1)} of walkers end within 2 sd (±{2 * m['sd_theory']:.1f}) of the start; a normal "
                  f"distribution puts {_pct(_WITHIN2, 1)} there.",
                  f"The largest gap between the simulated count and the curve is {abs(cnt[j] - nrm[j]):.0f} walkers, at position "
                  f"{vals[j]:.0f}: {_pct(abs(cnt[j] - nrm[j]) / nrm.max())} of the curve's height. Relative gaps shrink like "
                  "1/√walkers."],
    }


@figure("walk-drift")
def _fig_walk_drift(p):
    """Drift n(2p − 1) against spread 2√(4np(1 − p)) over a long walk."""
    pr = p.num("p", 0.6, 0.5, 0.9)
    steps = 1000
    r = _walk(pr, steps, 500)
    n = np.asarray(r["n"])
    drift, spread = np.asarray(r["mean_theory"]), 2 * np.sqrt(np.asarray(r["var_theory"]))
    first = (f"The drift overtakes twice the spread after n = 16p(1 − p)/(2p − 1)² = {16 * pr * (1 - pr) / (2 * pr - 1) ** 2:.0f} "
             "steps; from then on the start lies more than two standard deviations behind the centre of the crowd."
             if pr > 0.5 else "A fair walk has no drift: only the spread grows, like √n, and walkers stay balanced about the start.")
    cross = 16 * pr * (1 - pr) / (2 * pr - 1) ** 2 if pr > 0.5 else None
    return {
        "kind": "lines", "x": n, "xlabel": "Step n", "ylabel": "Positions",
        "series": [{"name": "Drift n(2p − 1)", "color": "s1", "y": drift},
                   {"name": "Twice the spread, 2√(4np(1 − p))", "color": "s2", "y": spread, "dash": True},
                   {"name": "Mean of 500 simulated walkers", "color": "s3", "y": r["mean"], "thin": True}],
        "markers": [{"x": cross, "label": "drift = 2 sd"}] if cross is not None and cross <= steps else [],
        "notes": [first, f"Drift grows like n and spread like √n: at step {steps} the drift is {drift[-1]:.0f} and twice the spread "
                         f"is {spread[-1]:.0f}."],
    }


@figure("walk-2d")
def _fig_walk_2d(p):
    steps = p.num("steps", W2["steps"], 50, 500, integer=True)
    r = _walk2d(steps)
    paths = np.asarray(r["paths"], float)
    end = paths[:, -1, :]
    dist = np.sqrt((end ** 2).sum(axis=1))
    m = r["summary"]
    far = int(np.argmax(dist))
    return {
        "kind": "paths", "equal": True, "height": 380,
        "paths": [{"points": path.tolist(), "color": _SERIES[k % 4], "width": 1.3} for k, path in enumerate(paths)],
        "dots": [{"x": 0, "y": 0, "color": "neutral", "label": "start"}]
                + [{"x": float(x), "y": float(y), "color": _SERIES[k % 4], "r": 3.5} for k, (x, y) in enumerate(end)],
        "notes": [f"After {steps} steps the {len(paths)} walkers shown end on average {dist.mean():.1f} places from the start; "
                  f"the furthest is {dist[far]:.1f} away.",
                  f"Over all {W2['walkers']} walkers the root-mean-square distance is {math.sqrt(m['msd']):.1f}, against "
                  f"√({M.WALK2D_MSD_SLOPE:g} n) = {math.sqrt(m['msd_theory']):.1f}."],
    }


@figure("walk-msd")
def _fig_walk_msd(p):
    r = _walk2d()
    m = r["summary"]
    return {
        "kind": "lines", "x": list(range(len(r["msd"]))), "xlabel": "Step n", "ylabel": "Mean squared distance", "ymin": 0,
        "series": [{"name": f"Simulated, {W2['walkers']} walkers", "color": "s1", "y": r["msd"]},
                   {"name": f"Theory {M.WALK2D_MSD_SLOPE:g} n", "color": "s2", "y": r["msd_theory"], "dash": True}],
        "notes": [f"After {m['steps']} steps the mean squared distance is {m['msd']:.0f} against {m['msd_theory']:.0f} in theory "
                  f"({(m['msd'] / m['msd_theory'] - 1) * 100:+.0f}%; with {W2['walkers']} walkers, sampling error of about "
                  f"1/√{W2['walkers']} = {_pct(1 / math.sqrt(W2['walkers']))} is expected).",
                  f"Why {M.WALK2D_MSD_SLOPE:g}: four of the eight moves have length 1 and four have length √2, so a step's mean squared "
                  f"length is (4 × 1 + 4 × 2) / 8 = {_MSQ8:g}."],
    }


# ===================================================================== Random walk: text
_W0 = _walk()["summary"]
_W6 = _walk(0.6)["summary"]
_LONG = 500                                       # the live view's 'Long walk' preset
_P51 = 0.51
_N51 = 16 * _P51 * (1 - _P51) / (2 * _P51 - 1) ** 2


def _sig2(x: float) -> float:
    """Round to two significant figures."""
    return float(f"{x:.2g}")

explainer("spread/walk", {
    "title": "How far a random walker gets: diffusion from coin flips",
    "kicker": "Modeling Lab · MC312 Lab 10",
    "summary": f"A walker tosses a coin at every step and moves one place left or right. After {WD['steps']} steps of a fair coin "
               f"a typical walker is only about {_W0['sd_theory']:.0f} places from the start, not {WD['steps']}: distance grows like "
               "the square root of time. That square-root law is diffusion, and the same arithmetic covers biased walks, "
               "two-dimensional walks and the random-walk model of prices.",
    "sections": [
        {"id": "question", "title": "The walker's question", "blocks": [
            {"type": "p", "text": "A random walk is the simplest model of movement without a plan: a molecule knocked about by its "
                                  "neighbours, a gambler's balance, a price that moves up or down each day. Each step is random, "
                                  "but the crowd of walkers spreads out in a precise way."},
            {"type": "cards", "items": [
                {"title": "Steps of ±1", "text": "Right with probability p, left with probability 1 − p, independently every step."},
                {"title": "Many walkers", "text": f"{WD['walkers']:,} walkers in the live view, each taking {WD['steps']} steps."},
                {"title": "Two questions", "text": "Where is the crowd centred, and how widely does it spread?"},
            ]},
            {"type": "p", "text": "The question: <i>after n steps, where do the walkers end up, and how fast does the crowd spread "
                                  "out?</i>"},
        ]},
        {"id": "step", "title": "One step", "blocks": [
            {"type": "p", "text": "Everything follows from one step. A step is +1 with probability p and −1 otherwise, so its mean "
                                  "is p − (1 − p) = 2p − 1 and its variance is 4p(1 − p)."},
            {"type": "figure", "figure": "walk-step", "title": "Left or right",
             "caption": "Change p. Arrow widths are the two probabilities; the notes check the formulas against the simulated walkers.",
             "controls": [{"k": "p", "label": "Step right with probability p", "min": 0, "max": 1, "step": 0.01, "value": WD["p"]}]},
        ]},
        {"id": "equations", "title": "Adding up the steps", "blocks": [
            {"type": "eq", "text": "Xₙ = ξ₁ + ξ₂ + … + ξₙ\nE[Xₙ]   = n (2p − 1)\nVar[Xₙ] = 4 n p (1 − p)",
             "where": [["Xₙ", "position after n steps, starting from 0"], ["ξₖ", "the k-th step: +1 with probability p, −1 otherwise"],
                       ["E, Var", "mean and variance over all possible walks"]]},
            {"type": "steps", "title": "Why the formulas hold", "items": [
                "The mean of a sum is the sum of the means: n steps of mean 2p − 1 give n(2p − 1).",
                "For independent steps the variances add too: n steps of variance 4p(1 − p) give 4np(1 − p).",
                "The standard deviation is the square root of the variance: 2√(np(1 − p)), which grows like √n.",
                "So the centre of the crowd moves like n, but its width grows only like √n.",
            ]},
        ]},
        {"id": "paths", "title": "Many walkers", "blocks": [
            {"type": "p", "text": f"Here are {M.WALK_PATHS_SHOWN} of the walkers. Each path is jagged and unpredictable; together "
                                  "they fill a region bounded by the theory's ±2 standard deviation curves."},
            {"type": "figure", "figure": "walk-paths", "title": "Walkers and the theory",
             "caption": "Change p. Thin lines are single walkers; the dashed curves are n(2p − 1) ± 2 standard deviations.",
             "controls": [{"k": "p", "label": "Step right with probability p", "min": 0, "max": 1, "step": 0.01, "value": WD["p"]}]},
            {"type": "quiz", "q": "With a fair coin (p = 0.5), where is a single walker most likely to be after many steps?",
             "options": ["Near the start, but typically about √n away", "Exactly at the start", "Far away, about n steps out"],
             "answer": 0, "why": f"The mean is 0, so the start is the single most likely spot, but the spread is √n. After "
                                 f"{WD['steps']} steps the simulated typical distance is {_W0['typical_distance']:.1f} against "
                                 f"√{WD['steps']} = {_W0['sd_theory']:.1f}."},
        ]},
        {"id": "sqrt", "title": "The square-root law", "blocks": [
            {"type": "p", "text": "Because the width grows like √n, a walker covers ground slowly: to get twice as far, it needs four "
                                  "times as many steps. This is the signature of <b>diffusion</b>, and it is why a drop of ink "
                                  "spreads through still water quickly at first and then ever more slowly."},
            {"type": "figure", "figure": "walk-sqrt", "title": "Typical distance against steps",
             "caption": f"A fair walk of {4 * WD['steps']} steps. Simulated √(variance) against √n."},
            {"type": "callout", "tone": "tip", "title": "Distance against displacement",
             "text": "After n steps the walker has travelled n steps in total, but its displacement from the start is typically "
                     "only √n. Most steps cancel each other out."},
        ]},
        {"id": "bell", "title": "Where walkers end up", "blocks": [
            {"type": "p", "text": "The final position is a sum of many independent steps, so by the <b>central limit theorem</b> its "
                                  "distribution is close to a normal (bell) curve with the mean and variance above. After an even "
                                  "number of steps only even positions can be reached, so the simulated counts sit on every other "
                                  "position and the normal curve is spread over that lattice spacing of 2."},
            {"type": "figure", "figure": "walk-hist", "title": "Final positions against the normal curve",
             "caption": "Change the number of walkers. More walkers make the counts closer to the curve.",
             "controls": [{"k": "walkers", "label": "Walkers", "min": 100, "max": 5000, "step": 100, "value": WD["walkers"]}]},
            {"type": "try", "label": "Open a long walk", "params": {"p": WD["p"], "steps": _LONG, "walkers": WD["walkers"]},
             "note": f"{_LONG} steps: the bell is √({_LONG}/{WD['steps']}) = {math.sqrt(_LONG / WD['steps']):.2f} times as wide."},
        ]},
        {"id": "drift", "title": "Drift against spread", "blocks": [
            {"type": "p", "text": "With a biased coin the crowd's centre moves at 2p − 1 per step while its width still grows like √n. "
                                  "However small the bias, the drift eventually wins: a straight line always overtakes a square root."},
            {"type": "eq", "text": "drift = n (2p − 1)     spread = 2√(4np(1 − p))\ndrift > spread   when   n > 16 p (1 − p) / (2p − 1)²",
             "where": [["spread", f"two standard deviations, the half-width that holds about {_pct(_WITHIN2)} of walkers"]]},
            {"type": "figure", "figure": "walk-drift", "title": "When the bias shows",
             "caption": "Change p. Before the marker, chance hides the bias; after it, the bias dominates.",
             "controls": [{"k": "p", "label": "Step right with probability p", "min": 0.5, "max": 0.9, "step": 0.01, "value": 0.6}]},
            {"type": "quiz", "q": f"A coin lands heads {_pct(_P51)} of the time. How many steps before the bias clearly shows "
                                  "(drift above 2 sd)?",
             "options": [f"About {_sig2(_N51 / 50):,.0f}", f"About {_sig2(_N51):,.0f}", f"About {_sig2(_N51 * 20):,.0f}"], "answer": 1,
             "why": f"16 × {_P51:g} × {1 - _P51:g} / {2 * _P51 - 1:.2f}² = {_N51:,.0f} steps. A small bias is invisible over short "
                    "stretches, which is why it takes long records to detect a small edge."},
            {"type": "try", "label": "Open a 60/40 walk", "params": {"p": 0.6, "steps": WD["steps"], "walkers": WD["walkers"]},
             "note": f"Mean {_W6['mean']:+.0f} after {WD['steps']} steps (theory {_W6['mean_theory']:+.0f})."},
        ]},
        {"id": "plane", "title": "Two dimensions", "blocks": [
            {"type": "p", "text": "On a grid each step goes to one of the eight neighbouring squares, chosen at random. The steps "
                                  "still add up independently, so the <b>mean squared distance</b> from the start grows in "
                                  "proportion to n, and the typical distance like √n, exactly as in one dimension."},
            {"type": "eq", "text": f"E[|Xₙ|²] = n · E[|ξ|²] = {M.WALK2D_MSD_SLOPE:g} n",
             "where": [["|Xₙ|²", "squared distance from the start after n steps"], ["E[|ξ|²]", "mean squared length of one step"]]},
            {"type": "figure", "figure": "walk-2d", "title": "Walkers in the plane",
             "caption": "Change the number of steps. Each colour is one walker; dots mark where they end.",
             "controls": [{"k": "steps", "label": "Steps", "min": 50, "max": 500, "step": 50, "value": W2["steps"]}]},
            {"type": "figure", "figure": "walk-msd", "title": "Mean squared distance grows linearly",
             "caption": "Simulated against the theory line. A straight line here is the definition of diffusion."},
        ]},
        {"id": "server", "title": "How the server computes it, and what it assumes", "blocks": [
            {"type": "p", "text": f"The server draws one uniform random number per walker per step from a seeded generator (seed "
                                  f"{WD['seed']} for one dimension, {W2['seed']} for two), turns each into a step, and takes "
                                  "cumulative sums to get every path. Means, variances and the histogram are computed across "
                                  "walkers at every step and compared with the formulas."},
            {"type": "compare", "items": [
                {"title": "What the formulas need", "points": ["Independent steps", "The same p at every step",
                                                               "Nothing else: no walls, no memory"]},
                {"title": "What the simulation adds", "points": ["Sampling noise that shrinks with more walkers",
                                                                 "A check that the formulas are right", "Pictures of single paths"]},
            ]},
            {"type": "callout", "tone": "warn", "title": "Assumptions",
             "text": "Every step is independent of the last, the walk has no walls or traps, and step sizes are fixed. Real "
                     "processes often have memory (momentum or mean reversion) or steps of varying size. The Time Series Lab "
                     "uses this walk as a null model: if stock prices were a random walk, what would their paths look like?"},
        ]},
    ],
    "takeaways": [
        "A walk is a sum of independent ±1 steps, so its mean is n(2p − 1) and its variance 4np(1 − p).",
        "Width grows like √n: four times the steps for twice the distance. That is diffusion.",
        "Final positions follow a bell curve, by the central limit theorem.",
        "Drift grows like n and spread like √n, so any bias eventually dominates, after about 16p(1 − p)/(2p − 1)² steps.",
        f"In two dimensions with eight neighbours, the mean squared distance is {M.WALK2D_MSD_SLOPE:g} n.",
    ],
    "glossary": [
        ["Random walk", "A path built by adding independent random steps."],
        ["Drift", "The average movement per step, 2p − 1 for a ±1 walk."],
        ["Variance", "The average squared distance from the mean; it adds across independent steps."],
        ["Diffusion", "Spreading in which mean squared distance grows in proportion to time."],
        ["Central limit theorem", "A sum of many independent random terms is close to normally distributed."],
        ["Mean squared distance", "The average of |Xₙ|² over walkers; for diffusion it grows linearly in n."],
        ["Null model", "A model of pure chance against which real data are compared."],
    ],
}, order=6)
