"""/api/spread: the six MC312 models, solved on every request."""
from __future__ import annotations

from flask import Blueprint

from ..spread import models as M
from .api import ApiError, choice, endpoint, flag, num
from .cache import memo

bp = Blueprint("spread", __name__, url_prefix="/api/spread")
MAX_CELLS = 4_000_000     # cap on simulated cells per request, to bound memory and time


def _budget(cells: int, what: str):
    if cells > MAX_CELLS:
        raise ApiError(f"{what} is {cells:,}; the limit per request is {MAX_CELLS:,}.", "too_large")


@bp.get("/dosing")
@endpoint
def dosing():
    mec, mtc = num("mec", 10, 0.5, 100), num("mtc", 20, 1, 200)
    if mtc <= mec:
        raise ApiError("The maximum safe level (mtc) must be above the minimum effective level (mec).", "invalid_parameter")
    return M.dosing(
        model=choice("model", "one", ["one", "two"]),
        dose_mg=num("dose", 100, 1, 2000), tau_h=num("tau", 8, 1, 48), half_life_h=num("half_life", 22, 0.5, 200),
        absorbed=num("absorbed", 0.12, 0.01, 1), ka=num("ka", 0.6, 0.01, 5), load=num("load", 1, 1, 20),
        mec=mec, mtc=mtc, t_end=num("hours", 240, 24, 720),
    )


@bp.get("/bass")
@endpoint
def bass():
    return M.bass(p=num("p", 0.03, 0.0005, 0.5), q=num("q", 0.38, 0, 2),
                  market=num("market", 100, 1, 10000), years=num("years", 25, 2, 60))


@bp.get("/sir")
@endpoint
def sir():
    return M.sir(beta=num("beta", 1.66, 0.05, 5), alpha=num("alpha", 0.5, 0.02, 3),
                 vaccinate_pct=num("vaccinate", 0, 0, 50), lockdown=flag("lockdown"),
                 lock_start=num("lock_start", 3, 0, 30), lock_days=num("lock_days", 10, 1, 60),
                 lock_cut_pct=num("lock_cut", 50, 0, 100), days=num("days", 20, 5, 120))


@memo(maxsize=4)
def _fit():
    return M.sir_fit()


@bp.route("/sir/fit", methods=["GET", "POST"])
@endpoint
def sir_fit():
    out, cached = _fit()
    return dict(out, _cached=cached)


@bp.get("/rumour")
@endpoint
def rumour():
    return M.rumour(beta=num("beta", 0.3, 0.01, 2), gamma=num("gamma", 0.1, 0.01, 1),
                    alpha=num("alpha", 0, 0, 0.5), newcomers=num("newcomers", 0, 0, 50), days=num("days", 100, 10, 365))


@bp.get("/decay")
@endpoint
def decay():
    return M.decay(lam=num("lam", 0.05, 0.005, 1), runs=num("runs", 100, 1, 1000, integer=True),
                   seed=num("seed", 9, 0, 2**31 - 1, integer=True))


@bp.get("/coins")
@endpoint
def coins():
    return M.coins(n=num("n", 5000, 10, 100000, integer=True), sequences=num("sequences", 6, 1, 12, integer=True),
                   seed=num("seed", 9, 0, 2**31 - 1, integer=True))


@bp.get("/walk")
@endpoint
def walk():
    steps, walkers = num("steps", 200, 1, 2000, integer=True), num("walkers", 2000, 10, 20000, integer=True)
    _budget(steps * walkers, "steps x walkers")
    return M.walk(p=num("p", 0.5, 0, 1), steps=steps, walkers=walkers, seed=num("seed", 10, 0, 2**31 - 1, integer=True))


@bp.get("/walk2d")
@endpoint
def walk2d():
    steps, walkers = num("steps", 500, 10, 5000, integer=True), num("walkers", 250, 10, 5000, integer=True)
    _budget(steps * walkers, "steps x walkers")
    return M.walk2d(walkers=walkers, steps=steps, seed=num("seed", 11, 0, 2**31 - 1, integer=True))
