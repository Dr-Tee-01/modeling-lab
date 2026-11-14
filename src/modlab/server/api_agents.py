"""/api/agents: Snake as a live environment, four agents, DQN training as a background job."""
from __future__ import annotations

from flask import Blueprint, request

from ..agents import snake as S
from .api import ApiError, choice, choices, endpoint, num
from .jobs import register

bp = Blueprint("agents", __name__, url_prefix="/api/agents")
AGENTS = list(S.AGENTS)


def _net(agent: str, model_id: str | None):
    if agent != "dqn":
        return None
    m = S.get_model(model_id or None)
    if m is None:
        raise ApiError("No trained DQN yet. Train one first (POST /api/jobs with kind snake_dqn).", "no_model", 409)
    return m["net"]


@bp.get("/snake/models")
@endpoint
def models():
    """Trained DQN models held in memory (newest first), with the newest one's learning curve."""
    return {"models": S.models(), "latest": S.latest_curve()}


@bp.get("/snake/episode")
@endpoint
def episode():
    """Play one game with an agent and return everything needed to replay it."""
    agent = choice("agent", "bfs", AGENTS)
    model_id = str(request.args.get("model", "")) or None
    out = S.episode(agent, num("seed", 1, 0, 2**31 - 1, integer=True), _net(agent, model_id),
                    max_steps=num("max_steps", 3000, 50, 10000, integer=True))
    if agent == "dqn":
        out["model"] = S.get_model(model_id)["id"]
    return out


@bp.get("/snake/weights")
@endpoint
def weights():
    """The trained Q-network's weights, for running it in the browser."""
    model_id = str(request.args.get("model", "")) or None
    w = S.weights(model_id)
    if w is None:
        raise ApiError("No trained DQN yet.", "no_model", 409)
    return w


def _train_params(body: dict) -> dict:
    def n(name, default, lo, hi, integer=False):
        v = body.get(name, default)
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise ApiError(f"Parameter '{name}' must be a number.", "invalid_parameter")
        if not lo <= v <= hi:
            raise ApiError(f"Parameter '{name}' must be between {lo:g} and {hi:g}.", "out_of_range")
        return int(v) if integer else v
    return {"games": n("games", 200, 10, 1500, True), "hidden": n("hidden", 256, 8, 512, True),
            "lr": n("lr", 0.001, 1e-5, 0.05), "gamma": n("gamma", 0.9, 0.5, 0.999),
            "explore_games": n("explore_games", 80, 0, 500, True), "seed": n("seed", 7, 0, 2**31 - 1, True)}


@register("snake_dqn", params=_train_params)
def _train(params, progress):
    return S.train_dqn(params, progress)


def _compare_params(body: dict) -> dict:
    agents = body.get("agents", ["random", "greedy", "bfs"])
    if isinstance(agents, str):
        agents = [a for a in agents.split(",") if a]
    if not agents or any(a not in AGENTS for a in agents):
        raise ApiError(f"agents must be a list from {', '.join(AGENTS)}.", "invalid_parameter")
    try:
        episodes, seed = int(body.get("episodes", 30)), int(body.get("seed", 1))
    except (TypeError, ValueError):
        raise ApiError("episodes and seed must be whole numbers.", "invalid_parameter")
    if not 3 <= episodes <= 200:
        raise ApiError("episodes must be between 3 and 200.", "out_of_range")
    model = body.get("model") or None
    if "dqn" in agents:
        m = S.get_model(model)
        if m is None:
            raise ApiError("Train the DQN before comparing it.", "no_model", 409)
        model = m["id"]
    return {"agents": list(dict.fromkeys(agents)), "episodes": episodes, "seed": seed, "model": model}


@register("snake_compare", params=_compare_params)
def _compare(params, progress):
    net = S.get_model(params["model"])["net"] if params["model"] else None
    out, total, done = {}, len(params["agents"]) * params["episodes"], 0
    for a in params["agents"]:
        runs = []
        for i in range(params["episodes"]):
            runs.append(S.episode(a, params["seed"] + i, net, record=False))
            done += 1
            if done % 3 == 0 or done == total:
                progress(done / total, f"{a}: game {i + 1}/{params['episodes']}")
        out[a] = S.summarise(runs)
    return {"agents": out, "episodes": params["episodes"], "seed": params["seed"], "model": params["model"]}
