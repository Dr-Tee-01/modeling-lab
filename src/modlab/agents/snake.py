"""Snake as a headless environment, four agents, and Deep Q-learning.

Ported from the Snake-Game project (pygame + PyTorch DQN). The rules are the same:
a 32 x 24 board (the original 640 x 480 window in 20-pixel blocks), a 3-cell snake
starting in the middle heading right, +10 for food, -10 for dying, and death on a
wall, on any body cell including the current tail (the original checks collisions
before the tail moves), or when the game has lasted more than 100 x length steps.
Actions are relative to the heading: 0 straight, 1 turn right, 2 turn left.

Agents
------
random   picks a relative move uniformly: Lab 10's random walker with a body
greedy   steps toward the food, avoiding moves that kill it on the next step
bfs      shortest path to the food through free cells; when there is none, the
         move that keeps the most room (flood fill), so it does not trap itself
dqn      the original 11 -> 256 -> 3 Q-network, trained here as a background job

The DQN trainer keeps the original algorithm (per-step update, replay of up to
1,000 remembered steps after each game, epsilon = 80 - games, gamma 0.9, Adam
1e-3) but computes the replay targets in one batched pass instead of a Python
loop, which gives the same targets much faster.
"""
from __future__ import annotations

import itertools
import random as _random
import threading
import time
from collections import deque

import numpy as np

W, H = 32, 24
DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]            # right, down, left, up (clockwise)
DIR_NAMES = ["right", "down", "left", "up"]
ACTIONS = ["straight", "right", "left"]
FEATURES = ["danger straight", "danger right", "danger left",
            "moving left", "moving right", "moving up", "moving down",
            "food left", "food right", "food up", "food down"]


class SnakeEnv:
    def __init__(self, w: int = W, h: int = H, seed: int | None = None):
        self.w, self.h = w, h
        self.rng = _random.Random(seed)
        self.reset()

    def reset(self):
        self.d = 0                                     # heading right
        hx, hy = self.w // 2, self.h // 2
        self.snake = deque([(hx, hy), (hx - 1, hy), (hx - 2, hy)])
        self.body = set(self.snake)
        self.score, self.steps = 0, 0
        self.place_food()
        return self

    def place_food(self):
        free = self.w * self.h - len(self.body)
        if free <= 0:
            self.food = None
            return
        while True:
            f = (self.rng.randrange(self.w), self.rng.randrange(self.h))
            if f not in self.body:
                self.food = f
                return

    @property
    def head(self):
        return self.snake[0]

    def collides(self, p) -> bool:
        x, y = p
        return x < 0 or x >= self.w or y < 0 or y >= self.h or p in self.body

    def turn(self, action: int) -> int:
        return (self.d + (0, 1, -1)[action]) % 4

    def next_head(self, action: int):
        dx, dy = DIRS[self.turn(action)]
        return (self.head[0] + dx, self.head[1] + dy)

    def step(self, action: int):
        """Returns (reward, done, ate, cause). cause is None, 'wall', 'self', 'timeout' or 'won'.
        Same order as the original play_step: move, check collisions (tail still in
        place), check the step limit, then eat or move the tail."""
        self.steps += 1
        self.d = self.turn(action)
        nh = self.next_head(0)
        x, y = nh
        if x < 0 or x >= self.w or y < 0 or y >= self.h:
            return -10, True, False, "wall"
        if nh in self.body:
            return -10, True, False, "self"
        self.snake.appendleft(nh)
        self.body.add(nh)
        if self.steps > 100 * len(self.snake):
            return -10, True, False, "timeout"
        if nh == self.food:
            self.score += 1
            self.place_food()
            return 10, self.food is None, True, None if self.food is not None else "won"
        tail = self.snake.pop()
        self.body.discard(tail)
        return 0, False, False, None

    def state(self) -> np.ndarray:
        """The original agent's 11 binary features."""
        hx, hy = self.head
        d = self.d
        danger = [self.collides(self.next_head(a)) for a in (0, 1, 2)]
        moving = [d == 2, d == 0, d == 3, d == 1]      # left, right, up, down
        fx, fy = self.food if self.food else (hx, hy)
        food = [fx < hx, fx > hx, fy < hy, fy > hy]
        return np.array(danger + moving + food, dtype=np.int8)


# --------------------------------------------------------------------- simple agents
def act_random(env: SnakeEnv, rng: _random.Random, **_) -> int:
    return rng.randrange(3)


def act_greedy(env: SnakeEnv, rng: _random.Random, **_) -> int:
    fx, fy = env.food
    safe = [a for a in (0, 1, 2) if not env.collides(env.next_head(a))]
    if not safe:
        return 0
    dist = lambda a: abs(env.next_head(a)[0] - fx) + abs(env.next_head(a)[1] - fy)
    return min(safe, key=lambda a: (dist(a), a))


def _flood(env: SnakeEnv, start, blocked, cap: int | None = None) -> int:
    """Cells reachable from start; stops early once `cap` cells are found."""
    if start in blocked or not (0 <= start[0] < env.w and 0 <= start[1] < env.h):
        return 0
    seen, q = {start}, deque([start])
    while q and (cap is None or len(seen) < cap):
        x, y = q.popleft()
        for dx, dy in DIRS:
            n = (x + dx, y + dy)
            if 0 <= n[0] < env.w and 0 <= n[1] < env.h and n not in blocked and n not in seen:
                seen.add(n)
                q.append(n)
    return len(seen)


def act_bfs(env: SnakeEnv, rng: _random.Random, **_) -> int:
    """Shortest path to the food; if none, the safe move with the most room."""
    blocked = set(env.body)                             # the tail cell is deadly too (original rules)
    start, goal = env.head, env.food
    prev = {start: None}
    q = deque([start])
    found = False
    while q:
        cur = q.popleft()
        if cur == goal:
            found = True
            break
        for dx, dy in DIRS:
            n = (cur[0] + dx, cur[1] + dy)
            if 0 <= n[0] < env.w and 0 <= n[1] < env.h and n not in blocked and n not in prev:
                prev[n] = cur
                q.append(n)
    safe = [a for a in (0, 1, 2) if not env.collides(env.next_head(a))]
    if found:
        step = goal
        while prev[step] != start:
            step = prev[step]
        for a in (0, 1, 2):
            if env.next_head(a) == step:
                # only take the path if the snake keeps room after the move
                if _flood(env, step, blocked, cap=len(env.snake) + 1) > len(env.snake):
                    return a
                break
    if not safe:
        return 0
    return max(safe, key=lambda a: (_flood(env, env.next_head(a), blocked), -a))


# --------------------------------------------------------------------- DQN
class QNet:
    """Inference-only copy of LinearQNet(11, hidden, 3) in NumPy."""

    def __init__(self, w1, b1, w2, b2):
        self.w1, self.b1, self.w2, self.b2 = (np.asarray(a, dtype=np.float32) for a in (w1, b1, w2, b2))

    def q(self, s: np.ndarray) -> np.ndarray:
        h = np.maximum(0, self.w1 @ s.astype(np.float32) + self.b1)
        return self.w2 @ h + self.b2


def act_dqn(env: SnakeEnv, rng: _random.Random, net: QNet = None, **_) -> int:
    return int(np.argmax(net.q(env.state())))


AGENTS = {"random": act_random, "greedy": act_greedy, "bfs": act_bfs, "dqn": act_dqn}


def episode(agent: str, seed: int, net: QNet | None = None, max_steps: int = 3000, record: bool = True,
            w: int = W, h: int = H) -> dict:
    """Play one game. With record=True, returns everything needed to replay it."""
    env = SnakeEnv(w, h, seed)
    rng = _random.Random(seed + 1)
    act = AGENTS[agent]
    heads, ate, foods, actions, states, qs = [], [], [env.food], [], [], []
    cause = "limit"
    for _ in range(max_steps):
        s = env.state()
        a = act(env, rng, net=net)
        if record:
            states.append("".join(map(str, s)))
            actions.append(a)
            if agent == "dqn":
                qs.append([round(float(v), 3) for v in net.q(s)])
        _, done, got, why = env.step(a)
        if record:
            heads.append(env.next_head(0) if why in ("wall", "self") else env.head)
            ate.append(bool(got))
            if got and env.food is not None:
                foods.append(env.food)
        if done:
            cause = why
            break
    out = {"agent": agent, "seed": seed, "score": env.score, "steps": env.steps, "cause": cause, "length": len(env.snake)}
    if record:
        hx, hy = w // 2, h // 2
        n = max(1, len(actions))
        out.update(w=w, h=h, start=[[hx, hy], [hx - 1, hy], [hx - 2, hy]], heads=heads, ate=ate, foods=foods,
                   actions=actions, states=states, q=qs or None, features=FEATURES, action_names=ACTIONS,
                   score_by_move=np.cumsum(ate, dtype=int).tolist(),
                   summary={"action_share": [actions.count(i) / n for i in range(3)],
                            "steps_per_food": env.steps / env.score if env.score else None,
                            "died": cause in ("wall", "self", "timeout")})
    return out


def compare(agents: list[str], episodes: int, seed: int, net: QNet | None, max_steps: int = 3000) -> dict:
    return {name: summarise([episode(name, seed + i, net, max_steps, record=False) for i in range(episodes)])
            for name in agents}


def compare_summary(results: dict, episodes: int, seed: int) -> dict:
    """Headline numbers for a comparison: the ranking, the best single game, DQN vs BFS."""
    ranking = sorted(results, key=lambda a: -results[a]["mean"])
    top = max(results, key=lambda a: results[a]["max"])
    out = {"ranking": ranking, "best": ranking[0], "top_game": {"agent": top, "score": results[top]["max"]},
           "games": episodes * len(results), "seeds": [seed, seed + episodes - 1]}
    if "dqn" in results and "bfs" in results and results["bfs"]["mean"] > 0:
        out["dqn_vs_bfs"] = results["dqn"]["mean"] / results["bfs"]["mean"]
    return out


def summarise(runs: list[dict]) -> dict:
    """Score statistics and causes of death over a set of games."""
    scores = np.array([r["score"] for r in runs])
    steps = np.array([r["steps"] for r in runs])
    n = len(scores)
    return {
        "scores": scores.tolist(), "mean": float(scores.mean()), "median": float(np.median(scores)),
        "max": int(scores.max()), "sd": float(scores.std(ddof=1)) if n > 1 else 0.0,
        "ci95": float(1.96 * scores.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "mean_steps": float(steps.mean()), "steps_per_food": float(steps.sum() / max(1, scores.sum())),
        "causes": {c: sum(r["cause"] == c for r in runs) for c in ("wall", "self", "timeout", "limit", "won")},
    }


# --------------------------------------------------------------------- training + model registry
_MODELS: dict[str, dict] = {}
_LOCK = threading.Lock()
_COUNTER = itertools.count(1)


def models() -> list[dict]:
    with _LOCK:
        return [{k: v for k, v in m.items() if k not in ("net", "curve")} for m in sorted(_MODELS.values(), key=lambda m: -m["created"])]


def latest_curve() -> dict | None:
    m = get_model(None)
    return m and {"model": m["id"], **m["curve"]}


def get_model(model_id: str | None) -> dict | None:
    with _LOCK:
        if model_id:
            return _MODELS.get(model_id)
        return max(_MODELS.values(), key=lambda m: m["created"], default=None)


def train_dqn(params: dict, progress) -> dict:
    """The original Snake-Game DQN, trained for `games` games."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        raise RuntimeError("PyTorch is not installed. Install it with: pip install -e '.[rnn]'")

    games, hidden, lr, gamma = params["games"], params["hidden"], params["lr"], params["gamma"]
    explore, seed, batch, max_memory = params["explore_games"], params["seed"], 1000, 100_000
    torch.manual_seed(seed)
    rng = _random.Random(seed)
    model = nn.Sequential(nn.Linear(11, hidden), nn.ReLU(), nn.Linear(hidden, 3))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    memory = deque(maxlen=max_memory)

    def train_step(states, actions, rewards, nexts, dones):
        s = torch.as_tensor(np.asarray(states), dtype=torch.float32)
        n = torch.as_tensor(np.asarray(nexts), dtype=torch.float32)
        a = torch.as_tensor(np.asarray(actions), dtype=torch.long)
        r = torch.as_tensor(np.asarray(rewards), dtype=torch.float32)
        d = torch.as_tensor(np.asarray(dones), dtype=torch.float32)
        pred = model(s)
        with torch.no_grad():                         # the original's per-sample loop, batched
            q_new = r + gamma * model(n).max(dim=1).values * (1 - d)
        target = pred.detach().clone()
        target[torch.arange(len(a)), a] = q_new
        loss = lossf(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return float(loss.detach())

    scores, means, epsilons, steps_per_game, losses = [], [], [], [], []
    record, total, t0 = 0, 0, time.perf_counter()
    env = SnakeEnv(seed=seed)
    for g in range(games):
        env.reset()
        eps = max(0, explore - g)
        done, last_loss = False, 0.0
        while not done:
            s = env.state()
            if rng.randint(0, 200) < eps:
                a = rng.randrange(3)
            else:
                with torch.no_grad():
                    a = int(model(torch.as_tensor(s, dtype=torch.float32)).argmax())
            reward, done, _, _ = env.step(a)
            s2 = env.state()
            train_step([s], [a], [reward], [s2], [float(done)])
            memory.append((s, a, reward, s2, float(done)))
        mini = rng.sample(list(memory), batch) if len(memory) > batch else list(memory)
        last_loss = train_step(*zip(*mini))
        scores.append(env.score)
        total += env.score
        record = max(record, env.score)
        means.append(total / (g + 1))
        epsilons.append(eps / 200)
        steps_per_game.append(env.steps)
        losses.append(last_loss)
        if g % 5 == 0 or g == games - 1:
            progress((g + 1) / games, f"Game {g + 1}/{games} · score {env.score} · record {record}")

    w1, b1 = model[0].weight.detach().numpy(), model[0].bias.detach().numpy()
    w2, b2 = model[2].weight.detach().numpy(), model[2].bias.detach().numpy()
    net = QNet(w1, b1, w2, b2)
    model_id = f"dqn-{next(_COUNTER)}"
    last = scores[-50:]
    entry = {"id": model_id, "games": games, "hidden": hidden, "lr": lr, "gamma": gamma, "explore_games": explore,
             "seed": seed, "record": record, "mean_last50": float(np.mean(last)), "seconds": time.perf_counter() - t0,
             "created": time.time(), "net": net,
             "epsilon_start": explore / 200,
             "curve": {"scores": scores, "mean": means, "epsilon": epsilons, "steps": steps_per_game, "loss": losses,
                       "roll20": [float(np.mean(scores[max(0, i - 19):i + 1])) for i in range(len(scores))],
                       "first20_mean": float(np.mean(scores[:20])), "last20_mean": float(np.mean(scores[-20:]))}}
    with _LOCK:
        _MODELS[model_id] = entry
        for old in sorted(_MODELS.values(), key=lambda m: m["created"])[:-5]:
            _MODELS.pop(old["id"], None)
    return {"model": {k: v for k, v in entry.items() if k not in ("net", "curve")}, **entry["curve"]}


def weights(model_id: str | None) -> dict | None:
    m = get_model(model_id)
    if m is None:
        return None
    n = m["net"]
    return {"id": m["id"], "w1": n.w1, "b1": n.b1, "w2": n.w2, "b2": n.b2, "features": FEATURES, "actions": ACTIONS}


# --------------------------------------------------------------------- reference data for the interface
AGENT_INFO = {
    "random": {"label": "Random walker", "short": "Random", "decides": "coin flip", "sees": "11 features",
               "what": "Picks straight, right or left at random: Lab 10's random walk with a body.",
               "insight": "A random walker almost never finds food: it hits a wall within about a hundred moves."},
    "greedy": {"label": "Greedy", "short": "Greedy", "decides": "closer to food?", "sees": "11 features",
               "what": "Steps toward the food, skipping moves that would kill it on the next step.",
               "insight": "Greedy grows quickly but has no plan: as the body gets long it walks into dead ends it cannot see."},
    "bfs": {"label": "BFS path-finder", "short": "BFS", "decides": "shortest path", "sees": "full grid",
            "what": "Searches the shortest free path to the food; if none is safe, takes the move with the most room.",
            "insight": "BFS sees the whole board and checks it keeps room to move, so it survives far longer than the others."},
    "dqn": {"label": "Deep Q-network", "short": "DQN", "decides": "argmax Q", "sees": "11 features",
            "what": "Your trained 11 → hidden → 3 network: it scores each move from 11 yes/no features and takes the best.",
            "insight": "The DQN plays from 11 features only. Watch the danger chips light up just before it turns."},
}
CAUSES = {"wall": "hit a wall", "self": "hit itself", "timeout": "ran out of moves", "limit": "reached the step cap",
          "won": "filled the board"}
REPLAY_CAP = 3000


def catalog() -> dict:
    """Everything the Agents pages show about the game itself."""
    return {"board": {"w": W, "h": H, "start_length": 3}, "features": FEATURES, "actions": ACTIONS,
            "directions": DIR_NAMES, "agents": AGENT_INFO, "causes": CAUSES,
            "rules": {"food_reward": 10, "death_reward": -10, "step_limit_per_length": 100, "replay_cap": REPLAY_CAP},
            "dqn": {"inputs": len(FEATURES), "outputs": len(ACTIONS), "hidden": 256, "gamma": 0.9, "lr": 0.001,
                    "explore_games": 80, "epsilon_divisor": 200, "replay_batch": 1000, "memory": 100_000}}


# --------------------------------------------------------------------- interactive games, played on the server
class PlaySession:
    def __init__(self, sid: str, seed: int):
        self.id, self.env = sid, SnakeEnv(seed=seed)
        self.cause, self.compared, self.agreed = None, 0, 0
        self.touched = time.time()


_PLAY: dict[str, PlaySession] = {}
_PLAY_LOCK = threading.Lock()
_PLAY_MAX, _PLAY_TTL = 500, 3600
_PLAY_RECORD = {"score": 0}


def _suggest(sess: PlaySession):
    m = get_model(None)
    if m is None or sess.cause:
        return None, None
    q = m["net"].q(sess.env.state())
    return m, [round(float(v), 3) for v in q]


def play_view(sess: PlaySession) -> dict:
    env = sess.env
    m, q = _suggest(sess)
    hint = None
    if q is not None:
        a = int(np.argmax(q))
        hint = {"action": a, "direction": env.turn(a), "model": m["id"]}
    return {"id": sess.id, "w": env.w, "h": env.h, "snake": [list(p) for p in env.snake],
            "food": list(env.food) if env.food else None, "heading": env.d, "score": env.score,
            "length": len(env.snake), "steps": env.steps, "over": sess.cause is not None, "cause": sess.cause,
            "features": env.state().tolist(), "q": q, "hint": hint,
            "agreement": {"compared": sess.compared, "agreed": sess.agreed,
                          "share": sess.agreed / sess.compared if sess.compared else None},
            "record": _PLAY_RECORD["score"]}


def play_new(seed: int | None = None) -> dict:
    with _PLAY_LOCK:
        now = time.time()
        for k in [k for k, v in _PLAY.items() if now - v.touched > _PLAY_TTL]:
            _PLAY.pop(k)
        while len(_PLAY) >= _PLAY_MAX:
            _PLAY.pop(min(_PLAY.values(), key=lambda v: v.touched).id)
        sid = f"g{next(_COUNTER)}{_random.getrandbits(32):08x}"
        sess = _PLAY[sid] = PlaySession(sid, seed if seed is not None else _random.getrandbits(31))
    return play_view(sess)


def play_step(sid: str, direction: int | None) -> dict | None:
    """Advance a game one move. `direction` is absolute (0 right, 1 down, 2 left, 3 up) or None to keep going;
    turning straight back into the body is ignored, as in most Snake games."""
    with _PLAY_LOCK:                                   # one move at a time, even if a client double-sends
        sess = _PLAY.get(sid)
        if sess is None:
            return None
        env = sess.env
        sess.touched = time.time()
        if sess.cause is None:
            a = 0
            if direction is not None and direction != (env.d + 2) % 4:
                a = {env.d: 0, (env.d + 1) % 4: 1, (env.d + 3) % 4: 2}[direction]
            _, q = _suggest(sess)
            if q is not None:
                sess.compared += 1
                sess.agreed += int(np.argmax(q)) == a
            _, done, _, why = env.step(a)
            if done:
                sess.cause = why
            _PLAY_RECORD["score"] = max(_PLAY_RECORD["score"], env.score)
        return play_view(sess)
