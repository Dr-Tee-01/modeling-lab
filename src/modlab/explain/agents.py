"""Explainers for the Agents section (the Snake game and its AI). Figures reuse modlab.agents.snake.

Every board, score and count in these pages comes from playing real games with the real
environment and agents. No DQN is trained here (that needs PyTorch and seconds of compute);
where a learned agent is needed the figures use a Q-table over the same 11 features, trained
with the same per-step rule, the same epsilon schedule and the same rewards in a fraction of a
second. Games are memoised, so each figure is computed once per setting.
"""
from __future__ import annotations

import functools
import random
from collections import Counter, defaultdict, deque, namedtuple
from statistics import NormalDist

import numpy as np

from ..agents import snake as S
from .registry import explainer, figure

# ===================================================================== reference data
CAT = S.catalog()
BOARD, RULES, DQN = CAT["board"], CAT["rules"], CAT["dqn"]
W, H, START = BOARD["w"], BOARD["h"], BOARD["start_length"]
FOOD, DIE, LIMIT, CAP = RULES["food_reward"], RULES["death_reward"], RULES["step_limit_per_length"], RULES["replay_cap"]
FEATURES, ACTIONS, CAUSES, INFO = CAT["features"], CAT["actions"], CAT["causes"], CAT["agents"]
NF, NA = len(FEATURES), len(ACTIONS)
DANGER = [i for i, f in enumerate(FEATURES) if f.startswith("danger")]
MOVING = [i for i, f in enumerate(FEATURES) if f.startswith("moving")]
FOODF = [i for i, f in enumerate(FEATURES) if f.startswith("food")]
# food sits left/right/level and up/down/level of the head, but never on it
N_VECTORS = 2 ** len(DANGER) * len(MOVING) * (3 * 3 - 1)
N_PARAMS = DQN["inputs"] * DQN["hidden"] + DQN["hidden"] + DQN["hidden"] * DQN["outputs"] + DQN["outputs"]
EPS0 = DQN["explore_games"] / DQN["epsilon_divisor"]

# settings of the illustrations themselves (all shown to the reader through these names)
SEED = 7                 # the board used for single-game figures (the Watch page's default)
TAB_GAMES = 150          # games of tabular Q-learning
TAB_ALPHA = 0.1          # step size of the tabular update
TAB_BUDGET = 150_000     # moves; a run that plays well is stopped here so a figure stays fast
CMP_SEED = 1             # first board of the comparison (the Compare page's default)
CMP_CAP = 800            # moves per game in the comparison figures
CMP_GAMES = 6            # boards per agent in the comparison, by default
CI_GAMES = 60            # greedy games for the confidence-interval figure
NOISE_GAMES = 30         # boards for the frozen-policy spread
ROLL = 20                # games in a running mean

# cell values on a board grid (palette "cat": value v is drawn in BOARD_COLORS[v]); body blue, head orange,
# food green, free or safe cells grey, deadly cells red, a planned path amber
EMPTY, BODY, HEAD, FOODC, FREE, DEADLY, PATH, SEP = range(8)
BOARD_COLORS = ["neutral", "s1", "s2", "s3", "neutral", "bad", "s4", "neutral"]


def _legend(*extra):
    """Legend for a board grid: the snake and food, then the overlays this figure uses."""
    return [{"name": "Body", "color": "s1"}, {"name": "Head", "color": "s2"}, {"name": "Food", "color": "s3"}] + \
        [{"name": n, "color": c} for n, c in extra]
AGENT_COLOR = {"random": "neutral", "greedy": "s3", "bfs": "s1", "table": "s2"}   # as on the live pages
TABLE_LABEL = f"Q-table ({NF} features)"
label = lambda a: TABLE_LABEL if a == "table" else INFO[a]["label"]
short = lambda a: "Q-table" if a == "table" else INFO[a]["short"]


# ===================================================================== games, recorded move by move
Frame = namedtuple("Frame", "snake food d feats action reward ate")
POW = 1 << np.arange(NF - 1, -1, -1)


def _sidx(env) -> int:
    return int(env.state().astype(np.int64) @ POW)


def _fidx(feats: str) -> int:
    return int(feats, 2)


@functools.lru_cache(maxsize=48)
def _qtable(games: int = TAB_GAMES, explore: int = DQN["explore_games"], gamma: float = DQN["gamma"], seed: int = SEED) -> dict:
    """Tabular Q-learning on the 11 features: the DQN's per-step update with a table instead of a network,
    the same epsilon schedule (random move when randint(0, divisor) < explore - game) and the same rewards."""
    Q = [[0.0] * NA for _ in range(2 ** NF)]
    rng, env, div = random.Random(seed), S.SnakeEnv(seed=seed), DQN["epsilon_divisor"]
    scores, steps, rewards, total, stopped, randoms = [], [], Counter(), 0, None, 0
    for g in range(games):
        env.reset()
        eps, done, s = max(0, explore - g), False, _sidx(env)
        while not done:
            if rng.randint(0, div) < eps:
                a = rng.randrange(NA)
                randoms += 1
            else:
                q = Q[s]
                a = q.index(max(q))
            r, done, _, _ = env.step(a)
            s2 = _sidx(env)
            target = r if done else r + gamma * max(Q[s2])
            Q[s][a] += TAB_ALPHA * (target - Q[s][a])
            rewards[r] += 1
            total += 1
            s = s2
        scores.append(env.score)
        steps.append(env.steps)
        if total >= TAB_BUDGET and g + 1 < games:
            stopped = g + 1
            break
    Q = np.array(Q)
    return {"Q": Q, "scores": scores, "steps": steps, "rewards": dict(rewards), "total": total, "stopped": stopped,
            "visited": int((np.abs(Q).sum(axis=1) > 0).sum()), "randoms": randoms, "games": len(scores),
            "explore": explore, "gamma": gamma}


def _table() -> dict:
    """The default run: the original exploration and discount."""
    return _qtable(TAB_GAMES, DQN["explore_games"], DQN["gamma"], SEED)


def _table_act(env, rng):
    return int(np.argmax(_table()["Q"][_sidx(env)]))


_ACT = {"random": S.act_random, "greedy": S.act_greedy, "bfs": S.act_bfs, "table": _table_act}


@functools.lru_cache(maxsize=160)
def _game(agent: str, seed: int, cap: int = CAP) -> dict:
    """One game with the same seeding as snake.episode, keeping the full board before every move."""
    env, rng, act = S.SnakeEnv(seed=seed), random.Random(seed + 1), _ACT[agent]
    frames, cause = [], "limit"
    for _ in range(cap):
        snake, food, d = tuple(env.snake), env.food, env.d
        feats = "".join(map(str, env.state()))
        a = act(env, rng)
        r, done, ate, why = env.step(a)
        frames.append(Frame(snake, food, d, feats, a, r, ate))
        if done:
            cause = why
            break
    return {"agent": agent, "seed": seed, "frames": frames, "score": env.score, "steps": env.steps, "cause": cause,
            "length": len(env.snake)}


def _env_at(fr: Frame) -> S.SnakeEnv:
    env = S.SnakeEnv(seed=0)
    env.snake, env.body, env.d, env.food = deque(fr.snake), set(fr.snake), fr.d, fr.food
    return env


def _on_board(c) -> bool:
    return 0 <= c[0] < W and 0 <= c[1] < H


def _region(start, blocked) -> set:
    if not _on_board(start) or start in blocked:
        return set()
    seen, q = {start}, deque([start])
    while q:
        x, y = q.popleft()
        for dx, dy in S.DIRS:
            n = (x + dx, y + dy)
            if _on_board(n) and n not in blocked and n not in seen:
                seen.add(n)
                q.append(n)
    return seen


def _path(start, goal, blocked):
    prev, q = {start: None}, deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            out = []
            while cur != start:
                out.append(cur)
                cur = prev[cur]
            return out[::-1]
        for dx, dy in S.DIRS:
            n = (cur[0] + dx, cur[1] + dy)
            if _on_board(n) and n not in blocked and n not in prev:
                prev[n] = cur
                q.append(n)
    return None


def _cells(fr: Frame, overlay: dict | None = None) -> list:
    cells = [EMPTY] * (W * H)
    for x, y in fr.snake[1:]:
        cells[y * W + x] = BODY
    for (x, y), v in (overlay or {}).items():
        if _on_board((x, y)):
            cells[y * W + x] = v
    if fr.food:
        cells[fr.food[1] * W + fr.food[0]] = FOODC
    hx, hy = fr.snake[0]
    cells[hy * W + hx] = HEAD
    return cells


def _read(feats: str) -> dict:
    word = lambda i: FEATURES[i].split()[1]
    return {"danger": [word(i) for i in DANGER if feats[i] == "1"],
            "heading": next((word(i) for i in MOVING if feats[i] == "1"), "?"),
            "food": [word(i) for i in FOODF if feats[i] == "1"]}


def _and(words) -> str:
    words = list(words)
    return "none" if not words else words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def _rw(r: int) -> str:
    return f"{r:+d}" if r else "0"


def _cause(c: str) -> str:
    return CAUSES.get(c, c)


def _foods(g: dict) -> list:
    """Successive food positions of a game."""
    out = [g["frames"][0].food]
    for fr in g["frames"][1:]:
        if fr.food != out[-1]:
            out.append(fr.food)
    return out


def _roll(v, k=ROLL):
    v = np.asarray(v, dtype=float)
    return [float(v[max(0, i - k + 1):i + 1].mean()) for i in range(len(v))]


# The recorded game used on several pages: greedy on board SEED (a few milliseconds to play).
_G = _game("greedy", SEED)
_GN = len(_G["frames"])
_Q_MOVE = next((i + 1 for i, f in enumerate(_G["frames"]) if i >= _GN // 3 and "1" not in f.feats[:len(DANGER)]), _GN // 2)
_PLAY_MOVE = next((i + 1 for i, f in enumerate(_G["frames"]) if i >= _GN // 3 and "1" in f.feats[:len(DANGER)]), _GN // 2)


# ===================================================================== Play: figures
@figure("snake-play-board")
def _fig_board(p):
    """The board before one move, with the three cells the actions lead to marked safe or deadly."""
    m = p.num("move", _PLAY_MOVE, 1, _GN, integer=True)
    fr = _G["frames"][m - 1]
    env = _env_at(fr)
    over = {env.next_head(a): DEADLY if env.collides(env.next_head(a)) else FREE for a in range(NA)}
    rd = _read(fr.feats)
    end = f" and the game ended: it {_cause(_G['cause'])}" if m == _GN else ""
    return {
        "kind": "grid", "cols": W, "rows": H, "cells": _cells(fr, over), "palette": "cat", "colors": BOARD_COLORS, "height": 380,
        "legend": _legend(("Safe next cell", "neutral"), ("Deadly next cell", "bad")),
        "notes": [f"Move {m} of {_GN}: the snake is {len(fr.snake)} cells long and heading {rd['heading']}. "
                  f"The marked cells are where straight, right and left would take the head.",
                  f"Features {fr.feats}: danger {_and(rd['danger'])}; food {_and(rd['food'])}. "
                  f"Greedy went {ACTIONS[fr.action]}{end}."],
    }


@figure("snake-play-loop")
def _fig_loop(p):
    """One turn of the agent-environment loop with the real values from that move."""
    m = p.num("move", _PLAY_MOVE, 1, _GN, integer=True)
    fr = _G["frames"][m - 1]
    nxt = _G["frames"][m].feats if m < _GN else "terminal"
    what = "food" if fr.ate else "death" if fr.reward == DIE else "nothing"
    n_food = sum(f.ate for f in _G["frames"])
    n_zero = sum(f.reward == 0 for f in _G["frames"])
    return {
        "kind": "flow", "height": 270,
        "nodes": [
            {"id": "s", "x": 15, "y": 30, "w": 175, "h": 84, "title": "State sₜ", "value": fr.feats, "sub": f"{NF} features"},
            {"id": "pi", "x": 215, "y": 30, "w": 170, "h": 84, "title": "Agent π", "value": ACTIONS[fr.action], "sub": "greedy rule", "hot": True},
            {"id": "env", "x": 410, "y": 30, "w": 175, "h": 84, "title": "Environment", "value": f"{W} × {H}", "sub": "moves the snake"},
            {"id": "r", "x": 410, "y": 175, "w": 175, "h": 74, "title": "Reward rₜ₊₁", "value": _rw(fr.reward), "sub": what},
            {"id": "s2", "x": 215, "y": 175, "w": 170, "h": 74, "title": "Next state sₜ₊₁", "value": nxt},
        ],
        "edges": [
            {"from": "s", "to": "pi", "label": "observe", "flow": True},
            {"from": "pi", "to": "env", "label": f"aₜ = {ACTIONS[fr.action]}", "flow": True, "color": "s2"},
            {"from": "env", "to": "r", "color": "s3"},
            {"from": "env", "to": "s2"},
            {"from": "s2", "to": "s", "label": "t → t + 1", "curve": [-40, 10]},
        ],
        "notes": [f"At move {m} the agent saw {fr.feats}, chose {ACTIONS[fr.action]} and got {_rw(fr.reward)} ({what}).",
                  f"Over the whole game {n_food} moves paid {FOOD:+d}, "
                  f"{1 if _G['cause'] in ('wall', 'self', 'timeout') else 0} paid {DIE:+d} and {n_zero} of {_GN} "
                  f"({100 * n_zero / _GN:.0f}%) paid nothing."],
    }


@functools.lru_cache(maxsize=1)
def _vectors():
    games = [_game("greedy", s) for s in range(1, 21)]
    seen = Counter(f.feats for g in games for f in g["frames"])
    return seen, sum(seen.values()), len(games)


@figure("snake-play-states")
def _fig_states(p):
    seen, moves, n = _vectors()
    top, top_n = seen.most_common(1)[0]
    situations = W * H * len(MOVING) * (W * H - 1)
    return {
        "kind": "stats",
        "items": [{"label": "Feature vectors possible", "value": f"{N_VECTORS}",
                   "note": f"2^{len(DANGER)} danger × {len(MOVING)} headings × {3 * 3 - 1} food directions"},
                  {"label": f"Seen in {n} greedy games", "value": f"{len(seen)}", "note": f"over {moves:,} moves"},
                  {"label": "Head, heading and food alone", "value": f"{situations:,}", "note": "before counting the body"},
                  {"label": "Most common vector", "value": top, "note": f"{100 * top_n / moves:.0f}% of all moves", "tone": "warn"}],
        "notes": [f"Even ignoring the body, each feature vector stands on average for {situations // N_VECTORS:,} different situations.",
                  f"The {n} games used {len(seen)} of the {N_VECTORS} vectors, and the 10 most common of them account for "
                  f"{100 * sum(k for _, k in seen.most_common(10)) / moves:.0f}% of all moves."],
    }


@functools.lru_cache(maxsize=1)
def _alias():
    """A pair of boards with identical features: straight leads into a sealed pocket on one, open space on the other."""
    trapped = None
    games = [_game("greedy", s) for s in range(1, 21)]
    for g in games:
        if g["cause"] != "self":
            continue
        for fr in g["frames"][-40:]:
            env = _env_at(fr)
            if fr.feats[DANGER[0]] == "0" and fr.action == 0:
                room = len(_region(env.next_head(0), set(fr.snake)))
                if room < len(fr.snake):
                    trapped = (g, fr, room)
                    break
        if trapped:
            break
    g0, fr0, room0 = trapped
    best = None
    for g in games:
        for fr in g["frames"]:
            if fr.feats == fr0.feats and fr is not fr0:
                room = len(_region(_env_at(fr).next_head(0), set(fr.snake)))
                if best is None or room > best[2]:
                    best = (g, fr, room)
                if room > W * H // 2:
                    break
        if best and best[2] > W * H // 2:
            break
    return trapped, best


@figure("snake-play-alias")
def _fig_alias(p):
    (g0, fr0, room0), (g1, fr1, room1) = _alias()
    left = _cells(fr0, {c: DEADLY for c in _region(_env_at(fr0).next_head(0), set(fr0.snake))})
    right = _cells(fr1, {c: FREE for c in _region(_env_at(fr1).next_head(0), set(fr1.snake))})
    cells = []
    for y in range(H):
        cells += left[y * W:(y + 1) * W] + [SEP] + right[y * W:(y + 1) * W]
    left_moves = len(g0["frames"]) - g0["frames"].index(fr0) - 1
    return {
        "kind": "grid", "cols": 2 * W + 1, "rows": H, "cells": cells, "palette": "cat", "colors": BOARD_COLORS, "height": 260,
        "legend": _legend(("Reachable: sealed pocket", "bad"), ("Reachable: open space", "neutral")),
        "notes": [f"Both boards read {fr0.feats}: straight is safe, the food is {_and(_read(fr0.feats)['food'])}.",
                  f"Left (board {g0['seed']}): straight leads into {room0} free cells for a {len(fr0.snake)}-cell snake, and it died "
                  f"{left_moves} moves later. Right (board {g1['seed']}{', earlier in the same game' if g1 is g0 else ''}): "
                  f"the same move opens onto {room1} free cells."],
    }


@figure("snake-play-qvalues")
def _fig_qvalues(p):
    m = p.num("move", _Q_MOVE, 1, _GN, integer=True)
    fr = _G["frames"][m - 1]
    T = _table()
    q = T["Q"][_fidx(fr.feats)]
    best = int(np.argmax(q))
    lo, hi = float(q.min()), float(q.max())
    fill = lambda v: 0.15 + 0.85 * ((v - lo) / (hi - lo) if hi > lo else 1)
    nodes = [{"id": "x", "x": 10, "y": 75, "w": 160, "h": 84, "title": "Input s", "value": fr.feats, "sub": f"{NF} features"},
             {"id": "f", "x": 205, "y": 75, "w": 160, "h": 84, "title": "Q-function", "value": "Q(s, ·)",
              "sub": f"DQN: {DQN['inputs']} → {DQN['hidden']} → {DQN['outputs']}", "hot": True}]
    edges = [{"from": "x", "to": "f", "flow": True}]
    for a in range(NA):
        nodes.append({"id": f"a{a}", "x": 410, "y": 8 + 78 * a, "w": 180, "h": 62, "title": ACTIONS[a], "value": f"{q[a]:.2f}",
                      "fill": fill(q[a]), "fill_color": "s2", "hot": a == best})
        edges.append({"from": "f", "to": f"a{a}", "width": 5 if a == best else 1.5, "flow": a == best, "color": "s2" if a == best else None})
    unseen = not np.abs(q).sum()
    zeros = [ACTIONS[a] for a in range(NA) if q[a] == 0]
    return {
        "kind": "flow", "height": 245, "nodes": nodes, "edges": [{k: v for k, v in e.items() if v is not None} for e in edges],
        "notes": [f"For {fr.feats} the learned values are " + ", ".join(f"{ACTIONS[a]} {q[a]:.2f}" for a in range(NA))
                  + f": the agent would go {ACTIONS[best]}" + (" (the vector was never visited, so every value is still 0 and the tie goes to straight)." if unseen
                                                              else f"; greedy went {ACTIONS[fr.action]}."),
                  (f"The value{'s' if len(zeros) > 1 else ''} for {_and(zeros)} {'are' if len(zeros) > 1 else 'is'} still exactly 0, the starting value: "
                   "once one move looked good here, acting greedily never tried the others. " if zeros and not unseen else "")
                  + f"The table was trained by Q-learning for {T['games']} games ({T['total']:,} moves) and visited "
                    f"{T['visited']} of the {N_VECTORS} possible vectors."],
    }


# ===================================================================== Play: text
explainer("agents/play", {
    "title": "Snake as a decision problem: states, actions, rewards and what the network sees",
    "kicker": "Reinforcement Learning Lab · Snake",
    "summary": "Every move of Snake is a choice between going straight, turning right and turning left. Written as a Markov "
               f"decision process, the game has a state, {NA} actions and a reward, which is all a learning agent needs. "
               f"The Q-network does not see the board: it sees {NF} yes/no features, and this page shows what they keep and what they lose.",
    "sections": [
        {"id": "game", "title": "The game in numbers", "blocks": [
            {"type": "p", "text": f"The board is {W} × {H} cells, as in the original game. The snake starts "
                                  f"{START} cells long in the middle, heading right. Each piece of food adds one cell and the next "
                                  "food appears on a random free cell."},
            {"type": "cards", "items": [
                {"title": f"{FOOD:+d} for food", "text": "The only positive reward. Score counts the pieces eaten."},
                {"title": f"{DIE:+d} for dying", "text": "Hitting a wall or any body cell, including the tail, ends the game."},
                {"title": f"{LIMIT} × length moves", "text": f"A game that lasts longer than {LIMIT} moves per cell of snake ends as a timeout, "
                                                           "so an agent cannot score by circling forever."},
                {"title": "0 otherwise", "text": "Most moves earn nothing. The agent must link a reward to the moves that led to it."},
            ]},
        ]},
        {"id": "mdp", "title": "A Markov decision process", "blocks": [
            {"type": "p", "text": "A <b>Markov decision process</b> (MDP) describes a world in which an agent repeatedly observes a "
                                  "<b>state</b>, picks an <b>action</b>, and receives a <b>reward</b> and a new state. "
                                  "<i>Markov</i> means the next state depends only on the current state and action, not on how the "
                                  "agent got there. The full board (body cells, heading, food) plus the move count has this property; the "
                                  "11 features, as we will see, do not quite."},
            {"type": "eq", "text": "sₜ₊₁ ~ P( · | sₜ, aₜ)\nrₜ₊₁ = R(sₜ, aₜ, sₜ₊₁)\naₜ = π(sₜ)",
             "where": [["sₜ", "the state before move t"],
                       ["aₜ", f"the action at move t: one of {_and(ACTIONS)}"],
                       ["P", "the rules: where the snake goes, and where the next food appears (random)"],
                       ["R", f"the reward: {FOOD:+d}, {DIE:+d} or 0"],
                       ["π", "the policy: the agent's rule for choosing an action"]]},
            {"type": "figure", "figure": "snake-play-loop", "title": "One turn of the loop",
             "caption": "A real move from greedy's game on board " + str(SEED) + ". Move the slider to step through the game.",
             "controls": [{"k": "move", "label": "Move", "min": 1, "max": _GN, "step": 1, "value": _PLAY_MOVE}]},
            {"type": "steps", "title": "What happens in one move", "items": [
                "The environment reports the state (here, the 11 features).",
                "The policy picks straight, right or left.",
                "The head moves one cell in the new direction.",
                f"If the new cell is a wall or body, the game ends with {DIE:+d}.",
                f"If it is food, the snake grows and the reward is {FOOD:+d}; otherwise the tail moves up and the reward is 0.",
            ]},
        ]},
        {"id": "actions", "title": "Three relative actions", "blocks": [
            {"type": "p", "text": "The actions are relative to the heading: <b>straight</b>, <b>right</b> (clockwise) and "
                                  "<b>left</b>. A snake heading up that turns right goes right; heading down, the same "
                                  "action takes it left on the screen. There is no fourth action, because reversing would drive "
                                  "the head straight into its own neck."},
            {"type": "quiz", "q": "Why do relative actions make learning easier than four absolute directions (up, down, left, right)?",
             "options": ["They make the board smaller", "One of the four absolute moves is always fatal, and the same rule (\"turn away from danger on the right\") works in every heading",
                         "The network trains faster with fewer outputs, for no other reason"],
             "answer": 1, "why": "Relative actions remove the always-fatal reverse move, and they let one learned rule apply in all "
                                 "four headings: danger on the right means the same thing whichever way the snake faces."},
        ]},
        {"id": "features", "title": "The 11 features", "blocks": [
            {"type": "p", "text": "The original project does not give the network the board. It gives it 11 yes/no answers "
                                  "computed from the board:"},
            {"type": "list", "items": [
                f"<b>Danger</b> ({len(DANGER)}): would going {_and(FEATURES[i].split()[1] for i in DANGER)} hit a wall or the body on the very next move?",
                f"<b>Heading</b> ({len(MOVING)}): which way the snake is moving; exactly one is 1.",
                f"<b>Food</b> ({len(FOODF)}): is the food to the left, right, above or below the head? Up to two are 1.",
            ]},
            {"type": "figure", "figure": "snake-play-board", "title": "What the features describe",
             "caption": "The cells marked around the head are where the three actions lead: safe or deadly on the next step. Those three answers are the danger features.",
             "controls": [{"k": "move", "label": "Move", "min": 1, "max": _GN, "step": 1, "value": _PLAY_MOVE}]},
            {"type": "eq", "text": "s = (d_straight, d_right, d_left, m_left, m_right, m_up, m_down, f_left, f_right, f_up, f_down) ∈ {0, 1}¹¹",
             "where": [["d", "danger one step ahead in that relative direction"], ["m", "current heading, one-hot"],
                       ["f", "food direction relative to the head, on each axis"]]},
        ]},
        {"id": "lost", "title": "What the features leave out", "blocks": [
            {"type": "p", "text": "The features are cheap and they generalise: the network sees the same vector wherever the "
                                  "snake is on the board. The price is that very different boards look identical."},
            {"type": "figure", "figure": "snake-play-states", "title": "How much the features compress"},
            {"type": "p", "text": "The danger features only look <b>one cell</b> ahead. They cannot tell an open board from a "
                                  "pocket the snake has walled off with its own body, and they say nothing about how far "
                                  "away the food is."},
            {"type": "figure", "figure": "snake-play-alias", "title": "Same features, different fate",
             "caption": "Two real boards with identical feature vectors. Shaded cells are everything reachable by going straight: a sealed pocket on the left, open space on the right."},
            {"type": "callout", "tone": "warn", "title": "The features break the Markov property",
             "text": "Because different boards share a feature vector, the same action can lead to very different futures from "
                     "the same observed state. Whatever the agent learns is an average over all the boards behind each vector, "
                     "so no agent that sees only these features can avoid every self-made trap."},
        ]},
        {"id": "qnet", "title": "From features to a move", "blocks": [
            {"type": "p", "text": "A <b>Q-function</b> gives each action a number: the total future reward the agent expects if "
                                  "it takes that action now and plays well afterwards. The policy is then simply to take the "
                                  "action with the largest Q-value."},
            {"type": "eq", "text": "Q(s, ·) = W₂ · ReLU(W₁ s + b₁) + b₂\na* = argmaxₐ Q(s, a)",
             "where": [["s", f"the {NF} features as a column of 0s and 1s"],
                       ["W₁, b₁", f"{DQN['hidden']} × {DQN['inputs']} weights and {DQN['hidden']} biases of the hidden layer"],
                       ["ReLU", "max(0, x), applied to each hidden unit"],
                       ["W₂, b₂", f"{DQN['outputs']} × {DQN['hidden']} weights and {DQN['outputs']} biases: one output per action"],
                       ["a*", "the move the agent makes"]]},
            {"type": "p", "text": f"In total the network has {N_PARAMS:,} numbers to learn. Since the input can only take "
                                  f"{N_VECTORS} values, the same job can be done by a table with one row per feature vector "
                                  f"and one column per action. The figure uses such a table, trained by Q-learning "
                                  f"(the Train page explains how) for {TAB_GAMES} games."},
            {"type": "figure", "figure": "snake-play-qvalues", "title": "Three numbers, one move",
             "caption": "The Q-values for the features at this move; the largest wins. Move the slider to see other states.",
             "controls": [{"k": "move", "label": "Move", "min": 1, "max": _GN, "step": 1, "value": _Q_MOVE}]},
            {"type": "compare", "items": [
                {"title": "Q-table", "points": [f"{N_VECTORS} × {NA} entries, one per vector and action", "Each entry learned separately",
                                                "An unvisited vector has no information"]},
                {"title": "Q-network", "points": [f"{N_PARAMS:,} weights shared by all inputs", "Learning one state shifts the values of similar ones",
                                                  "Scales to inputs too large for a table, such as raw pixels"]},
            ]},
        ]},
        {"id": "you", "title": "Playing against the network", "blocks": [
            {"type": "p", "text": "On the live page the server runs the game and, after every move, evaluates your latest trained "
                                  "network on the same 11 features. With the hint on you can see which way it would go, and the "
                                  "page counts how often your moves agree with it."},
            {"type": "quiz", "q": "The hint says \"straight\", the danger features are all 0, and the snake is heading into a narrow gap between its own body and the wall. Should you trust it?",
             "options": ["Yes: no danger means the move is safe for good", "Only for the next move: the features cannot see whether the gap is a dead end",
                         "No: the network ignores the danger features"],
             "answer": 1, "why": "Danger features look one cell ahead. The network has no way to know whether the gap leads to open "
                                 "space or a pocket smaller than the snake."},
            {"type": "try", "label": "Play with the network's hint on", "params": {"speed": 6, "hint": True},
             "note": "A slow game, so you can compare your choice with the network's at every move. Train a network first."},
            {"type": "try", "label": "Play at full speed without help", "params": {"speed": 20, "hint": False}},
        ]},
    ],
    "takeaways": [
        f"Snake is a Markov decision process with {NA} relative actions and rewards of {FOOD:+d}, {DIE:+d} and 0.",
        f"The agent sees {NF} binary features (danger, heading, food direction), not the board.",
        f"There are only {N_VECTORS} possible feature vectors, so very different boards look identical.",
        "Danger features look one move ahead: they cannot see a trap made by the snake's own body.",
        "A Q-function scores each action by expected future reward; the agent takes the highest score.",
    ],
    "glossary": [
        ["MDP", "Markov decision process: states, actions, transition probabilities and rewards."],
        ["State", "What the agent observes before choosing. Here, 11 yes/no features."],
        ["Policy π", "The rule that maps a state to an action."],
        ["Reward", "The number the environment returns after each move."],
        ["Q-value", "Expected total future reward for taking an action in a state and playing well afterwards."],
        ["Feature aliasing", "Different underlying situations that produce the same observed state."],
        ["ReLU", "The function max(0, x), used between the network's layers."],
    ],
}, order=1)


# ===================================================================== Watch: figures
def _bfs_game():
    return _game("bfs", SEED)


def _choices(fr: Frame) -> dict:
    """What every scripted agent would do in this exact position, and what each action leads to."""
    env = _env_at(fr)
    info = []
    for a in range(NA):
        c = env.next_head(a)
        fatal = env.collides(c)
        info.append({"fatal": fatal, "dist": abs(c[0] - fr.food[0]) + abs(c[1] - fr.food[1]),
                     "room": 0 if fatal else len(_region(c, set(fr.snake)))})
    return {"greedy": S.act_greedy(env, random.Random(0)), "bfs": S.act_bfs(env, random.Random(0)), "info": info}


def _last_disagreement() -> int:
    for i in range(_GN - 1, -1, -1):
        if _choices(_G["frames"][i])["bfs"] != _G["frames"][i].action:
            return i + 1
    return _GN


_DECIDE_MOVE = _last_disagreement()


@functools.lru_cache(maxsize=1)
def _no_return():
    """First move from which every available move leads into fewer free cells than the snake is long."""
    k = _GN
    for i in range(_GN - 1, -1, -1):
        fr = _G["frames"][i]
        env = _env_at(fr)
        rooms = [len(_region(env.next_head(a), set(fr.snake))) for a in range(NA)]
        if max(rooms) >= len(fr.snake):
            break
        k = i + 1
    return k


_TRAP_BACK = min(30, max(1, _GN - _no_return() + 1))


REPLAY_AGENTS = ["bfs", "greedy", "random"]


@figure("snake-watch-replay")
def _fig_replay(p):
    agent = p.choice("agent", "bfs", REPLAY_AGENTS)
    pct = p.num("pct", 25, 0, 100)
    g = _game(agent, SEED)
    n = len(g["frames"])
    i = min(n, max(1, round(pct / 100 * n)))
    fr = g["frames"][i - 1]
    path = _path(fr.snake[0], fr.food, set(fr.snake))
    over = {c: PATH for c in (path or [])[:-1]}
    manh = abs(fr.snake[0][0] - fr.food[0]) + abs(fr.snake[0][1] - fr.food[1])
    took = _env_at(fr).next_head(fr.action)
    if not path:
        plan = "There is no free path to the food" + (", so BFS takes the move that keeps the most free cells reachable."
                                                      if agent == "bfs" else f"; {short(agent)} went {ACTIONS[fr.action]}.")
    elif agent == "bfs":
        plan = (f"BFS's shortest free path to the food is {len(path)} moves; the straight-line distance is {manh}."
                + ("" if took == path[0] else " It does not take the path's first step, because that step would leave "
                                              "less room than the snake is long."))
    else:
        plan = (f"The shortest free path to the food is {len(path)} moves (straight-line distance {manh}). "
                f"{short(agent)} went {ACTIONS[fr.action]}, " + ("its first step." if took == path[0] else "off that path."))
    return {
        "kind": "grid", "cols": W, "rows": H, "cells": _cells(fr, over), "palette": "cat", "colors": BOARD_COLORS, "height": 380,
        "legend": _legend(("Shortest free path", "s4")),
        "notes": [f"{INFO[agent]['label']}, move {i} of {n}: length {len(fr.snake)}, score {len(fr.snake) - START}. "
                  f"The game ended when the snake {_cause(g['cause'])}.", plan],
    }


@figure("snake-watch-decide")
def _fig_decide(p):
    m = p.num("move", _DECIDE_MOVE, 1, _GN, integer=True)
    fr = _G["frames"][m - 1]
    ch = _choices(fr)
    nodes, edges = [], []
    for j, (a, sub) in enumerate((("random", "each move ⅓"), ("greedy", "closest to food"), ("bfs", "shortest path, keeps room"))):
        nodes.append({"id": a, "x": 15, "y": 12 + 88 * j, "w": 170, "h": 64, "title": INFO[a]["short"], "sub": sub})
    for a in range(NA):
        inf = ch["info"][a]
        nodes.append({"id": f"a{a}", "x": 400, "y": 12 + 88 * a, "w": 185, "h": 64, "title": ACTIONS[a],
                      "value": "fatal" if inf["fatal"] else f"{inf['dist']} from food",
                      "sub": "" if inf["fatal"] else f"{inf['room']} cells of room",
                      "hot": a in (fr.action, ch["bfs"])})
        edges.append({"from": "random", "to": f"a{a}", "width": 1, "color": "neutral"})
    edges.append({"from": "greedy", "to": f"a{fr.action}", "width": 4, "flow": True, "color": "s3", "label": "greedy"})
    edges.append({"from": "bfs", "to": f"a{ch['bfs']}", "width": 4, "flow": True, "color": "s1", "label": "BFS",
                  **({"curve": [0, 30]} if ch["bfs"] == fr.action else {})})
    gi, bi = ch["info"][fr.action], ch["info"][ch["bfs"]]
    same = ch["bfs"] == fr.action
    return {
        "kind": "flow", "height": 275, "nodes": nodes, "edges": edges,
        "notes": [f"Move {m} of {_GN} on board {SEED}, snake length {len(fr.snake)}: greedy goes {ACTIONS[fr.action]} "
                  f"({gi['dist']} cells from the food, {gi['room']} cells of room)"
                  + (", and BFS agrees." if same else f"; BFS goes {ACTIONS[ch['bfs']]} ({bi['dist']} from the food, {bi['room']} cells of room)."),
                  "Greedy compares only the distance to the food; BFS also checks that the snake keeps room to move after the step."],
    }


@figure("snake-watch-flood")
def _fig_flood(p):
    back = p.num("back", _TRAP_BACK, 1, min(30, _GN - 1), integer=True)
    i = _GN - back
    fr = _G["frames"][i]
    env = _env_at(fr)
    reg = _region(env.next_head(fr.action), set(fr.snake))
    tight = len(reg) < len(fr.snake)
    k = _no_return()
    frk = _G["frames"][k - 1]
    return {
        "kind": "grid", "cols": W, "rows": H, "cells": _cells(fr, {c: DEADLY if tight else FREE for c in reg}),
        "palette": "cat", "colors": BOARD_COLORS, "height": 380,
        "legend": _legend(("Reachable, smaller than the snake", "bad"), ("Reachable, enough room", "neutral")),
        "notes": [f"{back} move{'s' if back > 1 else ''} before the end: going {ACTIONS[fr.action]} leads into {len(reg)} free cells "
                  f"(shaded) for a snake {len(fr.snake)} cells long" + (": not enough room." if tight else "."),
                  f"From move {k} on, every move led into fewer free cells than the snake's length: the game was lost "
                  f"{_GN - k + 1} moves before the snake {_cause(_G['cause'])}, while the danger features at that move still showed "
                  f"{_and(ACTIONS[j] for j, i in enumerate(DANGER) if frk.feats[i] == '0')} as safe."],
    }


@figure("snake-watch-score")
def _fig_score(p):
    games = {a: _game(a, SEED) for a in ("random", "greedy", "bfs")}
    n = max(len(g["frames"]) for g in games.values())
    every = max(1, n // 300)
    x = list(range(0, n + 1, every))
    series = []
    for a, g in games.items():
        cum = np.concatenate([[0], np.cumsum([f.ate for f in g["frames"]])])
        series.append({"name": INFO[a]["label"], "color": AGENT_COLOR[a], "y": [int(cum[t]) if t < len(cum) else None for t in x]})
    b = games["bfs"]
    return {
        "kind": "lines", "x": x, "series": series, "xlabel": "Move", "ylabel": "Score (food eaten)", "ymin": 0,
        "notes": [" · ".join(f"{INFO[a]['short']}: {g['score']} food in {g['steps']} moves, then it {_cause(g['cause'])}" for a, g in games.items()) + ".",
                  f"BFS was still playing when the replay cap of {CAP} moves stopped it." if b["cause"] == "limit"
                  else f"BFS ended its game on move {b['steps']}."],
    }


def _pace(g):
    """Moves taken for each food, and the straight-line distance from the head when that food appeared."""
    moves, dist, last, start = [], [], 0, g["frames"][0]
    for i, fr in enumerate(g["frames"]):
        if fr.ate:
            moves.append(i + 1 - last)
            dist.append(abs(start.snake[0][0] - start.food[0]) + abs(start.snake[0][1] - start.food[1]))
            last = i + 1
            if i + 1 < len(g["frames"]):
                start = g["frames"][i + 1]
    return moves, dist


@figure("snake-watch-pace")
def _fig_pace(p):
    bm, bd = _pace(_bfs_game())
    gm, gd = _pace(_G)
    n = len(bm)
    x = list(range(1, n + 1))
    pad = lambda v: _roll(v, 10) + [None] * (n - len(v))
    head = lambda v: float(np.mean(v[:20]))
    tail = lambda v: float(np.mean(v[-20:]))
    return {
        "kind": "lines", "x": x, "xlabel": "Food number", "ylabel": "Moves to reach it (10-food average)", "ymin": 0,
        "series": [{"name": "BFS: moves taken", "color": "s1", "y": pad(bm)},
                   {"name": "BFS: straight-line distance", "color": "s1", "y": pad(bd), "dash": True, "thin": True},
                   {"name": "Greedy: moves taken", "color": "s3", "y": pad(gm)}],
        "notes": [f"BFS's first 20 food took {head(bm):.1f} moves each against a straight-line distance of {head(bd):.1f}; "
                  f"its last 20 took {tail(bm):.1f} against {tail(bd):.1f}.",
                  f"Greedy took {np.mean(gm):.1f} moves per food over its {len(gm)} food, close to the straight-line {np.mean(gd):.1f}: "
                  "it never detours, which is fast until the body is in the way."],
    }


@functools.lru_cache(maxsize=1)
def _random_stats():
    runs = [_game("random", s) for s in range(1, 201)]
    return int(np.median([r["steps"] for r in runs])), sum(r["score"] for r in runs), len(runs), Counter(r["cause"] for r in runs)


@figure("snake-watch-random")
def _fig_random(p):
    med, food, n, causes = _random_stats()
    walk = W // 2  # moves from the start to the right-hand wall going straight
    return {
        "kind": "stats",
        "items": [{"label": "Boards played", "value": str(n), "note": "random walker"},
                  {"label": "Median game", "value": f"{med} moves", "note": f"straight from the start reaches the wall in {walk}"},
                  {"label": "Food eaten in all", "value": str(food), "note": f"{food / n:.2f} per game", "tone": "warn"},
                  {"label": "Ended at a wall", "value": f"{100 * causes['wall'] / n:.0f}%", "note": f"{causes['self']} games ended on its own body"}],
        "notes": [f"Over {n} boards the random walker lasted a median of {med} moves and ate {food} food in total.",
                  f"{100 * causes['wall'] / n:.0f}% of its games ended at a wall: {NA - 1} of every {NA} moves are turns, so the head wanders until it meets an edge."],
    }


# ===================================================================== Watch: text
_RS = _random_stats()
explainer("agents/watch", {
    "title": "Three ways to play without learning: random, greedy and breadth-first search",
    "kicker": "Reinforcement Learning Lab · Snake",
    "summary": "Before training anything it helps to know what simple rules can do. A random walker sets the floor, a greedy "
               "agent heads straight for the food, and a breadth-first search plans a path over the whole board and checks "
               "it keeps room to move. Replays of real games show how each one decides and why each one dies.",
    "sections": [
        {"id": "baselines", "title": "Why scripted agents", "blocks": [
            {"type": "p", "text": "A <b>baseline</b> is a simple agent whose score a learned agent must beat to be worth "
                                  "training. The three here need no training at all; they are a few lines of code each."},
            {"type": "cards", "items": [{"title": INFO[a]["label"], "text": INFO[a]["what"]} for a in ("random", "greedy", "bfs")]},
            {"type": "p", "text": f"Random ignores the board. Greedy uses the same one-step danger check as the features, plus the "
                                  "exact distance to the food. BFS reads the full board."},
        ]},
        {"id": "random", "title": "The random walker", "blocks": [
            {"type": "p", "text": f"The random walker picks one of the {NA} relative moves with equal probability. It never "
                                  "reverses into itself, but it does not look where it is going."},
            {"type": "figure", "figure": "snake-watch-random", "title": "How a random walker does"},
            {"type": "p", "text": "Its games are short because a turn is as likely as going straight: the head wanders, "
                                  "and the edge of the board is never far away. It is the floor every other agent must clear."},
        ]},
        {"id": "greedy", "title": "Greedy: closest to the food", "blocks": [
            {"type": "p", "text": "Greedy throws out the moves that would kill it on the next step, then takes the one that "
                                  "brings the head closest to the food, measured in grid steps (the <b>Manhattan distance</b>)."},
            {"type": "eq", "text": "a = argmin over safe a of |x′ₐ − x_food| + |y′ₐ − y_food|",
             "where": [["(x′ₐ, y′ₐ)", "the cell the head would move to with action a"],
                       ["safe", "the cell is on the board and not part of the body"]]},
            {"type": "figure", "figure": "snake-watch-decide", "title": "Three agents, one position",
             "caption": "For one position of greedy's game, what each action leads to and which one each agent picks. "
                        "Random's thin lines go to all three moves with equal probability.",
             "controls": [{"k": "move", "label": "Move", "min": 1, "max": _GN, "step": 1, "value": _DECIDE_MOVE}]},
            {"type": "quiz", "q": "Greedy is about to enter a corridor between its body and the wall that leads nowhere. What does it check before going in?",
             "options": ["Whether the corridor is long enough", "Only that the very next cell is free", "Nothing at all"],
             "answer": 1, "why": "Greedy's only safety check is the next cell. It cannot see that the corridor is a dead end until "
                                 "every move is fatal."},
        ]},
        {"id": "trap", "title": "How greedy traps itself", "blocks": [
            {"type": "p", "text": "A <b>flood fill</b> counts every free cell the head can still reach. If that region is "
                                  "smaller than the snake, the snake will run out of room before its tail clears a way out, "
                                  "and the game is effectively lost even though no move is yet fatal."},
            {"type": "figure", "figure": "snake-watch-flood", "title": "The pocket closes",
             "caption": f"The end of greedy's game on board {SEED}. Shaded cells are reachable after the move greedy made, marked by whether they give the snake enough room.",
             "controls": [{"k": "back", "label": "Moves before the end", "min": 1, "max": min(30, _GN - 1), "step": 1, "value": _TRAP_BACK}]},
        ]},
        {"id": "bfs", "title": "Breadth-first search with a room check", "blocks": [
            {"type": "p", "text": "<b>Breadth-first search</b> (BFS) explores the board outwards from the head one ring at a time, "
                                  "so the first time it reaches the food it has found a shortest free path. The agent then adds "
                                  "one safety rule that greedy lacks."},
            {"type": "steps", "items": [
                "Treat every body cell, including the tail, as a wall.",
                "Search outwards from the head; stop when the food is reached, or when no free cell is left to visit.",
                "If a path exists, flood-fill from its first cell. Take that step only if the snake would still have more free cells than its length.",
                "Otherwise, flood-fill from each safe move and take the one with the most room.",
            ]},
            {"type": "figure", "figure": "snake-watch-replay", "title": "Replays with the shortest path",
             "caption": f"Each agent on board {SEED}, the Watch page's default game. The marked path is the shortest free route to the food at that moment: "
                        "the route BFS plans, shown for the other agents for comparison.",
             "controls": [{"k": "agent", "label": "Agent", "options": [[a, INFO[a]["short"]] for a in REPLAY_AGENTS], "value": "bfs"},
                          {"k": "pct", "label": "Point in the game", "min": 0, "max": 100, "step": 1, "value": 25, "unit": "%"}]},
            {"type": "callout", "tone": "warn", "title": "BFS is not optimal",
             "text": "It plans one food at a time, treats the tail as fixed although it will move, and its room check is a rule "
                     "of thumb: enough cells now does not guarantee a way out later. A perfect player would follow a route "
                     "that visits every cell (a Hamiltonian cycle) and could fill the board; BFS trades that safety for speed."},
        ]},
        {"id": "replay", "title": "What a replay records", "blocks": [
            {"type": "p", "text": "The live page asks the server for one game. The server plays it to the end (or the replay "
                                  f"cap of {CAP} moves) and sends back, for every move, the head position, whether food was "
                                  "eaten, the action, and the 11 features the agent saw. The browser redraws the board from "
                                  "that list; it does not simulate anything."},
            {"type": "figure", "figure": "snake-watch-score", "title": "Score through the game, same board",
             "caption": f"All three agents on board {SEED}. Each step up is a piece of food."},
            {"type": "p", "text": "Because the food positions come from the board's seed, two agents on the same seed see the "
                                  "same food in the same places until one of them is standing on the spot where the next food "
                                  "would have gone. That makes side-by-side replays a fair comparison of strategies."},
        ]},
        {"id": "pace", "title": "Detours: the cost of staying safe", "blocks": [
            {"type": "p", "text": "As the body grows it gets in the way, and the shortest free path becomes longer than the "
                                  "straight-line distance. BFS's moves per food show how much longer."},
            {"type": "figure", "figure": "snake-watch-pace", "title": "Moves needed for each food",
             "caption": "Moves from one food to the next, against the straight-line distance when that food appeared (10-food running averages)."},
            {"type": "quiz", "q": "Why does BFS need more moves per food late in the game?",
             "options": ["The food appears farther away", "The body blocks the direct route, and the room check rejects risky shortcuts",
                         "BFS becomes less accurate as the board fills"],
             "answer": 1, "why": "The straight-line distance stays about the same on average; what grows is the detour around the body "
                                 "and the moves spent keeping room."},
            {"type": "try", "label": f"Watch BFS on board {SEED}", "params": {"agent": "bfs", "seed": SEED}},
            {"type": "try", "label": f"Watch greedy on the same board", "params": {"agent": "greedy", "seed": SEED},
             "note": "Scrub to the end and look for the move where the pocket closes."},
            {"type": "try", "label": "Watch the random walker", "params": {"agent": "random", "seed": SEED}},
        ]},
    ],
    "takeaways": [
        f"The random walker is the floor: over {_RS[2]} boards it lasted a median of {_RS[0]} moves and ate {_RS[1] / _RS[2]:.2f} food per game.",
        "Greedy heads for the food by Manhattan distance and checks only the next cell.",
        "Greedy dies by sealing itself into a pocket smaller than its body, often many moves before the crash.",
        "BFS finds a shortest free path over the whole board and refuses steps that leave too little room.",
        "BFS pays for safety with detours that grow as the snake gets longer.",
    ],
    "glossary": [
        ["Baseline", "A simple agent that sets the score a learned agent must beat."],
        ["Manhattan distance", "Distance on a grid counted in horizontal plus vertical steps."],
        ["Breadth-first search", "Graph search that explores all cells at distance 1, then 2, and so on, so the first path found is shortest."],
        ["Flood fill", "Counting every cell reachable from a starting cell without crossing a wall or the body."],
        ["Replay", "The recorded list of moves, positions and features of one game, redrawn by the browser."],
        ["Seed", "The number that fixes the random food positions, so two agents can play the same board."],
    ],
}, order=2)


# ===================================================================== Train: figures
def _returns(rewards, gamma):
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    return out[::-1]


@figure("snake-train-return")
def _fig_return(p):
    gamma = p.num("gamma", DQN["gamma"], 0.5, 0.99)
    rewards = [f.reward for f in _G["frames"]]
    G = _returns(rewards, gamma)
    tail = 0
    for v in reversed(G):
        if v >= 0:
            break
        tail += 1
    return {
        "kind": "lines", "x": list(range(1, _GN + 1)), "xlabel": "Move", "ylabel": "Return Gₜ",
        "series": [{"name": f"Gₜ with γ = {gamma:.2f}", "color": "s2", "y": G, "area": True}],
        "markers": [{"x": _GN, "label": f"death {DIE:+d}"}],
        "notes": [f"With γ = {gamma:.2f}, food 5 moves ahead is worth {FOOD * gamma ** 5:.1f} now and food 20 moves ahead "
                  f"{FOOD * gamma ** 20:.2f}; the horizon 1/(1 − γ) is about {1 / (1 - gamma):.0f} moves.",
                  f"The final {DIE:+d} pulls the return below zero for the last {tail} moves of greedy's game on board {SEED}."],
    }


def _transition(case: str):
    """A real transition from greedy's game: an ordinary move just before food, a move that eats, or the fatal move."""
    fr = _G["frames"]
    if case == "death":
        i = _GN - 1
    else:
        eat = [k for k, f in enumerate(fr) if f.ate and k >= 3]
        k = eat[len(eat) // 2]
        i = k if case == "food" else k - 2
    return i, fr[i], (fr[i + 1].feats if i + 1 < _GN else None)


BACKUP_CASES = [["move", "Ordinary move"], ["food", "Eats"], ["death", "Fatal move"]]


@figure("snake-train-backup")
def _fig_backup(p):
    gamma = p.num("gamma", DQN["gamma"], 0.5, 0.99)
    case = p.choice("case", "move", [c for c, _ in BACKUP_CASES])
    T = _table()
    i, fr, nxt = _transition(case)
    q = float(T["Q"][_fidx(fr.feats), fr.action])
    terminal = case == "death"
    qn = 0.0 if terminal else float(T["Q"][_fidx(nxt)].max())
    y = fr.reward + (0 if terminal else gamma * qn)
    new = q + TAB_ALPHA * (y - q)
    return {
        "kind": "flow", "height": 270,
        "nodes": [
            {"id": "q", "x": 15, "y": 12, "w": 170, "h": 64, "title": "Q(s, a) now", "value": f"{q:.2f}", "sub": f"{fr.feats} · {ACTIONS[fr.action]}"},
            {"id": "r", "x": 15, "y": 100, "w": 170, "h": 64, "title": "Reward r", "value": _rw(fr.reward)},
            {"id": "m", "x": 15, "y": 188, "w": 170, "h": 64, "title": "max Q(s′, ·)", "value": "end of game" if terminal else f"{qn:.2f}",
             "sub": "" if terminal else nxt},
            {"id": "y", "x": 240, "y": 100, "w": 150, "h": 64, "title": "Target y", "value": f"{y:.2f}", "hot": True},
            {"id": "n", "x": 440, "y": 100, "w": 145, "h": 64, "title": "New Q(s, a)", "value": f"{new:.2f}"},
        ],
        "edges": [
            {"from": "r", "to": "y", "label": "+", "color": "s3"},
            {"from": "m", "to": "y", "label": "× 0" if terminal else f"× γ = {gamma:.2f}", "color": "s2"},
            {"from": "y", "to": "n", "label": f"× α = {TAB_ALPHA}", "flow": True, "color": "accent"},
            {"from": "q", "to": "n", "label": f"× (1 − α) = {1 - TAB_ALPHA:.1f}", "curve": [0, -10]},
        ],
        "notes": [f"Move {i + 1} of greedy's game on board {SEED}: features {fr.feats}, action {ACTIONS[fr.action]}, reward {_rw(fr.reward)}."
                  + (" The game ended, so there is no future term and the target is just the reward." if terminal else ""),
                  f"Target {y:.2f} against the current {q:.2f}: one update moves the estimate by {TAB_ALPHA * (y - q):+.2f}, "
                  f"a share α = {TAB_ALPHA} of the gap."],
    }


@figure("snake-train-eps")
def _fig_eps(p):
    explore = p.num("explore_games", DQN["explore_games"], 0, 300, integer=True)
    games = p.num("games", 200, 20, 1000, integer=True)
    div = DQN["epsilon_divisor"]
    eps = [max(0, explore - g) / div for g in range(games)]
    free = sum(e == 0 for e in eps)
    return {
        "kind": "lines", "x": list(range(games)), "xlabel": "Game", "ylabel": "ε, chance of a random move", "ymin": 0, "ymax": max(0.05, explore / div),
        "series": [{"name": "ε = max(0, E − game) / " + str(div), "color": "s4", "y": eps, "area": True}],
        "markers": [{"x": min(explore, games - 1), "label": "ε = 0"}] if 0 < explore < games else [],
        "notes": [f"ε starts at {explore / div:.2f} and reaches 0 at game {explore}." if explore else "With no exploration games every move is the network's own choice from the start.",
                  f"{free} of the {games} games ({100 * free / games:.0f}%) are played with no random moves at all."],
    }


@figure("snake-train-curve")
def _fig_curve(p):
    explore = p.num("explore_games", DQN["explore_games"], 0, 150, integer=True)
    gamma = p.num("gamma", DQN["gamma"], 0.5, 0.99)
    T = _qtable(TAB_GAMES, explore, round(gamma, 4), SEED)
    sc = T["scores"]
    n = len(sc)
    last = sc[-50:]
    notes = [f"The first {ROLL} games averaged {np.mean(sc[:ROLL]):.1f}; the last {ROLL} averaged {np.mean(sc[-ROLL:]):.1f} (record {max(sc)}).",
             f"Over the last {len(last)} games the score ranged from {min(last)} to {max(last)} (standard deviation {np.std(last, ddof=1):.1f}) "
             "with no change to the rules: much of the jaggedness is the board, not the learning."]
    if T["stopped"]:
        notes[1] = f"The run reached its budget of {TAB_BUDGET:,} moves and stopped after game {T['stopped']}: long games mean a good agent."
    return {
        "kind": "lines", "x": list(range(1, n + 1)), "xlabel": "Training game", "ylabel": "Score", "ymin": 0,
        "series": [{"name": "Score of each game", "color": "neutral", "y": sc, "thin": True},
                   {"name": f"Running mean of {ROLL}", "color": "s2", "y": _roll(sc)}],
        "bands": [{"x0": 1, "x1": min(explore, n), "label": "exploring"}] if explore else [],
        "notes": notes,
    }


@figure("snake-train-memory")
def _fig_memory(p):
    T = _table()
    r, tot = T["rewards"], T["total"]
    deaths, food, zero = r.get(DIE, 0), r.get(FOOD, 0), r.get(0, 0)
    batch, cap = DQN["replay_batch"], DQN["memory"]
    return {
        "kind": "stats",
        "items": [{"label": "Moves remembered", "value": f"{min(tot, cap):,}", "note": f"{T['games']} games · capacity {cap:,}"},
                  {"label": f"Reward {FOOD:+d}", "value": f"{100 * food / tot:.1f}%", "note": f"{food:,} moves"},
                  {"label": f"Reward {DIE:+d}", "value": f"{100 * deaths / tot:.2f}%", "note": f"{deaths:,} moves", "tone": "warn"},
                  {"label": "Reward 0", "value": f"{100 * zero / tot:.1f}%", "note": f"{zero:,} moves"},
                  {"label": f"Deaths in a batch of {batch:,}", "value": f"{batch * deaths / tot:.1f}", "note": "on average, sampled uniformly"}],
        "notes": [f"Only {100 * deaths / tot:.2f}% of remembered moves end a game, yet those are the moves that teach the agent what to avoid.",
                  f"A replay batch of {batch:,} random moves contains about {batch * deaths / tot:.0f} deaths and {batch * food / tot:.0f} meals, "
                  "drawn from many different games rather than the last few seconds of one."],
    }


@functools.lru_cache(maxsize=1)
def _frozen():
    return [_game("table", s) for s in range(1, NOISE_GAMES + 1)]


@figure("snake-train-noise")
def _fig_noise(p):
    runs = _frozen()
    sc = np.array([g["score"] for g in runs])
    width = 5
    edges = list(range(0, int(sc.max()) + width + 1, width))
    counts, _ = np.histogram(sc, bins=edges)
    return {
        "kind": "bars", "labels": [f"{a}–{a + width - 1}" for a in edges[:-1]], "ylabel": "Boards", "xlabel": "Score",
        "series": [{"name": "Boards", "color": "s2", "values": counts.tolist()}],
        "notes": [f"The same frozen table, with no exploration and no learning, scores between {sc.min()} and {sc.max()} on "
                  f"{len(sc)} boards (mean {sc.mean():.1f}, standard deviation {sc.std(ddof=1):.1f}).",
                  "A learning curve plots one such draw per game, so even a policy that has stopped improving produces a jagged line."],
    }


# ===================================================================== Train: text
explainer("agents/train", {
    "title": "Learning from reward: from the Bellman equation to a deep Q-network",
    "kicker": "Reinforcement Learning Lab · Snake",
    "summary": "The DQN starts with random weights and learns only from the rewards its own games produce. Q-learning turns one "
               "equation, the Bellman equation, into an update rule; ε-greedy exploration and a replay memory make that rule "
               "work in practice. This page builds the method step by step with a Q-table on the same 11 features, trained in a "
               "fraction of a second, and explains why the learning curve is so noisy.",
    "sections": [
        {"id": "value", "title": "What the agent is trying to maximise", "blocks": [
            {"type": "p", "text": "The agent wants a high score, but rewards arrive late: a move now may only pay off when the "
                                  "food is reached ten moves later. The <b>return</b> adds up all future rewards, each shrunk by "
                                  "a <b>discount factor</b> γ per move of delay."},
            {"type": "eq", "text": "Gₜ = rₜ₊₁ + γ rₜ₊₂ + γ² rₜ₊₃ + …  =  rₜ₊₁ + γ Gₜ₊₁",
             "where": [["Gₜ", "the return from move t"], ["rₜ₊₁", f"reward after move t ({FOOD:+d}, {DIE:+d} or 0)"],
                       ["γ", f"discount factor between 0 and 1 (the original uses {DQN['gamma']})"]]},
            {"type": "figure", "figure": "snake-train-return", "title": "The return through a real game",
             "caption": f"Gₜ for every move of greedy's game on board {SEED}. Each piece of food lifts the return before it; the death at the end pulls it down.",
             "controls": [{"k": "gamma", "label": "Discount γ", "min": 0.5, "max": 0.99, "step": 0.01, "value": DQN["gamma"]}]},
            {"type": "p", "text": "A small γ makes the agent short-sighted: only food a few moves away counts. A γ close to 1 "
                                  "makes distant rewards and distant dangers count almost as much as near ones, which is harder "
                                  "to learn because the values depend on many more future moves."},
        ]},
        {"id": "bellman", "title": "The Bellman equation", "blocks": [
            {"type": "p", "text": "The <b>optimal Q-value</b> Q*(s, a) is the expected return from taking action a in state s and "
                                  "then playing perfectly. It satisfies a consistency condition: the value of a move is the "
                                  "reward it earns plus the discounted value of the best move from wherever it lands."},
            {"type": "eq", "text": "Q*(s, a) = E[ r + γ · maxₐ′ Q*(s′, a′) ]",
             "where": [["s′", "the state after the move (random, because the food is placed at random)"],
                       ["maxₐ′ Q*(s′, a′)", "the value of playing the best move from s′"],
                       ["E", "the average over what can happen next"]]},
            {"type": "p", "text": "<b>Q-learning</b> turns this into an update. After each move, compare the current estimate with "
                                  "the right-hand side computed from the estimate itself (the <b>target</b>), and move the "
                                  "estimate part of the way towards it."},
            {"type": "eq", "text": "y = r + γ · maxₐ′ Q(s′, a′)    (y = r if the game ended)\nQ(s, a) ← Q(s, a) + α (y − Q(s, a))",
             "where": [["y", "the target"], ["α", f"the step size (the table here uses α = {TAB_ALPHA})"]]},
            {"type": "figure", "figure": "snake-train-backup", "title": "One backup, worked through",
             "caption": "A real transition and the learned table: an ordinary move two before a meal, the move that eats, or the fatal move.",
             "controls": [{"k": "gamma", "label": "Discount γ", "min": 0.5, "max": 0.99, "step": 0.01, "value": DQN["gamma"]},
                          {"k": "case", "label": "Transition", "options": BACKUP_CASES, "value": "move"}]},
        ]},
        {"id": "table", "title": "A Q-table on 11 features", "blocks": [
            {"type": "p", "text": f"With only {N_VECTORS} possible feature vectors, Q can be stored as a table of {N_VECTORS} × {NA} "
                                  "numbers, all starting at 0. The figure trains such a table with the rule above, the original "
                                  f"ε schedule and the original rewards, for {TAB_GAMES} games."},
            {"type": "figure", "figure": "snake-train-curve", "title": "Learning curve of the Q-table",
             "caption": "Score of every training game and its running mean. The shaded band is the exploration phase.",
             "controls": [{"k": "explore_games", "label": "Exploration games E", "min": 0, "max": 150, "step": 10, "value": DQN["explore_games"]},
                          {"k": "gamma", "label": "Discount γ", "min": 0.5, "max": 0.99, "step": 0.01, "value": DQN["gamma"]}]},
            {"type": "callout", "tone": "tip", "title": "Exploration is not always worth its cost",
             "text": f"Every value starts at 0 and dying costs {DIE:+d}, so a move that has killed the agent soon scores below a move "
                     "it has never tried, and acting greedily on the estimates already tests untried moves. Random moves also cost "
                     "games: while ε is high the snake dies early and rarely reaches the long-body positions it most needs to learn. "
                     "Set E to 0 and compare the curves. Whether the network behaves the same way is an experiment for the live page."},
        ]},
        {"id": "dqn", "title": "From the table to a deep Q-network", "blocks": [
            {"type": "p", "text": "A DQN replaces the table with a neural network Q(s, a; θ) and replaces the update with a step of "
                                  "gradient descent on the squared gap between prediction and target."},
            {"type": "eq", "text": "L(θ) = ( r + γ · maxₐ′ Q(s′, a′; θ) − Q(s, a; θ) )²",
             "where": [["θ", f"the network's {N_PARAMS:,} weights and biases"],
                       ["L", "the loss for one move, averaged over a batch"]]},
            {"type": "steps", "title": "The original training loop", "items": [
                "Read the 11 features; with probability ε move at random, otherwise take the network's best move.",
                "Play the move, observe the reward and the next features.",
                f"Take one Adam step (learning rate {DQN['lr']}) on that single move, and store it in the replay memory.",
                f"When the game ends, take one more step on a random batch of up to {DQN['replay_batch']:,} remembered moves.",
                "Start the next game with a smaller ε.",
            ]},
            {"type": "compare", "items": [
                {"title": "Q-table", "points": ["Exact update, one cell at a time", "Converges for a small, fully visited state space",
                                                "Cannot generalise to an unseen state"]},
                {"title": "DQN", "points": ["Gradient step changes every output a little", "The target moves as the weights change, which can make learning unstable",
                                            "Generalises, and scales to inputs no table could hold"]},
            ]},
        ]},
        {"id": "explore", "title": "ε-greedy exploration", "blocks": [
            {"type": "p", "text": "An agent that always takes its current best move never learns about the others. ε-greedy "
                                  "takes a random move with probability ε and its best move otherwise. The original lowers ε "
                                  "by a fixed amount each game until it reaches zero."},
            {"type": "eq", "text": f"ε(g) = max(0, E − g) / {DQN['epsilon_divisor']}",
             "where": [["g", "games played so far"], ["E", f"exploration games ({DQN['explore_games']} in the original, so ε starts at {EPS0:.2f})"]]},
            {"type": "figure", "figure": "snake-train-eps", "title": "The exploration schedule",
             "caption": "The chance of a random move in each game, for the settings on the Train page.",
             "controls": [{"k": "explore_games", "label": "Exploration games E", "min": 0, "max": 300, "step": 10, "value": DQN["explore_games"]},
                          {"k": "games", "label": "Games to play", "min": 20, "max": 1000, "step": 10, "value": 200}]},
            {"type": "quiz", "q": f"With E = {DQN['explore_games']} and 50 games of training, what is ε in the last game?",
             "options": ["0", f"{(DQN['explore_games'] - 49) / DQN['epsilon_divisor']:.3f}", f"{EPS0:.2f}"], "answer": 1,
             "why": f"ε(49) = ({DQN['explore_games']} − 49) / {DQN['epsilon_divisor']}. A short run never reaches the fully greedy phase, "
                    "so some of its moves are random to the end."},
        ]},
        {"id": "replay", "title": "Replay memory", "blocks": [
            {"type": "p", "text": "Consecutive moves are almost identical, so learning only from the latest move tunes the network "
                                  "to the last few seconds of play and makes it forget earlier lessons. A <b>replay memory</b> "
                                  f"keeps up to {DQN['memory']:,} past moves; after each game the original trains on a random "
                                  "sample of them."},
            {"type": "figure", "figure": "snake-train-memory", "title": "What the memory holds",
             "caption": f"Rewards of every move from the {TAB_GAMES}-game Q-table run above, as a replay memory would store them."},
            {"type": "p", "text": "Random sampling breaks the correlation between neighbouring moves and lets one rare, informative "
                                  "move (a death or a meal) be learned from more than once."},
        ]},
        {"id": "noise", "title": "Why the learning curve is noisy", "blocks": [
            {"type": "p", "text": "Each point on a learning curve is the score of one game. That score depends on where the food "
                                  "happens to appear, on ε's random moves, and on the policy, which changes after every move. "
                                  "Freezing the trained table and replaying it on fresh boards isolates the first source."},
            {"type": "figure", "figure": "snake-train-noise", "title": "One frozen policy, many boards",
             "caption": f"The Q-table from the run above, with no exploration and no learning, on {NOISE_GAMES} boards."},
            {"type": "callout", "tone": "warn", "title": "Read learning curves with care",
             "text": "Compare running means, not single games, and compare several training seeds before concluding one setting is "
                     "better. A DQN can also get worse after improving, because each gradient step changes the values of states "
                     "it is not currently visiting. The table here is a stand-in: it learns the same values a perfect network would "
                     "need, but it is not the network you train on the live page."},
            {"type": "quiz", "q": "Two runs differ only in γ, and the second's last 20 games average a few points higher. What can you conclude?",
             "options": ["The second γ is better", "Nothing yet: run both with several seeds and compare the spread", "The first run had a bug"],
             "answer": 1, "why": "Single-game scores spread widely even for a fixed policy, so a gap of a few points between two runs can easily be "
                                 "chance. Repeat with different seeds and compare means with their uncertainty."},
            {"type": "try", "label": "Train with the original settings",
             "params": {"explore_games": DQN["explore_games"], "gamma": DQN["gamma"], "lr": DQN["lr"], "hidden": DQN["hidden"]}},
            {"type": "try", "label": "Train with no exploration", "params": {"explore_games": 0},
             "note": "Compare the curve with the original: does the network still find food?"},
            {"type": "try", "label": "Train a short-sighted agent (γ = 0.5)", "params": {"gamma": 0.5}},
        ]},
    ],
    "takeaways": [
        "The return adds future rewards discounted by γ per move; 1/(1 − γ) is roughly how far ahead the agent looks.",
        "Q-learning moves each estimate towards the Bellman target r + γ · max Q(s′, ·).",
        f"With {NF} binary features a table can hold Q exactly; the DQN learns the same function with shared weights.",
        "ε-greedy trades a few random moves early for better knowledge later; the original's ε falls linearly to zero.",
        "Replay memory reuses rare moves and breaks the correlation between consecutive ones.",
        "Learning curves are noisy because one game's score varies widely even for a fixed policy.",
    ],
    "glossary": [
        ["Return", "The discounted sum of future rewards from a given move."],
        ["Discount γ", "How much a reward one move later is worth compared with a reward now."],
        ["Bellman equation", "Q*(s, a) = E[r + γ max Q*(s′, ·)]: a value is its reward plus the discounted best value next."],
        ["Target", "The right-hand side of the Bellman equation, computed from the current estimates."],
        ["ε-greedy", "Take a random action with probability ε, otherwise the best-looking one."],
        ["Replay memory", "A store of past moves that training samples from at random."],
        ["DQN", "Deep Q-network: a neural network trained to output Q-values, one per action."],
    ],
}, order=3)


# ===================================================================== Compare: figures
CMP_AGENTS = ["random", "greedy", "bfs", "table"]


def _pool(n: int) -> dict:
    return {a: [_game(a, CMP_SEED + i, CMP_CAP) for i in range(n)] for a in CMP_AGENTS}


def _summary(games: list) -> dict:
    return S.summarise([{"score": g["score"], "steps": g["steps"], "cause": g["cause"]} for g in games])


@figure("snake-cmp-design")
def _fig_design(p):
    pool = _pool(CMP_GAMES)
    nodes = [{"id": "b", "x": 10, "y": 110, "w": 160, "h": 80, "title": f"{CMP_GAMES} boards",
              "sub": f"seeds {CMP_SEED}–{CMP_SEED + CMP_GAMES - 1}", "value": f"≤ {CMP_CAP} moves"}]
    edges = []
    for j, a in enumerate(CMP_AGENTS):
        s = _summary(pool[a])
        nodes.append({"id": a, "x": 330, "y": 8 + 73 * j, "w": 255, "h": 58, "title": label(a),
                      "value": f"mean {s['mean']:.1f} ± {s['ci95']:.1f}", "hot": a == "bfs"})
        edges.append({"from": "b", "to": a, "width": 2.5, "flow": True, "color": AGENT_COLOR[a]})
    fg, fb = _foods(pool["greedy"][0]), _foods(pool["bfs"][0])
    k = next((i for i, (u, v) in enumerate(zip(fg, fb)) if u != v), min(len(fg), len(fb)))
    return {
        "kind": "flow", "height": 300, "nodes": nodes, "edges": edges,
        "notes": [f"Every agent plays the same {CMP_GAMES} boards, so differences in score come from the strategies, not from luckier food.",
                  f"On board {CMP_SEED}, greedy and BFS were given the same first {k} food positions; the sequences split only when a "
                  "new food landed on a cell one snake occupied and the other did not."],
    }


@figure("snake-cmp-dist")
def _fig_dist(p):
    n = p.num("games", CMP_GAMES, 3, 10, integer=True)
    pool = _pool(n)
    seeds = list(range(CMP_SEED, CMP_SEED + n))
    points, lines, parts = [], [], []
    for j, a in enumerate(CMP_AGENTS):
        s = _summary(pool[a])
        off = (j - 1.5) * 0.12
        points += [{"x": sd + off, "y": g["score"], "color": AGENT_COLOR[a], "label": None} for sd, g in zip(seeds, pool[a])]
        lines.append({"x": [seeds[0] - 0.3, seeds[-1] + 0.3], "y": [s["mean"], s["mean"]], "color": AGENT_COLOR[a], "dash": True, "name": f"{short(a)} mean"})
        parts.append(f"{short(a)} {s['mean']:.1f} ± {s['ci95']:.1f}")
    points = [{k: v for k, v in pt.items() if v is not None} for pt in points]
    capped = sum(g["cause"] == "limit" for g in pool["bfs"])
    return {
        "kind": "scatter", "points": points, "lines": lines, "xlabel": "Board (seed)", "ylabel": f"Score in the first {CMP_CAP} moves",
        "notes": ["Means with 95% intervals: " + ", ".join(parts) + ".",
                  f"{capped} of {n} BFS games were still running when the {CMP_CAP}-move cap stopped them, so its mean is a lower bound."
                  if capped else f"Every BFS game ended before the {CMP_CAP}-move cap."],
    }


@functools.lru_cache(maxsize=1)
def _greedy_scores():
    return np.array([_game("greedy", s)["score"] for s in range(1, CI_GAMES + 1)], dtype=float)


@figure("snake-cmp-ci")
def _fig_ci(p):
    level = p.num("level", 95, 80, 99)
    z = NormalDist().inv_cdf(0.5 + level / 200)
    sc = _greedy_scores()
    ns = list(range(3, len(sc) + 1))
    mean = [float(sc[:k].mean()) for k in ns]
    half = [float(z * sc[:k].std(ddof=1) / np.sqrt(k)) for k in ns]
    q = len(sc) // 4
    hq, hn = half[ns.index(q)], half[-1]
    return {
        "kind": "lines", "x": ns, "xlabel": "Games played", "ylabel": "Greedy's mean score", "ymin": 0,
        "series": [{"name": "Mean so far", "color": "s3", "y": mean},
                   {"name": f"{level:.0f}% interval", "color": "s3", "y": [m + h for m, h in zip(mean, half)], "dash": True, "thin": True},
                   {"name": f"{level:.0f}% interval (lower)", "color": "s3", "y": [m - h for m, h in zip(mean, half)], "dash": True, "thin": True}],
        "notes": [f"After {len(sc)} games greedy's mean is {mean[-1]:.1f} ± {hn:.1f} at {level:.0f}% confidence (z = {z:.2f}).",
                  f"With {q} games the interval was ±{hq:.1f}; with four times as many it is ±{hn:.1f}, {hn / hq:.2f} times the width "
                  "(the 1/√n rule predicts 0.50)."],
    }


@figure("snake-cmp-causes")
def _fig_causes(p):
    pool = _pool(CMP_GAMES)
    keys = [c for c in ("wall", "self", "timeout", "limit") if any(g["cause"] == c for gs in pool.values() for g in gs)]
    series = [{"name": label(a), "color": AGENT_COLOR[a], "values": [sum(g["cause"] == c for g in pool[a]) for c in keys]} for a in CMP_AGENTS]
    top = {a: Counter(g["cause"] for g in pool[a]).most_common(1)[0] for a in CMP_AGENTS}
    return {
        "kind": "bars", "labels": [_cause(c) for c in keys], "ylabel": f"Games (of {CMP_GAMES})", "series": series,
        "notes": ["Most common ending: " + "; ".join(f"{short(a)} {_cause(c)} ({k} of {CMP_GAMES})" for a, (c, k) in top.items()) + ".",
                  f"{sum(g['cause'] == 'self' for a in ('greedy', 'table') for g in pool[a])} of the {2 * CMP_GAMES} greedy and Q-table games "
                  "ended on the snake's own body, the trap the danger features cannot see."],
    }


@functools.lru_cache(maxsize=1)
def _ceiling():
    pool = _pool(CMP_GAMES)
    groups, agree_t, agree_g, total = defaultdict(Counter), 0, 0, 0
    Q = _table()["Q"]
    for g in pool["bfs"]:
        for fr in g["frames"]:
            groups[fr.feats][fr.action] += 1
            agree_t += int(np.argmax(Q[_fidx(fr.feats)])) == fr.action
            agree_g += S.act_greedy(_env_at(fr), random.Random(0)) == fr.action
            total += 1
    best = sum(max(c.values()) for c in groups.values())
    return {"vectors": len(groups), "total": total, "ceiling": best / total, "table": agree_t / total, "greedy": agree_g / total,
            "means": {a: _summary(pool[a])["mean"] for a in CMP_AGENTS}}


@figure("snake-cmp-ceiling")
def _fig_ceiling(p):
    c = _ceiling()
    m = c["means"]
    miss = c["total"] - round(c["ceiling"] * c["total"])
    return {
        "kind": "stats",
        "items": [{"label": "BFS looks at", "value": f"{W * H} cells", "note": "the whole board"},
                  {"label": "The DQN looks at", "value": f"{NF} bits", "note": f"{c['vectors']} distinct vectors in BFS's games"},
                  {"label": f"Best possible {NF}-feature copy of BFS", "value": f"{100 * c['ceiling']:.1f}%", "note": "of BFS's moves matched", "tone": "good"},
                  {"label": "Q-table agrees with BFS", "value": f"{100 * c['table']:.1f}%", "note": f"mean score {m['table']:.1f} vs {m['bfs']:.1f}"},
                  {"label": "Greedy agrees with BFS", "value": f"{100 * c['greedy']:.1f}%", "note": f"mean score {m['greedy']:.1f}", "tone": "warn"}],
        "notes": [f"Taking BFS's most common move for each feature vector would still differ from BFS on {miss} of its {c['total']:,} moves "
                  f"({100 * (1 - c['ceiling']):.1f}%).",
                  f"Greedy already makes BFS's move {100 * c['greedy']:.0f}% of the time yet scores {m['greedy']:.1f} to BFS's {m['bfs']:.1f}: "
                  "the few moves where they differ decide whether the snake walls itself in."],
    }


# ===================================================================== Compare: text
explainer("agents/compare", {
    "title": "A fair contest: comparing Snake agents by Monte Carlo",
    "kicker": "Reinforcement Learning Lab · Snake",
    "summary": "One game proves little: the food lands somewhere different every time and scores vary widely. The comparison "
               "plays every agent on the same set of boards, reports each mean with a confidence interval, and records how "
               "each game ended. The evidence explains why a path-finder that sees the whole board beats a network that sees "
               f"{NF} features.",
    "sections": [
        {"id": "question", "title": "What makes a comparison fair", "blocks": [
            {"type": "p", "text": "A <b>Monte Carlo</b> comparison estimates each agent's average score by playing many random "
                                  "games. To be fair, every agent should face the same randomness: board k always uses seed k, "
                                  "so every agent gets the same food positions. Statisticians call this <b>common random "
                                  "numbers</b>; it removes the luck of the draw from the difference between agents."},
            {"type": "figure", "figure": "snake-cmp-design", "title": "The design",
             "caption": f"The figures on this page play the first {CMP_CAP} moves of each game, to stay fast; the live page plays up to {CAP}. "
                        f"The fourth agent is the {TAB_GAMES}-game Q-table from the Train page's explainer, standing in for your DQN."},
            {"type": "eq", "text": "Var(X̄_A − X̄_B) = ( Var X_A + Var X_B − 2 Cov(X_A, X_B) ) / n",
             "where": [["X_A, X_B", "scores of agents A and B on the same board"], ["n", "number of boards"],
                       ["Cov", "positive when the same boards are easy or hard for both agents, which shrinks the variance of the difference"]]},
        ]},
        {"id": "scores", "title": "Scores, board by board", "blocks": [
            {"type": "figure", "figure": "snake-cmp-dist", "title": "Every game",
             "caption": "Each dot is one game, placed at its board's seed and nudged sideways per agent. Dashed lines are the means.",
             "controls": [{"k": "games", "label": "Boards per agent", "min": 3, "max": 10, "step": 1, "value": CMP_GAMES}]},
            {"type": "p", "text": "The spread within one agent is often larger than the gap between agents. Without repeated games "
                                  "a lucky run of one agent could easily look like a better strategy."},
        ]},
        {"id": "ci", "title": "Confidence intervals", "blocks": [
            {"type": "p", "text": "The mean of n games is itself random. Its standard error shrinks like 1/√n, and for enough "
                                  "games the mean is close to normally distributed, which gives an interval that covers the "
                                  "true average in a chosen share of repeated experiments."},
            {"type": "eq", "text": "x̄ ± z · s / √n",
             "where": [["x̄", "the mean score over n games"], ["s", "the sample standard deviation of the scores"],
                       ["z", "the normal quantile for the confidence level (1.96 for 95%)"]]},
            {"type": "steps", "items": [
                "Play n games on boards 1 to n and record each score.",
                "Compute the mean x̄ and the standard deviation s.",
                "The 95% interval is x̄ ± 1.96 s/√n. Quadrupling n halves its width.",
                "Two agents whose intervals do not overlap differ clearly; overlapping intervals need more games or a paired test.",
            ]},
            {"type": "figure", "figure": "snake-cmp-ci", "title": "The interval narrows as games are added",
             "caption": f"Greedy's running mean over {CI_GAMES} complete games, with its confidence interval.",
             "controls": [{"k": "level", "label": "Confidence level", "min": 80, "max": 99, "step": 1, "value": 95, "unit": "%"}]},
            {"type": "quiz", "q": "An agent's 95% interval over n games is ± h. Roughly how many games give ± h/2?",
             "options": ["2n", "4n", "10n"], "answer": 1,
             "why": "The width scales as 1/√n, so halving it needs four times as many games."},
        ]},
        {"id": "causes", "title": "How the games end", "blocks": [
            {"type": "p", "text": "The mean says who wins; the cause of death says why the others lose. Each game ends at a wall, "
                                  f"on the body, by timeout ({LIMIT} × length moves without finishing), or at the step cap."},
            {"type": "figure", "figure": "snake-cmp-causes", "title": "Causes of the end of each game",
             "caption": f"Counts over the same {CMP_GAMES} boards per agent."},
        ]},
        {"id": "why", "title": "Why BFS beats the learned agent", "blocks": [
            {"type": "p", "text": "The DQN has to choose from 11 features. BFS reads the whole board and searches it. The question "
                                  "is not whether the network is trained well, but how much a perfect learner could achieve "
                                  "with only those features."},
            {"type": "figure", "figure": "snake-cmp-ceiling", "title": "What the features allow",
             "caption": "Moves from BFS's games above, grouped by the features the DQN would have seen at each one."},
            {"type": "cards", "items": [
                {"title": "Sees the board", "text": "BFS knows where every body cell is; the features only know about the three cells next to the head."},
                {"title": "Plans ahead", "text": "BFS checks a whole path and the room left after it; a Q-value is a learned average over every board that shares a vector."},
                {"title": "Rare mistakes decide", "text": "Agreeing on almost every move is not enough. The moves that seal a pocket are few, and the features cannot see them coming."},
            ]},
            {"type": "quiz", "q": "Which change would most help the DQN close the gap to BFS?",
             "options": ["More hidden units", "Training for more games with the same features",
                         "Inputs that describe the board beyond the next cell, such as the free room after each move"],
             "answer": 2, "why": "The ceiling is set by what the features can distinguish. A bigger network or longer training cannot "
                                 "recover information the input never contained."},
        ]},
        {"id": "limits", "title": "Limits of this comparison", "blocks": [
            {"type": "callout", "tone": "warn", "title": "What these numbers do not show",
             "text": f"The figures here stop each game after {CMP_CAP} moves, so BFS's mean understates what it does in a full game. "
                     f"They use {CMP_GAMES} boards by default, which gives wide intervals; the normal approximation behind them is rough "
                     "for so few games and for skewed scores. The learned agent is a Q-table, not your network: the live page "
                     "compares your trained DQN directly."},
            {"type": "try", "label": "Run the full comparison, including your DQN",
             "params": {"random": True, "greedy": True, "bfs": True, "dqn": True, "episodes": 20, "seed": CMP_SEED},
             "note": "Train a network first. BFS is the slowest agent, so this takes a while."},
            {"type": "try", "label": "Compare only the cheap agents over many boards",
             "params": {"random": True, "greedy": True, "bfs": False, "dqn": False, "episodes": 100, "seed": CMP_SEED}},
        ]},
    ],
    "takeaways": [
        "Play every agent on the same boards (common random numbers) so the difference reflects strategy, not luck.",
        "Report each mean with an interval x̄ ± z·s/√n; four times the games halves its width.",
        "Causes of death explain rankings: random hits walls, greedy and the learned agent trap themselves.",
        f"With {NF} features even a perfect learner cannot copy BFS on every move, and the moves it misses are the costly ones.",
        "Better inputs, not a bigger network, are what a feature-based agent needs to catch up.",
    ],
    "glossary": [
        ["Monte Carlo", "Estimating an average by running many random trials."],
        ["Common random numbers", "Giving every compared option the same random inputs, here the same board seeds."],
        ["Standard error", "The standard deviation of a mean: s/√n."],
        ["Confidence interval", "A range built so that, over repeated experiments, it covers the true mean in the stated share of cases."],
        ["Step cap", "A maximum number of moves after which a game is stopped without a death."],
        ["Ceiling", "The best agreement any policy using only the features could reach with BFS."],
    ],
}, order=4)
