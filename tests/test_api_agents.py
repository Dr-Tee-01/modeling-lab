import time

import numpy as np
import pytest

from modlab.agents import snake as S


def test_rules_match_the_original():
    env = S.SnakeEnv(seed=0)
    assert env.head == (16, 12) and list(env.snake) == [(16, 12), (15, 12), (14, 12)] and env.d == 0
    # turning back into the body is death on "self"; running straight into the right wall is death on "wall"
    r, done, _, cause = S.SnakeEnv(seed=0).step(0)
    assert (r, done) == (0, False)
    env = S.SnakeEnv(seed=0)
    for _ in range(20):          # x = 16 reaches the wall (x = 32) on step 16
        r, done, _, cause = env.step(0)
        if done:
            break
    assert done and cause == "wall" and r == -10


def test_moving_into_the_current_tail_is_fatal():
    env = S.SnakeEnv(seed=0)
    env.snake = __import__("collections").deque([(5, 5), (5, 6), (4, 6), (4, 5)])   # tail directly left of the head
    env.body = set(env.snake)
    env.d = 3                                                                         # heading up
    _, done, _, cause = env.step(2)                                                   # turn left -> into the tail
    assert done and cause == "self"


def test_state_features():
    env = S.SnakeEnv(seed=0)
    env.food = (env.head[0] + 3, env.head[1] - 2)
    s = env.state()
    assert s.tolist() == [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0]       # no danger, moving right, food right and up


def test_baselines_rank_as_expected():
    r = S.compare(["random", "greedy"], 10, 3, None)
    assert r["random"]["mean"] < 2 < r["greedy"]["mean"]
    e = S.episode("bfs", 4, None, max_steps=600)
    assert e["score"] > 10 and len(e["heads"]) == e["steps"] == len(e["actions"])


def test_replay_reconstructs_the_game():
    e = S.episode("greedy", 5, None)
    snake, foods, fi = [tuple(p) for p in e["start"]], list(map(tuple, e["foods"])), 0
    for head, ate in zip(e["heads"], e["ate"]):
        snake.insert(0, tuple(head))
        if ate:
            fi += 1
        else:
            snake.pop()
    assert fi == e["score"] and len(foods) >= e["score"]


def get(client, url, status=200):
    r = client.get(url)
    assert r.status_code == status, r.get_json()
    return r.get_json()


def test_episode_endpoint_and_validation(client):
    j = get(client, "/api/agents/snake/episode?agent=greedy&seed=2")
    assert j["score"] >= 1 and j["features"][0] == "danger straight" and len(j["states"][0]) == 11
    assert get(client, "/api/agents/snake/episode?agent=alien", 400)["error"]["code"] == "invalid_parameter"


def test_dqn_needs_a_model_then_trains(client):
    S._MODELS.clear()
    assert get(client, "/api/agents/snake/episode?agent=dqn", 409)["error"]["code"] == "no_model"
    pytest.importorskip("torch")
    job = client.post("/api/jobs", json={"kind": "snake_dqn", "games": 15, "seed": 1}).get_json()["job"]
    for _ in range(120):
        job = client.get(f"/api/jobs/{job['id']}").get_json()["job"]
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.25)
    assert job["status"] == "done", job["error"]
    res = job["result"]
    assert len(res["scores"]) == 15 and res["model"]["id"].startswith("dqn-")
    j = get(client, "/api/agents/snake/episode?agent=dqn&seed=1")
    assert j["q"] and len(j["q"][0]) == 3
    w = get(client, "/api/agents/snake/weights")
    assert np.array(w["w1"]).shape == (256, 11) and np.array(w["w2"]).shape == (3, 256)


def test_compare_job(client):
    job = client.post("/api/jobs", json={"kind": "snake_compare", "agents": ["random", "greedy"], "episodes": 5}).get_json()["job"]
    for _ in range(80):
        job = client.get(f"/api/jobs/{job['id']}").get_json()["job"]
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.25)
    assert job["status"] == "done", job["error"]
    a = job["result"]["agents"]
    assert set(a) == {"random", "greedy"} and len(a["greedy"]["scores"]) == 5
    bad = client.post("/api/jobs", json={"kind": "snake_compare", "agents": ["nope"]})
    assert bad.status_code == 400


def test_catalog_describes_the_game():
    from modlab.server import create_app
    cat = create_app().test_client().get("/api/catalog").get_json()["agents"]
    assert cat["board"] == {"w": S.W, "h": S.H, "start_length": 3}
    assert cat["features"] == S.FEATURES and cat["actions"] == S.ACTIONS
    assert set(cat["agents"]) == set(S.AGENTS) and all("label" in a and "short" in a for a in cat["agents"].values())


def test_interactive_game_runs_on_the_server():
    from modlab.server import create_app
    c = create_app().test_client()
    g = c.post("/api/agents/snake/play", json={"seed": 0}).get_json()
    assert g["snake"] == [[16, 12], [15, 12], [14, 12]] and g["heading"] == 0 and not g["over"]
    assert len(g["features"]) == len(S.FEATURES)
    # reversing is ignored (keeps going right); then turn down
    g = c.post(f"/api/agents/snake/play/{g['id']}/step", json={"direction": 2}).get_json()
    assert g["snake"][0] == [17, 12] and g["heading"] == 0
    g = c.post(f"/api/agents/snake/play/{g['id']}/step", json={"direction": 1}).get_json()
    assert g["snake"][0] == [17, 13] and g["heading"] == 1 and g["steps"] == 2
    # run into the bottom wall: the game ends and further steps change nothing
    for _ in range(S.H):
        g = c.post(f"/api/agents/snake/play/{g['id']}/step", json={}).get_json()
    assert g["over"] and g["cause"] == "wall"
    again = c.post(f"/api/agents/snake/play/{g['id']}/step", json={}).get_json()
    assert again["steps"] == g["steps"]
    assert c.post("/api/agents/snake/play/missing/step", json={}).status_code == 404
    assert c.post(f"/api/agents/snake/play/{g['id']}/step", json={"direction": 9}).status_code == 400


def test_episode_and_compare_summaries():
    e = S.episode("greedy", 2)
    assert e["score_by_move"][-1] == e["score"] and len(e["score_by_move"]) == len(e["actions"])
    assert abs(sum(e["summary"]["action_share"]) - 1) < 1e-9
    res = S.compare(["random", "greedy"], 4, 1, None)
    summ = S.compare_summary(res, 4, 1)
    assert summ["best"] == "greedy" and summ["ranking"] == ["greedy", "random"]
    assert summ["top_game"]["score"] == max(r["max"] for r in res.values()) and summ["seeds"] == [1, 4]
