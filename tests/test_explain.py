"""Every explainer page is well formed, every figure computes, and every slider range works."""
import pytest

from modlab import explain as X

KEYS = sorted(X.EXPLAINERS)


def test_there_are_explainers():
    assert "spread/sir" in X.EXPLAINERS


@pytest.mark.parametrize("key", KEYS)
def test_explainer_structure(key):
    assert X.problems(key) == []


@pytest.mark.parametrize("key", KEYS)
def test_explainer_builds_and_figures_are_valid(client, key):
    r = client.get(f"/api/explain/{key}")
    assert r.status_code == 200, r.get_json()
    page = r.get_json()
    assert page["minutes"] >= 1 and page["sections"]
    for sec in page["sections"]:
        for b in sec["blocks"]:
            if b["type"] != "figure":
                continue
            assert X.check_figure(b["data"]) == [], (key, b["figure"])
            # every slider at both ends of its range must still compute
            for c in b.get("controls", []):
                values = [o[0] for o in c["options"]] if "options" in c else (c["min"], c["max"])
                for v in values:
                    q = {**b.get("params", {}), c["k"]: v}
                    rr = client.get(f"/api/explain/figure/{b['figure']}", query_string=q)
                    assert rr.status_code == 200, (key, b["figure"], c["k"], v, rr.get_json())
                    assert X.check_figure(rr.get_json()) == [], (key, b["figure"], c["k"], v)


def test_index_and_errors(client):
    idx = client.get("/api/explain").get_json()["explainers"]
    assert {e["key"] for e in idx} == set(X.EXPLAINERS)
    assert client.get("/api/explain/spread/nope").status_code == 404
    assert client.get("/api/explain/figure/nope").status_code == 404
    assert client.get("/api/explain/figure/sir-curves", query_string={"beta": 99}).status_code == 400
