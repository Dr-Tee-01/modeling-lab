def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.get_json() == {"status": "ok"}


def test_unknown_route_uses_error_envelope(client):
    r = client.get("/api/nope")
    assert r.status_code == 404
    assert set(r.get_json()["error"]) == {"code", "message"}


def test_unknown_job_kind(client):
    r = client.post("/api/jobs", json={"kind": "nope"})
    assert r.status_code == 400 and r.get_json()["error"]["code"] == "invalid_parameter"
