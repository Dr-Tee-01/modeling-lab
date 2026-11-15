"""The four labs cover every model exactly once, and the catalog publishes them."""
from modlab import explain as X
from modlab.labs import LABS


def test_every_view_is_in_exactly_one_lab():
    keys = [k for lab in LABS for k in lab["views"]]
    assert len(keys) == len(set(keys))
    assert set(keys) == set(X.EXPLAINERS)          # each lab view has an explainer, and vice versa
    assert [lab["name"] for lab in LABS] == ["Modeling Lab", "Geometry Lab", "Time Series Lab", "Reinforcement Learning Lab"]


def test_catalog_publishes_labs_in_order(client):
    labs = client.get("/api/catalog").get_json()["labs"]["labs"]
    assert [lab["id"] for lab in labs] == ["modeling", "geometry", "timeseries", "rl"]
    ts = next(lab for lab in labs if lab["id"] == "timeseries")
    assert ts["views"][:3] == ["markets/volatility", "markets/returns", "markets/neural"]
    assert "crossover/crash-shapes" in ts["views"] and all(lab["blurb"] for lab in labs)


def test_explainer_links_follow_the_lab(client):
    neural = client.get("/api/explain/markets/neural").get_json()
    assert neural["next"]["key"] == "crossover/random-walk"
    assert client.get("/api/explain/spread/dosing").get_json()["prev"] is None


def test_app_icon_is_served_and_linked(client):
    ico = client.get("/favicon.ico")
    assert ico.status_code == 200 and ico.data[:4] == b"\x00\x00\x01\x00"          # ICO header
    page = client.get("/").get_data(as_text=True)
    for f in ("favicon.svg", "favicon-32.png", "apple-touch-icon.png", "site.webmanifest"):
        assert f in page
        assert client.get(f"/static/icons/{f}").status_code == 200
