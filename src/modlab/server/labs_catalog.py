"""Publishes the four labs (names, order, descriptions) in GET /api/catalog under "labs"."""
from __future__ import annotations

from .. import labs as L
from .catalog import _PROVIDERS, section


@section("labs")
def _labs() -> dict:
    markets, crossover, agents = _PROVIDERS["markets"](), _PROVIDERS["crossover"](), _PROVIDERS["agents"]()
    blurbs = {
        "modeling": f"{len(L.LABS[0]['views'])} models from the MC312 mathematical-modelling labs: dosing, adoption, "
                    "epidemics, rumours and randomness, solved with RK4 and Monte Carlo on every request.",
        "geometry": "The ShapeLab library on real outlines and meshes: Fourier descriptors, statistical shape models, "
                    "Laplace–Beltrami spectra, curvature and skinning.",
        "timeseries": f"Walk-forward forecasting of returns and volatility for {len(markets['tickers'])} stocks, "
                      f"{markets['first_year']}–{markets['last_year']}, plus random-walk tests, fear cleared like a drug "
                      f"dose and the shapes of {crossover['crash_count']} market crashes.",
        "rl": f"Play Snake, watch {len(agents['agents'])} agents play it, train the original project's deep Q-network "
              "live, and compare them on the same boards.",
    }
    return {"labs": [{**lab, "blurb": blurbs[lab["id"]]} for lab in L.LABS]}
