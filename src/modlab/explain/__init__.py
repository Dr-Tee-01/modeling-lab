"""Explainer pages for every model view (see registry.py for the content format)."""
from .registry import (EXPLAINERS, FIGURES, FigureParamError, Params, build, check_figure, explainer, figure,
                       index, problems, render_figure)
from . import spread  # noqa: F401  (each module registers its explainers and figures)

for _name in ("shape", "markets", "crossover", "agents"):
    try:
        __import__(f"{__name__}.{_name}")
    except ModuleNotFoundError as e:          # a section whose explainers are not written yet
        if e.name != f"{__name__}.{_name}":
            raise

# Big Data Lab explainers: one module per topic, named bigdata_<topic>.py.
import pkgutil as _pkgutil

for _m in sorted(m.name for m in _pkgutil.iter_modules(__path__)):
    if _m.startswith("bigdata_"):
        __import__(f"{__name__}.{_m}")

__all__ = ["EXPLAINERS", "FIGURES", "FigureParamError", "Params", "build", "check_figure", "explainer", "figure",
           "index", "problems", "render_figure"]
