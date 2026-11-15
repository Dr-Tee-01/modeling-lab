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

__all__ = ["EXPLAINERS", "FIGURES", "FigureParamError", "Params", "build", "check_figure", "explainer", "figure",
           "index", "problems", "render_figure"]
