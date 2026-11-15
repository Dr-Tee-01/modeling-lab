"""/api/explain: the long-form explainer pages, with every figure computed on the server."""
from __future__ import annotations

from flask import Blueprint, request

from .. import explain as X
from .api import ApiError, endpoint

bp = Blueprint("explain", __name__, url_prefix="/api/explain")


@bp.get("")
@endpoint
def index():
    """Every explainer: key, title, summary and reading time."""
    return {"explainers": X.index()}


@bp.get("/figure/<fid>")
@endpoint
def figure(fid):
    """One explainer figure, recomputed with the query parameters (the figure's sliders)."""
    try:
        return X.render_figure(fid, dict(request.args))
    except KeyError:
        raise ApiError(f"There is no figure '{fid}'.", "not_found", 404)
    except X.FigureParamError as e:
        raise ApiError(str(e), "invalid_parameter")


@bp.get("/<group>/<view>")
@endpoint
def page(group, view):
    """The explainer for one model view, with its figures computed at their default settings."""
    key = f"{group}/{view}"
    if key not in X.EXPLAINERS:
        raise ApiError(f"There is no explanation for '{key}' yet.", "not_found", 404)
    try:
        return X.build(key)
    except X.FigureParamError as e:
        raise ApiError(str(e), "invalid_parameter")
