"""/api/bigdata: the Big Data Lab simulations.

The blueprint is shared; each topic adds its routes in its own module named
``bigdata_routes_<topic>.py`` (``from .api_bigdata import bp``), imported below.
"""
from __future__ import annotations

import importlib
import os
import pkgutil

from flask import Blueprint

bp = Blueprint("bigdata", __name__, url_prefix="/api/bigdata")

for _m in sorted(m.name for m in pkgutil.iter_modules([os.path.dirname(__file__)])):
    if _m.startswith("bigdata_routes_"):
        importlib.import_module(f"{__package__}.{_m}")
