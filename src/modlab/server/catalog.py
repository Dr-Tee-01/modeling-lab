"""Reference data the interface needs before it can draw anything.

The browser holds no data of its own: lists of tickers, meshes and outlines, model and
agent names, dataset sizes and date ranges all come from here. Each API module
registers a provider for its section with ``@section("name")``; ``GET /api/catalog``
returns ``{name: provider()}`` for every section.
"""
from __future__ import annotations

from typing import Callable

_PROVIDERS: dict[str, Callable[[], dict]] = {}


def section(name: str):
    def deco(fn: Callable[[], dict]):
        _PROVIDERS[name] = fn
        return fn
    return deco


def build() -> dict:
    return {name: fn() for name, fn in _PROVIDERS.items()}
