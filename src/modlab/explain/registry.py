"""Explainer pages: long-form, illustrated explanations of each model, served as data.

An explainer is a dict of sections made of typed blocks. Text blocks hold the prose;
figure blocks name a *figure* that the server computes on request from the same
numerics as the live models, so every chart in an explanation is real output, and
figures with ``controls`` recompute when the reader moves a slider.

Block types (``{"type": ..., ...}``)
------------------------------------
p         {"text"}                                   paragraph; inline <b> <i> <em> <strong> <code> <sub> <sup> <br>
figure    {"figure", "title", "caption", "params", "controls": [...]}
          a control is a slider {k, label, min, max, step, value, unit?}
          or a choice {k, label, options: [[value, text], ...], value}
eq        {"text", "where": [[symbol, meaning], ...]}   equation(s), one per line
callout   {"tone": "info"|"tip"|"warn", "title", "text"}
steps     {"title"?, "items": [text, ...]}           numbered walkthrough
cards     {"items": [{"title", "text"}]}             key ideas side by side
compare   {"items": [{"title", "points": [text]}]}   two or three columns to contrast
list      {"items": [text, ...]}
quiz      {"q", "options": [text], "answer": index, "why"}
try       {"label", "params": {...}, "note"?}        opens the live model with these settings

Figure kinds (what a figure function returns)
---------------------------------------------
lines    {x, series: [{name, color, y, dash?, area?, thin?, legend?: false, hover?: false}], xlabel, ylabel,
          xlog?, ylog?, ymin?, ymax?, xdate?: true (x in fractional years), markers?: [{x, label}],
          hlines?: [{y, label?, color?}], points?: [{x, y, color, label?}], bands?: [{x0, x1, label}], height?}
bars     {labels, series: [{name, color, values}], ylabel, hlines?: [{y, label?, color?}], height?}
scatter  {points: [{x, y, color, r?, label?, name? (hover title)}], lines?: [{x, y, color, dash?, name?}],
          xlabel, ylabel, equal?, xdate?, hlines?, height?}
Any figure may give ``legend: [{name, color, dash?}]`` to set its legend explicitly.
paths    {paths: [{points: [[x, y]], color, closed?, width?, dash?, fill?, name?}], dots?: [{x, y, color, label?}],
          equal?, axes?, xlabel?, ylabel?, height?}
flow     {height, nodes: [{id, x, y, w, h, title, sub?, value?, hot?, fill?, fill_color?}],
          edges: [{from, to, label?, below?, width?, flow?, color?, curve?}], texts?: [{x, y, text, cls?}]}
grid     {cols, rows, cells (flat, row-major), palette: "seq"|"div"|"cat", colors?: [token], height?,
          xlabel?, ylabel?, xticks?: [[col, text]], yticks?: [[row, text]]}
stats    {items: [{label, value, note?, tone?}]}
Every figure may add ``notes``: short sentences computed from the numbers it shows.
Colours are theme tokens: s1 s2 s3 s4 neutral accent good warn bad.
"""
from __future__ import annotations

import copy
import functools
import math
import re
from typing import Callable

BLOCK_TYPES = {"p", "figure", "eq", "callout", "steps", "cards", "compare", "list", "quiz", "try"}
FIGURE_KINDS = {"lines", "bars", "scatter", "paths", "flow", "grid", "stats"}
COLORS = {"s1", "s2", "s3", "s4", "neutral", "accent", "good", "warn", "bad"}

FIGURES: dict[str, Callable] = {}
EXPLAINERS: dict[str, dict] = {}


class FigureParamError(ValueError):
    pass


class Params:
    """Validated access to a figure's parameters (strings from a query, or Python values)."""

    def __init__(self, raw: dict | None = None):
        self.raw = dict(raw or {})

    def num(self, name: str, default: float, lo: float, hi: float, integer: bool = False):
        v = self.raw.get(name, default)
        if v is None or v == "":
            v = default
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise FigureParamError(f"Parameter '{name}' must be a number.")
        if not math.isfinite(v) or not lo <= v <= hi:
            raise FigureParamError(f"Parameter '{name}' must be between {lo:g} and {hi:g}.")
        return int(round(v)) if integer else v

    def choice(self, name: str, default: str, options):
        v = str(self.raw.get(name, default))
        if v not in options:
            raise FigureParamError(f"Parameter '{name}' must be one of {', '.join(options)}.")
        return v

    def flag(self, name: str, default: bool = False) -> bool:
        v = self.raw.get(name, default)
        if isinstance(v, bool):
            return v
        return str(v).lower() in ("1", "true", "yes", "on")


def figure(fid: str):
    """Register a figure function ``fn(p: Params) -> dict``."""
    def deco(fn):
        if fid in FIGURES:
            raise ValueError(f"Figure '{fid}' is registered twice.")
        FIGURES[fid] = fn
        return fn
    return deco


def explainer(key: str, content: dict, order: int = 0) -> None:
    """Register the explainer for view ``group/view``. ``order`` is the view's position in its
    section's sidebar (1-based), which sets the previous/next links."""
    EXPLAINERS[key] = {**content, "_order": order}


@functools.lru_cache(maxsize=512)
def _render(fid: str, frozen: tuple) -> dict:
    out = FIGURES[fid](Params(dict(frozen)))
    out.setdefault("kind", "lines")
    return out


def render_figure(fid: str, params: dict | None = None) -> dict:
    if fid not in FIGURES:
        raise KeyError(fid)
    return copy.deepcopy(_render(fid, tuple(sorted((k, str(v)) for k, v in (params or {}).items()))))


_TAGS = re.compile(r"<[^>]+>")


def _words(o) -> int:
    if isinstance(o, str):
        return len(_TAGS.sub(" ", o).split())
    if isinstance(o, dict):
        return sum(_words(v) for k, v in o.items() if k not in ("figure", "type", "params", "controls", "data", "tone"))
    if isinstance(o, list):
        return sum(_words(v) for v in o)
    return 0


def _neighbours(key: str):
    """Previous and next explainer within the same lab (see modlab.labs), in sidebar order."""
    from ..labs import ORDER, lab_of
    lab = lab_of(key)
    if lab:
        keys = sorted((k for k in EXPLAINERS if k in lab["views"]), key=lambda k: ORDER[k])
    else:
        group = key.split("/")[0]
        keys = sorted((k for k in EXPLAINERS if k.startswith(group + "/")), key=lambda k: EXPLAINERS[k]["_order"])
    i = keys.index(key)
    pick = lambda k: k and {"key": k, "title": EXPLAINERS[k]["title"]}
    return pick(keys[i - 1] if i > 0 else None), pick(keys[i + 1] if i + 1 < len(keys) else None)


def build(key: str) -> dict:
    """The full explainer with every figure computed at its default settings."""
    page = copy.deepcopy(EXPLAINERS[key])
    page.pop("_order", None)
    for sec in page["sections"]:
        for b in sec["blocks"]:
            if b["type"] == "figure":
                params = {c["k"]: c["value"] for c in b.get("controls", [])}
                params.update(b.get("params", {}))
                b["data"] = render_figure(b["figure"], params)
    page["key"] = key
    page["minutes"] = max(1, math.ceil(_words(EXPLAINERS[key]) / 200))
    page["prev"], page["next"] = _neighbours(key)
    return page


def index() -> list[dict]:
    return [{"key": k, "title": e["title"], "summary": e["summary"], "kicker": e["kicker"],
             "minutes": max(1, math.ceil(_words(e) / 200))}
            for k, e in sorted(EXPLAINERS.items(), key=lambda kv: _sort_key(kv[0]))]


def _sort_key(key: str):
    from ..labs import ORDER
    return ORDER.get(key, (99, EXPLAINERS[key]["_order"]))


def problems(key: str) -> list[str]:
    """Structural checks used by the tests: unknown blocks, missing figures, bad controls."""
    e, out = EXPLAINERS[key], []
    for f in ("title", "kicker", "summary", "sections", "takeaways"):
        if not e.get(f):
            out.append(f"{key}: missing {f}")
    for sec in e.get("sections", []):
        if not sec.get("id") or not sec.get("title") or not sec.get("blocks"):
            out.append(f"{key}: section without id/title/blocks")
        for b in sec.get("blocks", []):
            t = b.get("type")
            if t not in BLOCK_TYPES:
                out.append(f"{key}: unknown block {t!r}")
            if t == "figure":
                if b.get("figure") not in FIGURES:
                    out.append(f"{key}: unknown figure {b.get('figure')!r}")
                for c in b.get("controls", []):
                    if "options" in c:
                        if not all(x in c for x in ("k", "label", "value")) or c["value"] not in [o[0] for o in c["options"]]:
                            out.append(f"{key}: bad choice control {c}")
                    elif not all(x in c for x in ("k", "label", "min", "max", "step", "value")) or not c["min"] <= c["value"] <= c["max"]:
                        out.append(f"{key}: bad control {c}")
            if t == "quiz" and not 0 <= b.get("answer", -1) < len(b.get("options", [])):
                out.append(f"{key}: quiz answer out of range")
            if t == "try" and not isinstance(b.get("params", {}), dict):
                out.append(f"{key}: try params must be a dict")
    return out


def check_figure(fig: dict) -> list[str]:
    out = []
    if fig.get("kind") not in FIGURE_KINDS:
        out.append(f"unknown kind {fig.get('kind')!r}")
    for s in fig.get("series", []) + fig.get("paths", []):
        if s.get("color", "s1") not in COLORS:
            out.append(f"unknown colour {s.get('color')!r}")
    if fig.get("kind") == "lines":
        n = len(fig.get("x", []))
        out += [f"series {s.get('name')!r} has {len(s['y'])} values for {n} x" for s in fig.get("series", []) if len(s["y"]) != n]
    return out
