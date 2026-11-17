"""/api/bigdata: data warehousing, OLAP operations, SQL grouping operators and the cuboid lattice.

All three endpoints are pure functions of their parameters over the course datasets and run in a few
milliseconds, so they compute on every request (no background jobs needed).
"""
from __future__ import annotations

from flask import request

from ..bigdata import warehouse as W
from .api import ApiError, choice, endpoint, flag, num
from .api_bigdata import bp
from .catalog import section


@section("bigdata_warehouse")
def _catalog() -> dict:
    return W.catalog()


def _text(name: str, limit: int) -> str:
    raw = request.args.get(name, "") if request.method == "GET" else str((request.get_json(silent=True) or {}).get(name, ""))
    if len(raw) > limit:
        raise ApiError(f"Parameter '{name}' is too long (at most {limit} characters).", "invalid_parameter")
    return raw


@bp.get("/olap")
@endpoint
def olap():
    """OLAP operations on a data cube: roll-up, drill-down, slice, dice, pivot, and window functions."""
    op = choice("op", "rollup", W.OPS)
    if op == "window":
        items = list(W.D.PIVOT_SALES)
        w = W.window(choice("wkind", "sliding", ["sliding", "tumbling"]), num("wsize", 3, 2, 4, integer=True),
                     choice("wagg", "sum", ["sum", "avg", "max"]), choice("witem", items[0], items))
        return {"op": "window", "window": w, "ds": W.cube_info(W.CUBES["lab"])}
    ds = choice("ds", "hk", W.CUBES)
    dims = [d.key for d in W.CUBES[ds].dims]
    return W.olap(ds, op, choice("dim", dims[0], dims), choice("path", "dim", ["dim", "apex"]),
                  member=_text("member", 60), dice=_text("dice", 400))


@bp.get("/cube")
@endpoint
def cube():
    """SQL grouping operators: GROUP BY, ROLLUP, CUBE and GROUPING SETS with NULL as ALL and GROUPING() bits."""
    fk = choice("fact", "fact9", W.FACTS)
    dims = list(W.FACTS[fk].dims)
    raw = _text("order", 60)
    order = [d.strip() for d in raw.split(",") if d.strip()] if raw else dims
    if not order or len(set(order)) != len(order) or any(d not in dims for d in order):
        raise ApiError(f"Parameter 'order' must list distinct dimensions from {', '.join(dims)}; got {raw!r}.", "invalid_parameter")
    return W.grouping(fk, choice("op", "cube", ["groupby", "rollup", "cube", "gsets"]), order,
                      choice("gs", "pairs", W.GSETS), flag("null", False))


@bp.get("/lattice")
@endpoint
def lattice():
    """The lattice of cuboids: live sizes, concept hierarchies, layer-by-layer cubing, a query router and HRU view selection."""
    ds = choice("ds", "hk", W.LDIMS)
    n = num("n", 4, 2, 5, integer=True)
    raw = request.args.get("q")
    dims = [d for d, _ in W.LDIMS[ds][:n]]
    if raw is None:
        q = None
    elif raw.strip() in ("", "all"):
        q = []
    else:
        q = [d.strip() for d in raw.split(",") if d.strip() in dims]
        if not q:          # the query used dimensions this lattice does not have: fall back to the default query
            q = None
    return W.lattice(ds, n, num("k", 3, 0, 10, integer=True), q, flag("where", False),
                     mode=choice("mode", "all", ["all", "hru"]))

