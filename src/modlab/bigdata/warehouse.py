"""Data warehousing, OLAP and data cubes: the engine behind the three warehouse views.

Everything here works on the course datasets in ``datasets.py`` (imported, never retyped):

* OLAP operations on a cube with concept hierarchies: roll-up and drill-down (along a hierarchy or
  all the way to the apex), slice, dice and pivot, each returned as a list of *steps* so every
  intermediate cross-tab, its cuboid C(i,j,k) and the SQL that produces it can be shown.
* SQL grouping operators: GROUP BY, ROLLUP, CUBE and GROUPING SETS evaluated row by row with NULL as
  the ALL marker and GROUPING() bits, the 2^n UNION-of-GROUP-BYs construction, CUBE as the union of
  rotated ROLLUPs, the real-NULL pitfall and the corrected versions of the deck's SQL.
* The lattice of cuboids: live cuboid sizes, concept hierarchies (prod(L_i + 1) cuboids), layer-by-
  layer computation from the smallest parent, a query router and HRU greedy view selection.
* Window functions (tumbling and sliding frames, RANK, LAG) on the monthly lab-supplies table.

Grouping is done with pandas (``groupby(..., observed=True)``) for the OLAP cube, and in plain Python
dictionaries for the SQL operators, because a real NULL must stay a group of its own there.

Synthetic extension (lattice view only, documented): for 4 and 5 dimensions the H&K sales cube gets a
``supplier`` dimension (SUP1-SUP3, as on DW slide 20 / Lattice slide 5) and a ``channel`` dimension
(shop/online); every (city, quarter, item) cell is split into integer parts with weights drawn from
``numpy.random.default_rng(494)``, so all sums are unchanged. FACT30 gets the same two dimensions by
assigning each of its 30 rows a seeded supplier and channel (amounts unchanged).
"""
from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import datasets as D

SEED = 494
PRIMES = ["", "′", "″", "‴"]
TIME_YEAR = "Q1–Q4"            # the H&K table covers one year; its single 'year' member


# ===================================================================== datasets as cubes
@dataclass(frozen=True)
class Dim:
    key: str
    label: str
    levels: tuple            # finest first; index len(levels) means ALL
    exprs: tuple             # SQL expression for each level
    symbol: str              # i, j, k, ... in C(i,j,k)

    @property
    def top(self) -> int:
        return len(self.levels)


@dataclass(frozen=True)
class Cube:
    key: str
    label: str
    source: str
    measure: str
    unit: str
    dims: tuple
    table: str               # FROM clause, e.g. "sales s"
    sql_measure: str
    joins: tuple             # ((alias, JOIN clause), ...) added when an expression uses alias + "."
    rows_dim: str            # dimension shown on the rows by default
    frame: pd.DataFrame = field(compare=False, repr=False)

    def dim(self, key: str) -> Dim:
        return next(d for d in self.dims if d.key == key)

    def pos(self, key: str) -> int:
        return [d.key for d in self.dims].index(key)


def _cat(df: pd.DataFrame, cols_orders: dict) -> pd.DataFrame:
    for c, order in cols_orders.items():
        df[c] = pd.Categorical(df[c], categories=list(dict.fromkeys(order)), ordered=True)
    return df


def _hk_frame() -> pd.DataFrame:
    rows = []
    for city, grid in D.HK_SALES.items():
        prov, country = D.LOCATION[city]
        for qi, q in enumerate(D.HK_QUARTERS):
            for ii, item in enumerate(D.HK_ITEMS):
                rows.append((city, prov, country, q, TIME_YEAR, item, grid[qi][ii]))
    df = pd.DataFrame(rows, columns=["city", "province", "country", "quarter", "year", "item", "sales"])
    return _cat(df, {"city": list(D.HK_SALES), "province": [v[0] for v in D.LOCATION.values()],
                     "country": [v[1] for v in D.LOCATION.values()], "quarter": D.HK_QUARTERS,
                     "year": [TIME_YEAR], "item": D.HK_ITEMS})


def _enrol_frame() -> pd.DataFrame:
    rows = [(c, deg, n) for c, vals in D.ENROL.items() for deg, n in zip(D.DEGREES, vals)]
    return _cat(pd.DataFrame(rows, columns=["country", "degree", "students"]),
                {"country": list(D.ENROL), "degree": D.DEGREES})


def _fact30_frame() -> pd.DataFrame:
    df = pd.DataFrame(D.FACT30, columns=["item", "store", "cust", "amount"])
    return _cat(df, {c: sorted(df[c].unique()) for c in ("item", "store", "cust")})


MONTH_QUARTER = {"Jan": "Q1", "Feb": "Q1", "Mar": "Q1", "Apr": "Q2", "May": "Q2", "Jun": "Q2",
                 "Jul": "Q3", "Aug": "Q3", "Sep": "Q3", "Oct": "Q4", "Nov": "Q4", "Dec": "Q4"}


def _lab_frame() -> pd.DataFrame:
    rows = [(item, m, MONTH_QUARTER[m], v) for item, vals in D.PIVOT_SALES.items() for m, v in zip(D.PIVOT_MONTHS, vals)]
    df = pd.DataFrame(rows, columns=["item", "month", "quarter", "amount"])
    return _cat(df, {"item": list(D.PIVOT_SALES), "month": D.PIVOT_MONTHS,
                     "quarter": [MONTH_QUARTER[m] for m in D.PIVOT_MONTHS]})


CUBES: dict[str, Cube] = {
    "hk": Cube("hk", "Sales by city, quarter and item", "DW deck slide 18 (Han & Kamber)", "sales", "$ thousand",
               (Dim("location", "Location", ("city", "province", "country"), ("s.city", "l.province", "l.country"), "i"),
                Dim("time", "Time", ("quarter", "year"), ("s.quarter", "t.year"), "j"),
                Dim("item", "Item", ("item",), ("s.item",), "k")),
               "sales s", "s.sales",
               (("l", "JOIN location l ON l.city = s.city"), ("t", "JOIN time t ON t.quarter = s.quarter")),
               "time", _hk_frame()),
    "enrol": Cube("enrol", "Enrolments by country and degree", "DW deck slide 30 (Gupta)", "students", "students",
                  (Dim("country", "Country", ("country",), ("country",), "i"),
                   Dim("degree", "Degree", ("degree",), ("degree",), "j")),
                  "enrolment", "students", (), "country", _enrol_frame()),
    "fact30": Cube("fact30", "30-row sales fact table", "DW deck slide 29", "amount", "amount",
                   (Dim("item", "Item", ("item",), ("itemid",), "i"),
                    Dim("store", "Store", ("store",), ("storeid",), "j"),
                    Dim("cust", "Customer", ("cust",), ("custid",), "k")),
                   "sales", "amount", (), "item", _fact30_frame()),
    "lab": Cube("lab", "Lab supplies by item and month", "DW deck slide 32", "amount", "amount",
                (Dim("item", "Item", ("item",), ("p.item",), "i"),
                 Dim("month", "Month", ("month", "quarter"), ("p.month", "c.quarter"), "j")),
                "supplies p", "p.amount", (("c", "JOIN calendar c ON c.month = p.month"),), "item", _lab_frame()),
}

# What the slides print, used as the "expected output" to check the computed results against.
SLIDE_EXPECT = {
    "vancouver": {"slide": "DW slide 17", "rows": D.HK_QUARTERS, "cols": D.HK_ITEMS,
                  "cells": [[605, 825, 14, 400], [680, 952, 31, 512], [812, 1023, 30, 501], [927, 1038, 38, 580]]},
    "enrol_rows": {"slide": "DW slide 30", "values": {"Australia": 197, "India": 128, "Malaysia": 115, "Singapore": 104,
                                                     "Sweden": 78, "UK": 147, "USA": 94}},
    "enrol_cols": {"slide": "DW slide 30", "values": {"BSc": 71, "LLB": 68, "MBBS": 192, "BCom": 315, "BIT": 217}},
    "enrol_total": {"slide": "DW slide 30", "value": 863},
    "f30_total": {"slide": "DW slide 27", "value": 3350},
    "f30_items": {"slide": "DW slide 27", "values": {"item1": 135, "item2": 1325, "item3": 780, "item4": 655, "item5": 455}},
    "f30_item2": {"slide": "DW slide 27", "values": {"s1": 150, "s2": 680, "s3": 310, "s6": 185}},
    "f30_item_store": {"slide": "DW slide 28", "values": {
        ("item1", "s1"): 70, ("item1", "s2"): 65, ("item2", "s1"): 150, ("item2", "s2"): 680, ("item2", "s3"): 310,
        ("item2", "s6"): 185, ("item3", "s1"): 105, ("item3", "s2"): 60, ("item3", "s3"): 315, ("item3", "s4"): 135,
        ("item3", "s6"): 165, ("item4", "s4"): 90, ("item4", "s5"): 420, ("item4", "s6"): 145, ("item5", "s5"): 155,
        ("item5", "s6"): 300}},
    "fact9_cube": {"slide": "SQL deck slide 13", "rows": 33, "total": 735},
    "fact9_rollup": {"slide": "SQL deck slide 24", "rows": 19, "total": 735},
    "lab_cscl": {"slide": "DW slide 32", "values": [870, 3884, 9000, 9792, 11691, 14625, 15068, 13230, 7290]},
}

DEFAULT_SLICE = {"hk": {"location": "Vancouver", "time": "Q1", "item": "computer"},
                 "enrol": {"degree": "BIT", "country": "India"},
                 "fact30": {"item": "item2", "store": "s1", "cust": "c2"},
                 "lab": {"item": "Phenol", "month": "Jul"}}
DEFAULT_DICE = {"hk": {"location": ["Toronto", "Vancouver"], "time": ["Q1", "Q2"], "item": ["home entertainment", "computer"]},
                "enrol": {"degree": ["BCom", "BIT"], "country": ["Australia", "India", "Malaysia"]},
                "fact30": {"store": ["s1", "s3"], "item": ["item2", "item4"]},
                "lab": {"item": ["CSCL", "Phenol"], "month": ["Jun", "Jul", "Aug"]}}
OPS = {"rollup": "Roll-up", "drill": "Drill-down", "slice": "Slice", "dice": "Dice", "pivot": "Pivot", "window": "Window"}


def num(v):
    """Integers stay integers; everything else is rounded for display."""
    if v is None:
        return None
    f = float(v)
    return int(round(f)) if abs(f - round(f)) < 1e-9 else round(f, 4)


def sq(v) -> str:
    return "'" + str(v).replace("'", "''") + "'"


def members(cube: Cube, dim: str, level: int = 0) -> list[str]:
    col = cube.dim(dim).levels[level]
    present = set(cube.frame[col].astype(str))
    return [c for c in cube.frame[col].cat.categories if c in present]


# ===================================================================== cuboids, tables and SQL
@functools.lru_cache(maxsize=2048)
def group(ck: str, levels: tuple, where: tuple = ()) -> pd.DataFrame:
    """Aggregate the cube at the given level of each dimension (index == top means ALL).
    ``where`` is ((dim, level, (values...)), ...). The cached frame must not be mutated."""
    cube = CUBES[ck]
    df = cube.frame
    for dk, lvl, vals in where:
        df = df[df[cube.dim(dk).levels[lvl]].astype(str).isin(vals)]
    cols = [d.levels[l] for d, l in zip(cube.dims, levels) if l < d.top]
    if not cols:
        return pd.DataFrame({cube.measure: [df[cube.measure].sum()]})
    return df.groupby(cols, observed=True, sort=True)[cube.measure].sum().reset_index()


def notation(cube: Cube, levels: tuple) -> str:
    return "C(" + ",".join(d.symbol + PRIMES[l] if l < d.top else "0" for d, l in zip(cube.dims, levels)) + ")"


def level_names(cube: Cube, levels: tuple) -> dict:
    return {d.key: (d.levels[l] if l < d.top else "ALL") for d, l in zip(cube.dims, levels)}


def node_id(levels) -> str:
    return "-".join(str(int(l)) for l in levels)


def _where_text(cube: Cube, where: tuple) -> str:
    parts = []
    for dk, lvl, vals in where:
        col = cube.dim(dk).levels[lvl]
        parts.append(f"{col} = {vals[0]}" if len(vals) == 1 else f"{col} ∈ {{{', '.join(vals)}}}")
    return " and ".join(parts)


def _sql_where(cube: Cube, where: tuple) -> list[str]:
    out = []
    for dk, lvl, vals in where:
        e = cube.dim(dk).exprs[lvl]
        out.append(f"{e} = {sq(vals[0])}" if len(vals) == 1 else f"{e} IN ({', '.join(sq(v) for v in vals)})")
    return out


def _joins(cube: Cube, exprs: list[str]) -> list[str]:
    return [clause for alias, clause in cube.joins if any(e.startswith(alias + ".") for e in exprs)]


def sql(cube: Cube, levels: tuple, where: tuple = (), pivot: tuple | None = None) -> str:
    """The SQL that produces this cuboid from the fact table (star joins added only when needed).
    pivot = (row_dim_key, col_dim_key): conditional aggregation, the portable form of PIVOT/crosstab."""
    groups = [d.exprs[l] for d, l in zip(cube.dims, levels) if l < d.top]
    conds = _sql_where(cube, where)
    used = groups + [c.split(" ")[0] for c in conds]
    m = cube.sql_measure
    if pivot:
        rd, cd = cube.dim(pivot[0]), cube.dim(pivot[1])
        re_, ce = rd.exprs[levels[cube.pos(rd.key)]], cd.exprs[levels[cube.pos(cd.key)]]
        mem = members(cube, cd.key, levels[cube.pos(cd.key)])
        shown = mem[:4]
        lines = [f"SELECT {re_},"] + [f"       SUM(CASE WHEN {ce} = {sq(v)} THEN {m} END) AS \"{v}\"," for v in shown]
        if len(mem) > len(shown):
            lines.append(f"       -- … one SUM(CASE …) per {cd.levels[levels[cube.pos(cd.key)]]}: {len(mem)} columns in all")
        lines[-1] = lines[-1].rstrip(",")
        lines.append(f"FROM {cube.table}")
        lines += _joins(cube, [re_, ce] + used)
        if conds:
            lines.append("WHERE " + " AND ".join(conds))
        lines.append(f"GROUP BY {re_}")
        lines.append(f"ORDER BY {re_};")
        return "\n".join(lines)
    lines = [f"SELECT {', '.join(groups + [f'SUM({m}) AS {cube.measure}'])}", f"FROM {cube.table}"]
    lines += _joins(cube, used)
    if conds:
        lines.append("WHERE " + "\n  AND ".join(conds))
    if groups:
        lines.append(f"GROUP BY {', '.join(groups)}")
        lines.append(f"ORDER BY {', '.join(groups)}")
    lines[-1] += ";"
    return "\n".join(lines)


def table(ck: str, levels: tuple, where: tuple = (), rows: str | None = None, force: str | None = None) -> dict:
    """A cuboid laid out for display: a scalar, a list, a cross-tab (rows x nested columns) or a relation."""
    cube = CUBES[ck]
    g = group(ck, levels, where)
    m = cube.measure
    total = num(g[m].sum())
    present = [(d, l) for d, l in zip(cube.dims, levels) if l < d.top]
    base = {"measure": m, "unit": cube.unit, "total": total, "count": int(len(g))}
    if not present:
        return {**base, "view": "scalar"}
    want = rows or cube.rows_dim
    present.sort(key=lambda p: (p[0].key != want, cube.pos(p[0].key)))
    cols = [d.levels[l] for d, l in present]
    key_vals = [tuple(str(v) for v in r) for r in g[cols].itertuples(index=False)]
    vals = [num(v) for v in g[m]]
    labels = [[c for c in g[col].cat.categories if c in set(g[col].astype(str))] for col in cols]
    heads = [{"dim": d.key, "level": d.levels[l], "labels": lab} for (d, l), lab in zip(present, labels)]
    lookup = dict(zip(key_vals, vals))
    if len(present) == 1 and force != "relation":
        return {**base, "view": "list", "rows": heads[0], "values": [lookup.get((r,)) for r in labels[0]]}
    col_keys = [list(k) for k in itertools.product(*labels[1:])] if len(present) > 1 else []
    density = len(g) / max(1, len(labels[0]) * max(1, len(col_keys)))
    relation = force == "relation" or len(present) >= 4 or (len(present) == 3 and (len(col_keys) > 24 or density < 0.5))
    if relation:
        order = sorted(range(len(key_vals)), key=lambda i: tuple(labels[j].index(key_vals[i][j]) for j in range(len(cols))))
        return {**base, "view": "relation", "columns": cols + [m], "dims": [d.key for d, _ in present],
                "rows": [list(key_vals[i]) + [vals[i]] for i in order]}
    cells = [[lookup.get(tuple([r] + k)) for k in col_keys] for r in labels[0]]
    return {**base, "view": "crosstab", "rows": heads[0], "cols": heads[1:], "col_keys": col_keys, "cells": cells,
            "row_totals": [num(sum(v for v in row if v is not None)) for row in cells],
            "col_totals": [num(sum(row[j] for row in cells if row[j] is not None)) for j in range(len(col_keys))]}


def step(ck: str, levels: tuple, where: tuple = (), title: str = "", rows: str | None = None, force: str | None = None,
         pivot: tuple | None = None, note: str = "", reads: str | None = None) -> dict:
    cube = CUBES[ck]
    t = table(ck, levels, where, rows, force)
    names = level_names(cube, levels)
    return {"title": title, "cuboid": notation(cube, levels) + (f" | {_where_text(cube, where)}" if where else ""),
            "node": node_id(levels), "reads": reads or node_id(levels), "levels": names,
            "group_by": [v for v in names.values() if v != "ALL"], "filter": _where_text(cube, where),
            "sql": sql(cube, levels, where, pivot), "table": t, "cells": t["count"], "total": t["total"], "note": note}


def lattice_layout(level_counts: tuple, labeler=None) -> dict:
    """Nodes for every combination of levels (0..L_i, L_i meaning ALL), layered by aggregation depth:
    base cuboid at the bottom (y = 1), apex at the top (y = 0). Edges go from parent to child (one level up)."""
    combos = list(itertools.product(*[range(L + 1) for L in level_counts]))
    maxd = sum(level_counts) or 1
    layers: dict[int, list] = {}
    for c in combos:
        layers.setdefault(sum(c), []).append(c)
    nodes = []
    for depth, cs in layers.items():
        cs.sort(key=lambda c: tuple(-x for x in c))
        for i, c in enumerate(cs):
            nodes.append({"id": node_id(c), "levels": list(c), "depth": depth, "x": (i + 1) / (len(cs) + 1),
                          "y": 1 - depth / maxd, "label": labeler(c) if labeler else node_id(c)})
    edges = []
    for c in combos:
        for i, L in enumerate(level_counts):
            if c[i] < L:
                child = c[:i] + (c[i] + 1,) + c[i + 1:]
                edges.append([node_id(c), node_id(child)])
    return {"nodes": nodes, "edges": edges, "count": len(combos), "layers": len(layers)}


# ===================================================================== OLAP operations
def _transition(cube: Cube, a: tuple, b: tuple) -> str:
    for d, x, y in zip(cube.dims, a, b):
        if x == y:
            continue
        name = lambda l: d.levels[l] if l < d.top else "ALL"
        if y > x:
            return (f"Roll up {d.key}: {name(x)} → ALL (aggregate {d.key} away)" if y == d.top
                    else f"Roll up {d.key}: {name(x)} → {name(y)}")
        return (f"Drill down: introduce {d.key} at {name(y)}" if x == d.top
                else f"Drill down {d.key}: {name(x)} → {name(y)}")
    return "No change"


def olap_path(cube: Cube, op: str, dim: str, path: str) -> list[tuple]:
    """The sequence of level tuples visited by a roll-up or drill-down."""
    base = tuple(0 for _ in cube.dims)
    x = cube.pos(dim)
    X = cube.dims[x]
    if path == "dim":
        seq = [base[:x] + (l,) + base[x + 1:] for l in range(X.top + 1)]
    else:   # all the way to the apex: drill = X first (down its hierarchy), then the other dims in order
        cur = [d.top for d in cube.dims]
        seq = [tuple(cur)]
        for l in range(X.top - 1, -1, -1):
            cur[x] = l
            seq.append(tuple(cur))
        for i, d in enumerate(cube.dims):
            if i != x:
                cur[i] = 0
                seq.append(tuple(cur))
        seq.reverse()                       # base ... apex
    return seq if op == "rollup" else list(reversed(seq))


def parse_dice(cube: Cube, raw: str) -> dict:
    """'dim:m1|m2;dim2:m3' -> {dim: [members]} keeping only valid members; empty -> the default dice;
    '*' -> no restriction at all (the dice keeps the whole cube)."""
    if (raw or "").strip() == "*":
        return {}
    sel = {}
    for part in (raw or "").split(";"):
        if ":" not in part:
            continue
        dk, vals = part.split(":", 1)
        dk = dk.strip()
        if dk not in [d.key for d in cube.dims]:
            continue
        ok = members(cube, dk)
        picked = [v for v in ok if v in [x.strip() for x in vals.split("|")]]
        if picked:
            sel[dk] = picked
    return sel or {k: list(v) for k, v in DEFAULT_DICE[cube.key].items()}


def olap(ck: str, op: str, dim: str, path: str = "dim", member: str = "", dice: str = "") -> dict:
    cube = CUBES[ck]
    if dim not in [d.key for d in cube.dims]:
        dim = cube.dims[0].key
    base = tuple(0 for _ in cube.dims)
    steps, pick = [], {"mode": None}
    if op in ("rollup", "drill"):
        seq = olap_path(cube, op, dim, path)
        steps.append(step(ck, seq[0], title=("Start: the base cuboid" if op == "rollup" else "Start: the apex (grand total)")
                          if seq[0] in (base, tuple(d.top for d in cube.dims)) else "Start"))
        for a, b in zip(seq, seq[1:]):
            steps.append(step(ck, b, title=_transition(cube, a, b)))
    elif op == "slice":
        mem = members(cube, dim)
        member = member if member in mem else DEFAULT_SLICE[ck].get(dim, mem[0])
        x = cube.pos(dim)
        after = base[:x] + (cube.dim(dim).top,) + base[x + 1:]
        steps.append(step(ck, base, title="Start: the base cuboid"))
        steps.append(step(ck, after, where=((dim, 0, (member,)),), reads=node_id(base),
                          title=f"Slice {dim} = {member}: fix one member, the dimension disappears"))
        steps[-1]["cuboid"] = f"{notation(cube, base)} | {steps[-1]['filter']}"      # a slice reads part of the base cuboid
        pick = {"mode": "slice", "dims": [dim], "selected": {dim: [member]}}
    elif op == "dice":
        sel = parse_dice(cube, dice)
        where = tuple((dk, 0, tuple(v)) for dk, v in sel.items())
        steps.append(step(ck, base, title="Start: the base cuboid"))
        steps.append(step(ck, base, where=where, title="Dice: keep a sub-cube (a set of members on each chosen dimension)"))
        pick = {"mode": "dice", "dims": [d.key for d in cube.dims], "selected": sel}
    elif op == "pivot":
        first = cube.rows_dim if cube.rows_dim != dim else next(d.key for d in cube.dims if d.key != dim)
        other = lambda r: next(d.key for d in cube.dims if d.key != r)
        steps.append(step(ck, base, force="relation", title="Start: the result as a relation (one row per cell)"))
        steps.append(step(ck, base, rows=first, pivot=(first, other(first)),
                          title=f"Pivot: {first} on the rows, the other dimension{'s' if len(cube.dims) > 2 else ''} across"))
        steps.append(step(ck, base, rows=dim, pivot=(dim, other(dim)), title=f"Rotate: {dim} on the rows"))
    else:
        raise ValueError(f"unknown operation {op!r}")

    first, last = steps[0], steps[-1]
    lat = lattice_layout(tuple(d.top for d in cube.dims),
                         lambda c: notation(cube, c))
    for n in lat["nodes"]:
        n["cells"] = int(len(group(ck, tuple(n["levels"]))))
        n["title"] = ", ".join(v for v in level_names(cube, tuple(n["levels"])).values() if v != "ALL") or "all"
    visited = list(dict.fromkeys(s["reads"] for s in steps))
    bars = _bars(last)
    return {"ds": cube_info(cube), "op": op, "dim": dim, "path": path, "steps": steps, "pick": pick,
            "members": {d.key: members(cube, d.key) for d in cube.dims},
            "lattice": {**lat, "visited": visited, "final": last["reads"]},
            "bars": bars, "before_cells": first["cells"], "after_cells": last["cells"],
            "before_total": first["total"], "after_total": last["total"],
            "preserved": op in ("rollup", "drill", "pivot") and first["total"] == last["total"],
            "share": (last["total"] / first["total"]) if first["total"] else None,
            "checks": checks(ck, op, dim, steps), "notes": slide_notes(ck, op, dim, steps)}


def _bars(st: dict) -> dict:
    t = st["table"]
    if t["view"] == "scalar":
        return {"labels": ["ALL"], "values": [t["total"]], "title": "Grand total"}
    if t["view"] == "list":
        return {"labels": t["rows"]["labels"], "values": t["values"], "title": f"{t['measure']} by {t['rows']['level']}"}
    if t["view"] == "crosstab":
        return {"labels": t["rows"]["labels"], "values": t["row_totals"], "title": f"{t['measure']} by {t['rows']['level']} (row totals)"}
    # relation: totals by the first column
    agg: dict = {}
    for r in t["rows"]:
        agg[r[0]] = agg.get(r[0], 0) + r[-1]
    return {"labels": list(agg), "values": [num(v) for v in agg.values()], "title": f"{t['measure']} by {t['columns'][0]}"}


def _cells_of(st: dict) -> dict:
    """{(row, col...): value} for any table view."""
    t = st["table"]
    if t["view"] == "scalar":
        return {(): t["total"]}
    if t["view"] == "list":
        return {(r,): v for r, v in zip(t["rows"]["labels"], t["values"])}
    if t["view"] == "relation":
        return {tuple(r[:-1]): r[-1] for r in t["rows"]}
    out = {}
    for r, row in zip(t["rows"]["labels"], t["cells"]):
        for k, v in zip(t["col_keys"], row):
            if v is not None:
                out[tuple([r] + k)] = v
    return out


def checks(ck: str, op: str, dim: str, steps: list) -> list[dict]:
    """Compare the computed result with the numbers printed on the slides."""
    out, last = [], steps[-1]
    E = SLIDE_EXPECT
    add = lambda label, slide, expect, got: out.append({"label": label, "slide": slide, "expect": expect, "got": got, "ok": expect == got})
    if ck == "hk" and op == "slice" and dim == "location" and last["filter"] == "city = Vancouver":
        cells = _cells_of(last)
        got = [[cells.get((q, it)) for it in D.HK_ITEMS] for q in D.HK_QUARTERS]
        add("The Vancouver slice is the slide-17 '2-D table'", E["vancouver"]["slide"], E["vancouver"]["cells"], got)
    if ck == "enrol":
        if op == "slice" and dim == "degree" and last["filter"] == "degree = BIT":
            add("BIT slice total (slide 30 ALL row)", "DW slide 30", E["enrol_cols"]["values"]["BIT"], last["total"])
        if op == "dice":
            add("Every cell kept by the dice is the slide-30 value", "DW slide 30", True,
                all(D.ENROL[c][D.DEGREES.index(d)] == v for (c, d), v in _cells_of(last).items()))
        for s in steps:
            if s["levels"].get("country") == "ALL" and s["levels"].get("degree") == "degree":
                vals = {k[0]: v for k, v in _cells_of(s).items()}
                add("Degree totals (slide 30 ALL row)", "DW slide 30", E["enrol_cols"]["values"], vals)
            if s["levels"].get("country") == "country" and s["levels"].get("degree") == "ALL" and not s["filter"]:
                vals = {k[0]: v for k, v in _cells_of(s).items()}
                add("Country totals (slide 30 ALL column)", "DW slide 30", E["enrol_rows"]["values"], vals)
        if op in ("rollup", "drill", "pivot"):
            add("Grand total", "DW slide 30", E["enrol_total"]["value"], steps[0]["total"] if op != "drill" else last["total"])
    if ck == "fact30":
        for s in steps:
            lv = s["levels"]
            if s["filter"]:
                continue
            if lv == {"item": "item", "store": "ALL", "cust": "ALL"}:
                add("Item totals", "DW slide 27", E["f30_items"]["values"], {k[0]: v for k, v in _cells_of(s).items()})
            if lv == {"item": "item", "store": "store", "cust": "ALL"}:
                add("Item × store amounts", "DW slide 28", {f"{a}·{b}": v for (a, b), v in E["f30_item_store"]["values"].items()},
                    {f"{a}·{b}": v for (a, b), v in _cells_of(s).items()})
            if lv == {"item": "ALL", "store": "ALL", "cust": "ALL"}:
                add("Total sales", "DW slide 27", E["f30_total"]["value"], s["total"])
        if op == "slice" and last["filter"] == "item = item2":
            by_store: dict = {}
            for k, v in _cells_of(last).items():
                by_store[k[0]] = by_store.get(k[0], 0) + v
            add("item2 by store (slide 27 zoom)", "DW slide 27", E["f30_item2"]["values"], by_store)
    if ck == "lab" and op == "pivot":
        st = steps[-1] if steps[-1]["table"].get("rows", {}).get("dim") == "item" else steps[1]
        cells = _cells_of(st)
        if st["table"]["view"] == "crosstab" and st["table"]["rows"]["dim"] == "item":
            add("CSCL row of the pivoted table", "DW slide 32", E["lab_cscl"]["values"], [cells.get(("CSCL", m)) for m in D.PIVOT_MONTHS])
    return out


def slide_notes(ck: str, op: str, dim: str, steps: list) -> list[str]:
    cube, out = CUBES[ck], []
    if ck == "hk" and op in ("rollup", "drill") and dim == "location":
        out.append("Each province in the slide-18 data has exactly one city (Victoria, Ottawa, Buffalo and Urbana have no rows), "
                   "so city → province only renames the rows; the totals change at province → country.")
    if ck == "hk" and op in ("rollup", "drill") and dim == "time":
        out.append("The table covers a single year, so quarter → year gives the same numbers as quarter → ALL.")
    if ck == "hk" and op == "slice" and dim == "location":
        out.append("DW slide 17 calls its table a '2-D cube' of time × item; it is the location = Vancouver slice of the slide-18 cube.")
    if ck == "enrol" and op == "slice":
        out.append("Slide 30 shades one column of a semester × country × degree cube; only the 'ALL semesters' face is given, "
                   "so the BIT slice here is a vector of seven countries rather than a country × semester plane.")
    if ck == "fact30" and op in ("rollup", "drill") and dim == "item" and len(steps) == 4:
        out.append("This is the slide 27–29 drill-down (3350 → items → item × store → the fact rows) and, read backwards, "
                   "the slide-22 roll-up {item, store, cust} → {item, store} → {item} → {}.")
    return out


def cube_info(cube: Cube) -> dict:
    return {"key": cube.key, "label": cube.label, "source": cube.source, "measure": cube.measure, "unit": cube.unit,
            "rows": int(len(cube.frame)), "total": num(cube.frame[cube.measure].sum()),
            "dims": [{"key": d.key, "label": d.label, "levels": list(d.levels), "symbol": d.symbol} for d in cube.dims]}


# ===================================================================== window functions (DW slides 32-34)
def window(kind: str = "sliding", size: int = 3, agg: str = "sum", item: str = "CSCL") -> dict:
    months, items = list(D.PIVOT_MONTHS), list(D.PIVOT_SALES)
    if item not in items:
        item = items[0]
    A = np.array([D.PIVOT_SALES[i] for i in items], dtype=float)
    f = {"sum": np.sum, "avg": np.mean, "max": np.max}[agg]
    T = len(months)
    if kind == "sliding":
        frames = [(max(0, t - size + 1), t) for t in range(T)]
    else:
        frames = [((t // size) * size, min(T - 1, (t // size) * size + size - 1)) for t in range(T)]
    aggs = {it: [num(f(A[r, a:b + 1])) for a, b in frames] for r, it in enumerate(items)}
    windows = []
    if kind == "tumbling":
        for w, a in enumerate(range(0, T, size)):
            b = min(T - 1, a + size - 1)
            windows.append({"id": w + 1, "from": months[a], "to": months[b], "months": b - a + 1,
                            "values": {it: num(f(A[r, a:b + 1])) for r, it in enumerate(items)}})
    # RANK() OVER (PARTITION BY month ORDER BY amount DESC)   (ties share the lowest rank)
    rank = [[int(1 + np.sum(A[:, t] > A[r, t])) for t in range(T)] for r in range(len(items))]
    r = items.index(item)
    lag = [None] + [num(A[r, t - 1]) for t in range(1, T)]
    change = [None] + [num(A[r, t] - A[r, t - 1]) for t in range(1, T)]
    rows = [{"month": months[t], "amount": num(A[r, t]), "frame": f"{months[frames[t][0]]}–{months[frames[t][1]]}",
             "frame_rows": frames[t][1] - frames[t][0] + 1, "agg": aggs[item][t], "lag": lag[t], "change": change[t],
             "rank": rank[r][t]} for t in range(T)]
    fn = agg.upper()
    over = (f"{fn}(amount) OVER (PARTITION BY item ORDER BY month_no\n"
            f"                          ROWS BETWEEN {size - 1} PRECEDING AND CURRENT ROW)") if kind == "sliding" else None
    sql_win = (f"SELECT item, month, amount,\n       {over} AS {agg}_{size}\nFROM supplies NATURAL JOIN calendar\nORDER BY item, month_no;"
               if kind == "sliding" else
               f"SELECT item, (month_no - 3) / {size} AS win,      -- integer division: windows of {size} months from Mar\n"
               f"       MIN(month) AS first_month, {fn}(amount) AS {agg}_{size}\nFROM supplies NATURAL JOIN calendar\n"
               f"GROUP BY item, (month_no - 3) / {size}\nORDER BY item, win;")
    best = int(np.argmax(np.array(aggs[item], dtype=float)))
    return {"kind": kind, "size": size, "agg": agg, "item": item, "items": items, "months": months,
            "amount": {it: [num(v) for v in A[i]] for i, it in enumerate(items)}, "aggs": aggs, "windows": windows,
            "rank": rank, "rows": rows, "partial": sum(1 for a, b in frames if b - a + 1 < size),
            "best": {"month": months[best], "value": aggs[item][best]},
            "biggest_rise": max((x for x in rows if x["change"] is not None), key=lambda x: x["change"]),
            "biggest_fall": min((x for x in rows if x["change"] is not None), key=lambda x: x["change"]),
            "sql": {"window": sql_win,
                    "rank": "SELECT month, item, amount,\n       RANK() OVER (PARTITION BY month ORDER BY amount DESC) AS rank_in_month\n"
                            "FROM supplies NATURAL JOIN calendar\nORDER BY month_no, rank_in_month;",
                    "lag": "SELECT item, month, amount,\n       LAG(amount) OVER (PARTITION BY item ORDER BY month_no) AS prev_month,\n"
                           "       amount - LAG(amount) OVER (PARTITION BY item ORDER BY month_no) AS change\n"
                           "FROM supplies NATURAL JOIN calendar\nORDER BY item, month_no;"}}


# ===================================================================== SQL grouping operators (SQL deck)
@dataclass(frozen=True)
class Fact:
    key: str
    label: str
    source: str
    dims: tuple          # column names in table order
    measure: str
    rows: tuple          # ((dim values...), measure)
    sql_cols: tuple      # SQL column names for the dims
    null_row: tuple      # a fact row whose customer is genuinely unknown (NULL), for the pitfall demo


FACTS = {
    "fact9": Fact("fact9", "Sales, 9 rows", "DW slide 24 / SQL deck slide 13", ("store", "item", "cust"), "price",
                  tuple(((s, i, c), p) for s, i, c, p in D.FACT9), ("storeID", "itemID", "custID"), (("s2", "i1", None), 20)),
    "fact30": Fact("fact30", "Sales, 30 rows", "DW slide 29", ("item", "store", "cust"), "amount",
                   tuple(((i, s, c), a) for i, s, c, a in D.FACT30), ("itemid", "storeid", "custid"), (("item1", "s2", None), 20)),
}
GSETS = {
    "pairs": ("Every pair", lambda o: [tuple(x) for x in itertools.combinations(o, 2)]),
    "singles": ("Each dimension alone, plus ()", lambda o: [(d,) for d in o] + [()]),
    "margins": ("First two dimensions and their margins", lambda o: [tuple(o[:2]), (o[0],), (o[1],), ()] if len(o) >= 2 else [tuple(o), ()]),
}


def fact_rows(fk: str, with_null: bool = False) -> list[tuple]:
    f = FACTS[fk]
    return list(f.rows) + ([f.null_row] if with_null else [])


def sets_for(op: str, order: list[str], gs: str = "pairs") -> list[tuple]:
    n = len(order)
    if op == "groupby":
        return [tuple(order)]
    if op == "rollup":
        return [tuple(order[:i]) for i in range(n, -1, -1)]
    if op == "cube":
        return [tuple(c) for r in range(n, -1, -1) for c in itertools.combinations(order, r)]
    if op == "gsets":
        return GSETS[gs][1](order)
    raise ValueError(op)


def evaluate(fk: str, order: list[str], sets: list[tuple], with_null: bool = False, dedup: bool = False) -> list[dict]:
    """Evaluate GROUP BY GROUPING SETS (sets) row by row. Columns not in a set are NULL (None) and
    have GROUPING() = 1. A real NULL in the data is a group of its own with GROUPING() = 0."""
    f = FACTS[fk]
    idx = [f.dims.index(d) for d in order]
    out = []
    for si, s in enumerate(sets):
        agg: dict = {}
        for vals, m in fact_rows(fk, with_null):
            key = tuple(vals[idx[j]] if order[j] in s else None for j in range(len(order)))
            agg[key] = agg.get(key, 0) + m
        bits = [0 if d in s else 1 for d in order]
        gid = int("".join(map(str, bits)), 2) if bits else 0
        for key in sorted(agg, key=lambda k: tuple(("~" if v is None else v) for v in k)):
            out.append({"vals": list(key), "grouping": bits, "gid": gid, "set": si, "amount": num(agg[key]),
                        "real_null": [key[j] is None and bits[j] == 0 for j in range(len(order))]})
    if dedup:      # UNION (not UNION ALL): rows identical in every displayed column collapse
        seen, kept = set(), []
        for r in out:
            k = (tuple(r["vals"]), r["amount"])
            if k not in seen:
                seen.add(k)
                kept.append(r)
        out = kept
    return out


def _multiset(rows: list[dict]) -> list:
    return sorted((tuple("NULL" if v is None else v for v in r["vals"]), r["amount"]) for r in rows)


def rotations(order: list[str]) -> list[list[str]]:
    """Slide 25's rotated orders: (a,b,c), (b,c,a), (c,a,b)."""
    return [order[i:] + order[:i] for i in range(len(order))]


def rotated_rollup_coverage(n: int) -> dict:
    dims = [chr(97 + i) for i in range(n)]
    got = {frozenset(s) for o in rotations(dims) for s in sets_for("rollup", o)}
    allsets = {frozenset(c) for r in range(n + 1) for c in itertools.combinations(dims, r)}
    missing = sorted(("".join(sorted(s)) or "()") for s in allsets - got)
    return {"n": n, "covered": len(got), "all": 2 ** n, "missing": missing}


def _sel(cols: list[str], s: tuple, order: list[str]) -> str:
    return ", ".join(c if d in s else f"NULL AS {c}" for c, d in zip(cols, order))


def grouping_sql(fk: str, op: str, order: list[str], sets: list[tuple]) -> dict:
    f = FACTS[fk]
    cols = [f.sql_cols[f.dims.index(d)] for d in order]
    m = f.measure
    gcols = ", ".join(cols)
    gbits = ", ".join(f"GROUPING({c}) AS g_{d}" for c, d in zip(cols, order))
    if op == "groupby":
        native = f"SELECT {gcols}, SUM({m}) AS total\nFROM sales\nGROUP BY {gcols};"
    elif op == "gsets":
        spec = ", ".join("(" + ", ".join(cols[order.index(d)] for d in s) + ")" for s in sets)
        native = f"SELECT {gcols},\n       {gbits},\n       SUM({m}) AS total\nFROM sales\nGROUP BY GROUPING SETS ({spec});"
    else:
        native = (f"SELECT {gcols},\n       {gbits},\n       SUM({m}) AS total\nFROM sales\n"
                  f"GROUP BY {op.upper()} ({gcols});")
    parts = []
    for s in sets:
        gb = ", ".join(cols[order.index(d)] for d in s)
        parts.append(f"SELECT {_sel(cols, s, order)}, SUM({m}) AS total FROM sales" + (f" GROUP BY {gb}" if gb else ""))
    union = "\nUNION ALL\n".join(parts) + ";"
    rot = rotations(order)
    rollups = "\nUNION\n".join(f"SELECT {gcols}, SUM({m}) AS total FROM sales\nGROUP BY ROLLUP ({', '.join(cols[order.index(d)] for d in o)})"
                               for o in rot) + ";"
    mysql = (f"SELECT {gcols}, SUM({m}) FROM sales\nGROUP BY {gcols} WITH ROLLUP;   -- MySQL / SQL Server spelling")
    cube = f"SELECT {gcols},\n       {gbits},\n       SUM({m}) AS total\nFROM sales\nGROUP BY CUBE ({gcols});"
    return {"native": native, "cube": cube, "union": union, "rollups": rollups, "mysql": mysql}


def grouping(fk: str = "fact9", op: str = "cube", order: list[str] | None = None, gs: str = "pairs",
             with_null: bool = False) -> dict:
    f = FACTS[fk]
    order = list(order or f.dims)
    sets = sets_for(op, order, gs)
    rows = evaluate(fk, order, sets, with_null)
    total = num(sum(m for _, m in fact_rows(fk, with_null)))
    # the three constructions of CUBE on the same dimensions
    cube_rows = evaluate(fk, order, sets_for("cube", order), with_null)
    union_rows = evaluate(fk, order, sets_for("cube", order), with_null)      # 2^n GROUP BYs, UNION ALL
    rot_sets = [s for o in rotations(order) for s in sets_for("rollup", o)]
    rot_all = evaluate(fk, order, [tuple(d for d in order if d in s) for s in rot_sets], with_null)
    rot_union = evaluate(fk, order, [tuple(d for d in order if d in s) for s in rot_sets], with_null, dedup=True)
    per_set = [{"set": list(s), "label": "(" + ", ".join(s) + ")" if s else "()", "rows": sum(1 for r in rows if r["set"] == i),
                "bits": [0 if d in s else 1 for d in order], "total": num(sum(r["amount"] for r in rows if r["set"] == i))}
               for i, s in enumerate(sets)]
    # lattice of all subsets of the chosen dimensions, with the sets this operator produces highlighted
    lat = lattice_layout(tuple(1 for _ in order), lambda c: ", ".join(d for d, l in zip(order, c) if l == 0) or "()")
    chosen = {node_id(tuple(0 if d in s else 1 for d in order)) for s in sets}
    for n in lat["nodes"]:
        s = tuple(d for d, l in zip(order, n["levels"]) if l == 0)
        n["rows"] = len({tuple(v[f.dims.index(d)] for d in s) for v, _ in fact_rows(fk, with_null)}) if s else 1
        n["on"] = n["id"] in chosen
    nulls = [r for r in rows if any(r["real_null"])]
    ambiguous = []
    for r in nulls:
        twin = next((o for o in rows if o is not r and o["vals"] == r["vals"]), None)
        if twin:
            ambiguous.append({"vals": r["vals"], "data_row": r["amount"], "subtotal": twin["amount"],
                              "grouping_data": r["grouping"], "grouping_subtotal": twin["grouping"]})
    return {"fact": {"key": f.key, "label": f.label, "source": f.source, "dims": list(f.dims), "measure": f.measure,
                     "rows": len(fact_rows(fk, with_null)), "total": total},
            "op": op, "order": order, "sets": per_set, "rows": rows, "row_count": len(rows),
            "result_total": num(sum(r["amount"] for r in rows)),
            "apex": next((r["amount"] for r in rows if all(r["grouping"]) and r["grouping"]), None),
            "lattice": lat, "sql": grouping_sql(fk, op, order, sets),
            "constructions": {
                "cube_rows": len(cube_rows), "union_rows": len(union_rows),
                "union_equal": _multiset(union_rows) == _multiset(cube_rows),
                "rotations": [list(o) for o in rotations(order)],
                "rot_sets": len({frozenset(s) for s in rot_sets}), "rot_union_all": len(rot_all), "rot_union": len(rot_union),
                "rot_equal": _multiset(rot_union) == _multiset(cube_rows),
                "coverage": [rotated_rollup_coverage(n) for n in range(2, 7)]},
            "with_null": with_null, "null_rows": len(nulls), "ambiguous": ambiguous,
            "bugs": deck_bugs(), "schema": schema()}


def deck_bugs() -> list[dict]:
    """The deck's queries against the materialised FACT9 cube (33 rows), next to corrected versions.
    The predicates are evaluated here on the same rows the SQL would scan."""
    rows = [dict(zip(("store", "item", "cust"), r["vals"]), amount=r["amount"])
            for r in evaluate("fact9", ["store", "item", "cust"], sets_for("cube", ["store", "item", "cust"]))]
    nn = lambda r, *ks: all(r[k] is not None for k in ks)
    isnull = lambda r, *ks: all(r[k] is None for k in ks)
    count = lambda p: sum(1 for r in rows if p(r))
    cases = [
        ("SQL deck slide 18", "item axis",
         "SELECT itemid, amount FROM salesCube\nWHERE storeid IS NULL AND custid IS NULL",
         "Also returns the apex row (itemid NULL too): the item axis needs itemid IS NOT NULL.",
         count(lambda r: isnull(r, "store", "cust")),
         "SELECT itemid, amount FROM salesCube\nWHERE itemid IS NOT NULL AND storeid IS NULL AND custid IS NULL",
         count(lambda r: nn(r, "item") and isnull(r, "store", "cust"))),
        ("SQL deck slide 18", "item × store",
         "SELECT itemid, storeid, amount FROM salesCube\nWHERE custid IS NULL\nORDER BY itemid, storedid",
         "Mixes four cuboids (item×store, item, store, apex); 'storedid' is not a column.",
         count(lambda r: isnull(r, "cust")),
         "SELECT itemid, storeid, amount FROM salesCube\nWHERE itemid IS NOT NULL AND storeid IS NOT NULL AND custid IS NULL\nORDER BY itemid, storeid",
         count(lambda r: nn(r, "item", "store") and isnull(r, "cust"))),
        ("SQL deck slide 18", "item × store × customer",
         "SELECT itemid, storeid, custid, amount FROM salesCube\nORDER BY itemid, storedid, custid",
         "No WHERE: returns the whole cube, not the base cuboid.",
         len(rows),
         "SELECT itemid, storeid, custid, amount FROM salesCube\nWHERE itemid IS NOT NULL AND storeid IS NOT NULL AND custid IS NOT NULL",
         count(lambda r: nn(r, "item", "store", "cust"))),
        ("SQL deck slide 26", "slice cust = c2",
         "SELECT itemid, storeid, custid, amount FROM salesCube\nWHERE custid = 'c2'",
         "Rows from four cuboids that contain custid are mixed together.",
         count(lambda r: r["cust"] == "c2"),
         "SELECT itemid, storeid, amount FROM salesCube\nWHERE custid = 'c2' AND itemid IS NOT NULL AND storeid IS NOT NULL",
         count(lambda r: r["cust"] == "c2" and nn(r, "item", "store"))),
        ("SQL deck slides 28, 30", "dice store s1, item i2",
         "SELECT * FROM salesCube\nWHERE storeid = 's1 ' AND itemid = 'i2 '",
         "Trailing spaces inside the literals: no row matches on PostgreSQL text/varchar (MySQL and SQL Server pad and ignore them).",
         count(lambda r: r["store"] == "s1 " and r["item"] == "i2 "),
         "SELECT * FROM salesCube\nWHERE storeid = 's1' AND itemid = 'i2' AND custid IS NOT NULL",
         count(lambda r: r["store"] == "s1" and r["item"] == "i2" and nn(r, "cust"))),
        ("SQL deck slide 30", "dice from the fact table",
         "SELECT f.itemid, f.storeid, custid, SUM(price)\nFROM sales NATURAL JOIN store NATURAL JOIN item\nWHERE state = 'CA' AND category = 'Jacket'\nGROUP BY f.itemid, f.storeid, custid",
         "The alias f is never defined: 'missing FROM-clause entry for table f'.",
         "error",
         "SELECT f.itemid, f.storeid, f.custid, SUM(f.price)\nFROM sales f NATURAL JOIN store NATURAL JOIN item\nWHERE state = 'CA' AND category = 'Jacket'\nGROUP BY f.itemid, f.storeid, f.custid",
         "runs"),
        ("SQL deck slide 10", "base cuboid",
         "SELECT itemID,storeID,custID sum(price)\nFROM Sales GROUP BY itemID,storeID,CustID",
         "Missing comma before sum(price): a syntax error.",
         "error",
         "SELECT itemID, storeID, custID, SUM(price)\nFROM Sales GROUP BY itemID, storeID, custID",
         count(lambda r: nn(r, "item", "store", "cust"))),
        ("DW deck slide 23", "item axis with 0 as ALL",
         "SELECT itemID, sum_price FROM SalesCube\nWHERE itemID <> 0 AND storedID = 0 AND customerID = 0",
         "Uses 0 for ALL (the cube stores NULL) and misspells storeID; 'itemID <> 0' also compares text with a number.",
         "error",
         "SELECT itemID, sum_price FROM SalesCube\nWHERE itemID IS NOT NULL AND storeID IS NULL AND custID IS NULL",
         count(lambda r: nn(r, "item") and isnull(r, "store", "cust"))),
    ]
    return [{"slide": s, "query": q, "deck_sql": a, "problem": p, "deck_rows": dr, "fixed_sql": b, "fixed_rows": fr}
            for s, q, a, p, dr, b, fr in cases]


def schema() -> dict:
    """Widom's star schema (SQL deck slide 5) and the same dimensions normalised into a snowflake.
    Joins counts are the path lengths from the fact table to the table holding each attribute."""
    star = {"Sales": ["storeID", "itemID", "custID", "price", "qty"], "Store": ["storeID", "city", "county", "state"],
            "Item": ["itemID", "category", "color"], "Customer": ["custID", "cName", "gender", "age"]}
    star_edges = [["Sales", "Store", "storeID"], ["Sales", "Item", "itemID"], ["Sales", "Customer", "custID"]]
    snow = {"Sales": star["Sales"], "Store": ["storeID", "cityID"], "City": ["cityID", "city", "countyID"],
            "County": ["countyID", "county", "state"], "Item": ["itemID", "categoryID", "color"],
            "Category": ["categoryID", "category"], "Customer": star["Customer"]}
    snow_edges = star_edges + [["Store", "City", "cityID"], ["City", "County", "countyID"], ["Item", "Category", "categoryID"]]

    def joins(tables, edges, attr):
        dist, frontier = {"Sales": 0}, ["Sales"]
        while frontier:
            nxt = []
            for t in frontier:
                for a, b, _ in edges:
                    if a == t and b not in dist:
                        dist[b] = dist[t] + 1
                        nxt.append(b)
            frontier = nxt
        return min(dist[t] for t, cols in tables.items() if attr in cols)

    attrs = ["state", "city", "category", "gender"]
    return {"star": {"tables": star, "edges": star_edges, "joins": {a: joins(star, star_edges, a) for a in attrs}},
            "snowflake": {"tables": snow, "edges": snow_edges, "joins": {a: joins(snow, snow_edges, a) for a in attrs}},
            "attrs": attrs}


# ===================================================================== lattice, hierarchies, HRU
LDIMS = {
    "hk": [("time", ("quarter", "year")), ("item", ("item",)), ("location", ("city", "province", "country")),
           ("supplier", ("supplier",)), ("channel", ("channel",))],
    "fact30": [("item", ("item",)), ("store", ("store",)), ("cust", ("cust",)), ("supplier", ("supplier",)),
               ("channel", ("channel",))],
}
LSHORT = {"time": "T", "item": "I", "location": "L", "supplier": "S", "channel": "C", "store": "St", "cust": "Cu"}
SYNTHETIC = {"supplier", "channel"}


@functools.lru_cache(maxsize=4)
def lattice_frame(ck: str) -> pd.DataFrame:
    """The fact data used by the lattice view, with the documented seeded supplier and channel columns."""
    rng = np.random.default_rng(SEED)
    if ck == "hk":
        base = CUBES["hk"].frame
        out = []
        for r in base.itertuples(index=False):
            w = rng.integers(1, 10, size=3).astype(float)
            parts = np.floor(r.sales * w / w.sum()).astype(int)
            parts[int(np.argmax(w))] += r.sales - parts.sum()
            for si, p in enumerate(parts):
                if p <= 0:
                    continue
                online = int(round(p * rng.uniform(0.2, 0.6)))
                for ch, v in (("shop", p - online), ("online", online)):
                    if v > 0:
                        out.append((r.quarter, r.year, r.item, r.city, r.province, r.country, f"SUP{si + 1}", ch, v))
        return pd.DataFrame(out, columns=["quarter", "year", "item", "city", "province", "country", "supplier", "channel", "sales"])
    rows = []
    for item, store, cust, amt in D.FACT30:
        rows.append((item, store, cust, f"SUP{int(rng.integers(1, 4))}", ("shop", "online")[int(rng.integers(0, 2))], amt))
    return pd.DataFrame(rows, columns=["item", "store", "cust", "supplier", "channel", "amount"])


@functools.lru_cache(maxsize=64)
def _codes(ck: str) -> dict:
    df = lattice_frame(ck)
    return {c: pd.factorize(df[c])[0].astype(np.int64) for c in df.columns if df[c].dtype == object}


def _size(ck: str, cols: tuple) -> int:
    if not cols:
        return 1
    codes = _codes(ck)
    key = np.zeros(len(next(iter(codes.values()))), dtype=np.int64)
    for c in cols:
        key = key * (int(codes[c].max()) + 1) + codes[c]
    return int(len(np.unique(key)))


@functools.lru_cache(maxsize=64)
def lattice_sizes(ck: str, n: int) -> dict:
    """Distinct groups in every cuboid of the n-dimensional lattice (base levels), keyed by frozenset of dims."""
    dims = [d for d, _ in LDIMS[ck][:n]]
    lv = dict(LDIMS[ck])
    return {frozenset(s): _size(ck, tuple(lv[d][0] for d in s)) for r in range(n + 1) for s in itertools.combinations(dims, r)}


@functools.lru_cache(maxsize=64)
def hierarchy_lattice(ck: str, n: int) -> dict:
    """All prod(L_i + 1) cuboids when every dimension may sit at any level of its concept hierarchy."""
    dl = LDIMS[ck][:n]
    counts = tuple(len(l) for _, l in dl)
    by_depth: dict = {}
    sizes = []
    for combo in itertools.product(*[range(L + 1) for L in counts]):
        cols = tuple(dl[i][1][l] for i, l in enumerate(combo) if l < counts[i])
        sz = _size(ck, cols)
        sizes.append(sz)
        by_depth.setdefault(sum(combo), []).append(sz)
    return {"levels": [{"dim": d, "levels": list(l), "L": len(l)} for d, l in dl], "count": int(np.prod([L + 1 for L in counts])),
            "flat": 2 ** n, "by_depth": [{"depth": k, "cuboids": len(v), "rows": int(sum(v))} for k, v in sorted(by_depth.items())],
            "total_rows": int(sum(sizes))}


def _label(dims: tuple, all_dims: list, n: int) -> str:
    if not dims:
        return "all"
    ordered = [d for d in all_dims if d in dims]
    return ", ".join(ordered) if n <= 3 else "·".join(LSHORT[d] for d in ordered)


def layer_compute(ck: str, n: int) -> list[dict]:
    """Kylin-style layer-by-layer cubing: MR1 builds the base cuboid from the fact file, then each layer
    builds every cuboid with one dimension fewer from its smallest parent in the layer below."""
    dims = [d for d, _ in LDIMS[ck][:n]]
    sizes = lattice_sizes(ck, n)
    fact_rows = int(len(lattice_frame(ck)))
    out = [{"layer": 1, "dims": n, "cuboid": dims, "label": _label(tuple(dims), dims, n), "parent": "fact file",
            "parent_rows": fact_rows, "rows": sizes[frozenset(dims)], "key": "(" + ", ".join(dims) + ")"}]
    for r in range(n - 1, -1, -1):
        for s in itertools.combinations(dims, r):
            parents = [frozenset(s) | {d} for d in dims if d not in s]
            best = min(parents, key=lambda p: (sizes[p], sorted(p)))
            out.append({"layer": n - r + 1, "dims": r, "cuboid": list(s), "label": _label(s, dims, n),
                        "parent": _label(tuple(d for d in dims if d in best), dims, n), "parent_rows": sizes[best],
                        "rows": sizes[frozenset(s)], "key": "(" + ", ".join(s) + ")" if s else "()"})
    return out


def hru(sizes: dict, k: int) -> dict:
    """Harinarayan-Rajaraman-Ullman greedy view selection under the linear cost model: a query on view w
    costs the size of its smallest materialised ancestor; the base cuboid is always materialised."""
    views = sorted(sizes, key=lambda v: (-len(v), sorted(v)))
    top = max(views, key=len)
    S = [top]
    cost = lambda w, S_: min(sizes[v] for v in S_ if w <= v)
    total = lambda S_: sum(cost(w, S_) for w in views)
    rounds, totals = [], [total(S)]
    for _ in range(k):
        bens = {}
        for v in views:
            if v in S:
                continue
            bens[v] = sum(max(0, cost(w, S) - sizes[v]) for w in views if w <= v)
        if not bens:
            break
        best = max(bens, key=lambda v: (bens[v], -sizes[v], -len(v)))
        rounds.append({"pick": sorted(best), "benefit": bens[best], "size": sizes[best],
                       "benefits": {",".join(sorted(v)): b for v, b in bens.items()}})
        S.append(best)
        totals.append(total(S))
    return {"chosen": [sorted(v) for v in S[1:]], "rounds": rounds, "totals": totals, "S": S}


def route(sizes: dict, materialised: list, need: frozenset) -> dict:
    anc = sorted((v for v in materialised if need <= v), key=lambda v: (sizes[v], sorted(v)))
    return {"candidates": [{"dims": sorted(v), "rows": sizes[v]} for v in anc], "chosen": sorted(anc[0]), "cost": sizes[anc[0]]}


def lattice(ck: str = "hk", n: int = 4, k: int = 3, query: list[str] | None = None, where: bool = False,
            kmax: int = 10, mode: str = "all") -> dict:
    dl = LDIMS[ck]
    n = max(2, min(5, n))
    dims = [d for d, _ in dl[:n]]
    sizes = lattice_sizes(ck, n)
    q = [d for d in (query or []) if d in dims]
    if not q and query is None:
        q = [d for d in ("item", "location") if d in dims] or dims[1:2]
    filt = dims[0] if where else None
    need = frozenset(q) | ({filt} if filt else set())
    H = hru(sizes, min(kmax, max(k, 0), len(sizes) - 1))
    full = hru(sizes, min(kmax, len(sizes) - 1))
    mat = list(sizes) if mode == "all" else H["S"]
    r = route(sizes, mat, need)
    r_hru = route(sizes, H["S"], need)
    base_rows = sizes[frozenset(dims)]
    lat = lattice_layout(tuple(1 for _ in dims))
    chosen_ids, mat_ids = node_id(tuple(0 if d in r["chosen"] else 1 for d in dims)), set()
    for v in mat:
        mat_ids.add(node_id(tuple(0 if d in v else 1 for d in dims)))
    need_id = node_id(tuple(0 if d in need else 1 for d in dims))
    for nd in lat["nodes"]:
        s = tuple(d for d, l in zip(dims, nd["levels"]) if l == 0)
        nd["dims"] = list(s)
        nd["label"] = _label(s, dims, n)
        nd["rows"] = sizes[frozenset(s)]
        nd["materialised"] = nd["id"] in mat_ids
    fact_n = int(len(lattice_frame(ck)))
    layers = layer_compute(ck, n)
    lv = dict(dl)
    filt_val = None
    if filt:
        col = lv[filt][0]
        filt_val = str(lattice_frame(ck)[col].iloc[0])
    sel = ", ".join(q) if q else ""
    sql_q = (f"SELECT {sel + ', ' if sel else ''}SUM({'sales' if ck == 'hk' else 'amount'})\nFROM sales"
             + (f"\nWHERE {filt} = {sq(filt_val)}" if filt else "") + (f"\nGROUP BY {sel}" if sel else "") + ";")
    return {"ds": ck, "n": n, "dims": dims, "short": {d: LSHORT[d] for d in dims}, "synthetic": [d for d in dims if d in SYNTHETIC],
            "fact_rows": fact_n, "sizes": [{"dims": sorted(v), "rows": s} for v, s in sizes.items()],
            "lattice": {**lat, "need": need_id, "route": chosen_ids}, "base_rows": base_rows,
            "cuboids": 2 ** n, "hierarchy": hierarchy_lattice(ck, n),
            "layers": layers, "layer_reads": int(sum(x["parent_rows"] for x in layers)),
            "naive_reads": int(2 ** n * fact_n), "base_reads": int(fact_n + (2 ** n - 1) * base_rows),
            "query": {"group_by": q, "filter": filt, "filter_value": filt_val, "needs": sorted(need), "sql": sql_q,
                      "route": r, "route_hru": r_hru, "mode": mode, "materialised": len(mat), "base_cost": base_rows,
                      "speedup": base_rows / r["cost"] if r["cost"] else None},
            "hru": {"k": len(H["rounds"]), "chosen": H["chosen"], "rounds": H["rounds"], "totals": H["totals"],
                    "curve": full["totals"], "all_cost": int(sum(sizes.values())), "none_cost": int(len(sizes) * base_rows)}}


# ===================================================================== catalog facts
def catalog() -> dict:
    return {
        "cubes": {k: {**cube_info(c), "members": {d.key: members(c, d.key) for d in c.dims}} for k, c in CUBES.items()},
        "ops": OPS, "default_slice": DEFAULT_SLICE, "default_dice": DEFAULT_DICE,
        "window": {"items": list(D.PIVOT_SALES), "months": list(D.PIVOT_MONTHS)},
        "facts": {k: {"label": f.label, "source": f.source, "dims": list(f.dims), "measure": f.measure, "rows": len(f.rows),
                      "total": num(sum(m for _, m in f.rows))} for k, f in FACTS.items()},
        "gsets": {k: v[0] for k, v in GSETS.items()},
        "lattice": {k: {"dims": [d for d, _ in v], "levels": {d: list(l) for d, l in v}, "fact_rows": int(len(lattice_frame(k))),
                        "synthetic": sorted(SYNTHETIC)} for k, v in LDIMS.items()},
        "short": LSHORT, "seed": SEED,
        "orders": [",".join(p) for p in itertools.permutations(["item", "store", "cust"])],
    }
