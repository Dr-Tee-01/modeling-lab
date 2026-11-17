"""Explainers for the Big Data Lab warehouse views: OLAP operations, the SQL grouping operators and the
lattice of cuboids. Every figure is computed by ``modlab.bigdata.warehouse`` from the course datasets."""
from __future__ import annotations

import math

from ..bigdata import datasets as D
from ..bigdata import warehouse as W
from .registry import explainer, figure

KICKER = "Big Data Lab · IE494 data warehousing"


def _f(v) -> str:
    return f"{v:,}" if isinstance(v, int) else f"{v:,.1f}"


def _norm(vals):
    top = max(v for v in vals if v is not None) or 1
    return [None if v is None else v / top for v in vals]


# ===================================================================== OLAP: numbers quoted in the prose
_HK_TOTAL = W.olap("hk", "rollup", "location")["steps"][0]["total"]
_COUNTRY = W.olap("hk", "rollup", "location")["steps"][2]
_BIT = W.olap("enrol", "slice", "degree", member="BIT")
_DICE = W.olap("enrol", "dice", "degree")
_DRILL = W.olap("fact30", "drill", "item", "apex")
_LAT_HK = W.olap("hk", "rollup", "location")["lattice"]


def _country_totals() -> dict:
    out: dict = {}
    for (q, c, it), v in W._cells_of(_COUNTRY).items():
        out[c] = out.get(c, 0) + v
    return out


_CT = _country_totals()


@figure("olap-cube")
def _fig_cube(p):
    """The slide-18 cube laid out as on the slide: quarters down, city × item across."""
    st = W.olap("hk", "rollup", "location")["steps"][0]
    t = st["table"]
    flat = [v for row in t["cells"] for v in row]
    cities, items = t["cols"][0]["labels"], t["cols"][1]["labels"]
    big = max(flat)
    r, c = divmod(flat.index(big), len(t["col_keys"]))
    small = min(flat)
    r2, c2 = divmod(flat.index(small), len(t["col_keys"]))
    return {"kind": "grid", "cols": len(t["col_keys"]), "rows": len(t["rows"]["labels"]), "cells": _norm(flat), "palette": "seq",
            "height": 170, "xlabel": "city × item (home ent., computer, phone, security)", "ylabel": "quarter",
            "xticks": [[i * len(items) + 1.5, c] for i, c in enumerate(cities)], "yticks": [[i, q] for i, q in enumerate(t["rows"]["labels"])],
            "notes": [f"{len(flat)} cells, {len(cities)} cities × {len(t['rows']['labels'])} quarters × {len(items)} item types; "
                      f"they add up to {_f(st['total'])} (thousand dollars).",
                      f"The darkest cell is {t['col_keys'][c][0]} {t['col_keys'][c][1]} in {t['rows']['labels'][r]} ({_f(big)}); "
                      f"the lightest is {t['col_keys'][c2][0]} {t['col_keys'][c2][1]} in {t['rows']['labels'][r2]} ({_f(small)})."]}


@figure("olap-slice")
def _fig_slice(p):
    city = p.choice("city", "Vancouver", list(D.HK_SALES))
    r = W.olap("hk", "slice", "location", member=city)
    st = r["steps"][-1]
    t = st["table"]
    flat = [v for row in t["cells"] for v in row]
    same = flat == [v for row in W.SLIDE_EXPECT["vancouver"]["cells"] for v in row]
    best_q = t["rows"]["labels"][max(range(len(t["row_totals"])), key=lambda i: t["row_totals"][i])]
    return {"kind": "grid", "cols": len(t["col_keys"]), "rows": len(t["rows"]["labels"]), "cells": _norm(flat), "palette": "seq",
            "height": 200, "xlabel": "item type", "ylabel": "quarter",
            "xticks": [[i, k[0][:10]] for i, k in enumerate(t["col_keys"])], "yticks": [[i, q] for i, q in enumerate(t["rows"]["labels"])],
            "notes": [f"The {city} slice keeps {st['cells']} of the {r['before_cells']} cells; its total is {_f(st['total'])}, "
                      f"{100 * r['share']:.1f}% of the cube.",
                      "These are exactly the 16 numbers of DW slide 17, which the slide calls a '2-D cube'." if same else
                      f"Slide 17 shows the Vancouver slice; {city}'s numbers differ from it.",
                      f"{city}'s best quarter is {best_q} with {_f(max(t['row_totals']))}."]}


@figure("olap-hierarchy")
def _fig_hierarchy(p):
    """The location hierarchy of Lattice slide 10 with each node's sales from the slide-18 cube."""
    city_tot = {c: sum(map(sum, g)) for c, g in D.HK_SALES.items()}
    provs, countries = {}, {}
    for c, (pr, co) in D.LOCATION.items():
        provs.setdefault(pr, (co, []))[1].append(c)
        countries.setdefault(co, [])
        if pr not in countries[co]:
            countries[co].append(pr)
    nodes, edges = [{"id": "all", "x": 10, "y": 150, "w": 90, "h": 56, "title": "ALL", "value": _f(sum(city_tot.values())), "hot": True}], []
    cy = {}
    for i, c in enumerate(D.LOCATION):
        y = 8 + i * 40
        has = c in city_tot
        nodes.append({"id": "c:" + c, "x": 440, "y": y, "w": 152, "h": 34, "title": c, "sub": _f(city_tot[c]) if has else "no rows",
                      "fill": 1 if has else 0, "fill_color": "s1"})
        cy[c] = y
    py = {}
    for pr, (co, cs) in provs.items():
        y = sum(cy[c] for c in cs) / len(cs) - 4
        tot = sum(city_tot.get(c, 0) for c in cs)
        nodes.append({"id": "p:" + pr, "x": 262, "y": y, "w": 150, "h": 42, "title": pr, "sub": _f(tot)})
        py[pr] = y
        for c in cs:
            edges.append({"from": "p:" + pr, "to": "c:" + c, "width": 1.4})
    for co, prs in countries.items():
        y = sum(py[x] for x in prs) / len(prs) - 4
        tot = sum(city_tot.get(c, 0) for pr in prs for c in provs[pr][1])
        nodes.append({"id": "k:" + co, "x": 128, "y": y, "w": 104, "h": 50, "title": co, "value": _f(tot)})
        edges.append({"from": "all", "to": "k:" + co, "width": 2.2})
        for pr in prs:
            edges.append({"from": "k:" + co, "to": "p:" + pr, "width": 1.8})
    empty = [c for c in D.LOCATION if c not in city_tot]
    return {"kind": "flow", "height": 330, "nodes": nodes, "edges": edges,
            "notes": [f"Canada {_f(_CT['Canada'])} + USA {_f(_CT['USA'])} = {_f(_HK_TOTAL)}: every level of the tree adds up to the same total.",
                      f"{', '.join(empty)} are in the hierarchy (slide 10) but have no sales on slide 18, so each province with data "
                      "has a single city and city → province changes only the labels."]}


@figure("olap-rollup")
def _fig_rollup(p):
    level = p.choice("level", "country", ["city", "province", "country", "all"])
    idx = ["city", "province", "country", "all"].index(level)
    st = W.step("hk", (idx, 2, 1))
    t = st["table"]
    if t["view"] == "scalar":
        labels, vals = ["ALL"], [t["total"]]
    else:
        labels, vals = t["rows"]["labels"], t["values"]
    return {"kind": "bars", "labels": labels, "series": [{"name": f"sales by {level}", "color": "s1", "values": vals}],
            "ylabel": "sales ($ thousand)", "height": 240,
            "notes": [f"GROUP BY {level if level != 'all' else 'nothing'}: {len(labels)} bar{'s' if len(labels) > 1 else ''} adding to {_f(st['total'])}.",
                      f"The cuboid is {st['cuboid']}; every roll-up keeps the total at {_f(_HK_TOTAL)}."]}


@figure("olap-enrol")
def _fig_enrol(p):
    op = p.choice("op", "slice", ["slice", "dice", "none"])
    sel = ({"degree": ["BIT"]} if op == "slice" else W.DEFAULT_DICE["enrol"] if op == "dice" else {})
    cells, kept = [], 0
    for c in D.ENROL:
        for j, d in enumerate(D.DEGREES):
            on = all(v in sel.get(k, [v]) for k, v in (("country", c), ("degree", d))) and bool(sel)
            cells.append(2 if on else 1)
            kept += D.ENROL[c][j] if on else 0
    note = {"slice": f"Slice degree = BIT: one column, {len(D.ENROL)} cells adding to {kept} (the 217 in slide 30's ALL row).",
            "dice": f"Dice degree ∈ {{BCom, BIT}} and country ∈ {{Australia, India, Malaysia}}: 6 cells adding to {kept}; "
                    "both dimensions stay.",
            "none": f"The whole table: {len(cells)} cells, {sum(map(sum, D.ENROL.values()))} enrolments."}[op]
    return {"kind": "grid", "cols": len(D.DEGREES), "rows": len(D.ENROL), "cells": cells, "palette": "cat",
            "colors": ["neutral", "neutral", "s2"], "height": 230, "xlabel": "degree", "ylabel": "country",
            "xticks": [[j, d] for j, d in enumerate(D.DEGREES)], "yticks": [[i, c] for i, c in enumerate(D.ENROL)],
            "legend": [{"name": "kept", "color": "s2"}, {"name": "dropped", "color": "neutral"}],
            "notes": [note, f"The BIT slice computed on the server totals {_BIT['after_total']}; the slide-31 dice totals {_DICE['after_total']}."]}


@figure("olap-pivot")
def _fig_pivot(p):
    items, months = list(D.PIVOT_SALES), D.PIVOT_MONTHS
    flat = [v for it in items for v in D.PIVOT_SALES[it]]
    peaks = [f"{it} peaks in {months[max(range(len(months)), key=lambda t: D.PIVOT_SALES[it][t])]}" for it in items]
    return {"kind": "grid", "cols": len(months), "rows": len(items), "cells": _norm(flat), "palette": "seq", "height": 160,
            "xlabel": "month", "ylabel": "item", "xticks": [[i, m] for i, m in enumerate(months)], "yticks": [[i, it[:9]] for i, it in enumerate(items)],
            "notes": [f"Pivoting the {len(flat)}-row (item, month, amount) relation gives this {len(items)} × {len(months)} table: one column per month.",
                      "; ".join(peaks) + "."]}


@figure("olap-window")
def _fig_window(p):
    kind = p.choice("kind", "sliding", ["sliding", "tumbling"])
    size = p.num("size", 3, 2, 4, integer=True)
    w = W.window(kind, size, "avg", "CSCL")
    x = list(range(len(w["months"])))
    return {"kind": "lines", "x": x, "xlabel": "month (Mar = 0)", "ylabel": "CSCL amount",
            "series": [{"name": "monthly amount", "color": "s1", "y": w["amount"]["CSCL"]},
                       {"name": f"AVG over the {kind} frame", "color": "s2", "y": w["aggs"]["CSCL"], "dash": True}],
            "markers": [{"x": i, "label": m} for i, m in enumerate(w["months"]) if i % 2 == 0],
            "notes": [(f"A sliding {size}-month average smooths the curve; its first {w['partial']} values use shorter frames."
                       if kind == "sliding" else
                       f"Tumbling windows give one value per block of {size} months ({len(w['windows'])} blocks), repeated across the block."),
                      f"Its highest value is {_f(w['best']['value'])}, for the frame ending in {w['best']['month']}."]}


@figure("olap-lattice")
def _fig_olap_lattice(p):
    lat = _LAT_HK
    W_, H = 600, 440
    widest = max(sum(1 for n in lat["nodes"] if n["y"] == y) for y in {n["y"] for n in lat["nodes"]})
    w = min(92, W_ / widest - 8)
    path = W.olap("hk", "rollup", "location")["lattice"]["visited"]
    nodes = [{"id": n["id"], "x": n["x"] * W_ - w / 2, "y": 10 + n["y"] * (H - 60), "w": w, "h": 40, "title": n["label"],
              "sub": f"{n['cells']} cells", "hot": n["id"] in path} for n in lat["nodes"]]
    edges = [{"from": a, "to": b, "width": 2.6 if a in path and b in path else 1, "color": "s2" if a in path and b in path else None}
             for a, b in lat["edges"]]
    return {"kind": "flow", "height": H, "nodes": nodes, "edges": [{k: v for k, v in e.items() if v is not None} for e in edges],
            "notes": [f"{lat['count']} cuboids = (3 + 1) × (2 + 1) × (1 + 1): location has 3 levels, time 2, item 1, plus ALL for each.",
                      "Highlighted: rolling location up from city (bottom) to ALL keeps quarter and item, passing "
                      f"{' → '.join(W.notation(W.CUBES['hk'], tuple(int(x) for x in v.split('-'))) for v in path)}."]}


explainer("bigdata/olap", {
    "title": "OLAP operations: rolling, slicing and rotating a data cube",
    "kicker": KICKER,
    "summary": f"A data warehouse keeps its numbers in a cube: sales by city, quarter and item, {_f(_HK_TOTAL)} thousand dollars in "
               "Han & Kamber's example. Every analytical question is one of five moves on that cube: roll up, drill down, slice, dice "
               "or pivot. This page does each one on the slide data, shows the cuboid it lands on and the SQL that computes it.",
    "sections": [
        {"id": "cube", "title": "The cube on the slides", "blocks": [
            {"type": "p", "text": "DW slide 18 records sales (in thousands of dollars) for four cities, four quarters and four item types. "
                                  "Each number is a <b>cell</b> of a three-dimensional cube; the dimensions are the axes and the "
                                  "sales figure is the <b>measure</b>."},
            {"type": "figure", "figure": "olap-cube", "title": "Slide 18 as a heat map",
             "caption": "Rows are quarters; the columns are grouped by city, four item types each. Darker = more sales."},
            {"type": "cards", "items": [
                {"title": "Dimensions", "text": "location, time and item: the things you group or filter by."},
                {"title": "Measure", "text": "sales: the number you add up. SUM is what every cell stores."},
                {"title": "Cuboid", "text": "the cube summarised at some level of each dimension. C(i,j,k) is the base; C(0,0,0) the grand total."},
            ]},
        ]},
        {"id": "slice", "title": "A slice: the slide-17 '2-D cube'", "blocks": [
            {"type": "p", "text": "Slide 17 shows a quarter × item table and calls it a 2-D cube. It is not a separate cube: it is the "
                                  "<b>slice</b> location = Vancouver of the slide-18 cube. Fixing one member of a dimension removes "
                                  "that dimension from the table."},
            {"type": "figure", "figure": "olap-slice", "title": "Slice the cube at one city",
             "caption": "Pick a city. The server filters the base cuboid and checks the Vancouver slice against slide 17.",
             "controls": [{"k": "city", "label": "City", "options": [[c, c] for c in D.HK_SALES], "value": "Vancouver"}]},
            {"type": "eq", "text": "slice   σ_{location = 'Vancouver'} C(i, j, k)\nSELECT quarter, item, SUM(sales) FROM sales\nWHERE city = 'Vancouver' GROUP BY quarter, item;",
             "where": [["σ", "selection: keep only the cells that satisfy the condition"], ["C(i, j, k)", "the base cuboid: city × quarter × item"],
                       ["GROUP BY", "the remaining dimensions (location is constant inside the slice)"]]},
            {"type": "try", "label": "Slice the cube at Vancouver", "params": {"op": "slice", "ds": "hk", "dim_hk": "location"},
             "note": "Click any other city in the first table to slice there instead."},
        ]},
        {"id": "hierarchy", "title": "Concept hierarchies", "blocks": [
            {"type": "p", "text": "A dimension usually has levels. Cities belong to provinces or states, which belong to countries "
                                  "(Lattice slide 10); quarters belong to years. Moving between levels is what roll-up and drill-down do."},
            {"type": "figure", "figure": "olap-hierarchy", "title": "The location hierarchy with its sales",
             "caption": "Each node shows the sales of everything below it, computed from the slide-18 cube."},
            {"type": "callout", "tone": "info", "title": "Strict and non-strict hierarchies",
             "text": "A hierarchy is safe to roll up only if every member has exactly one parent. City → province → country is. "
                     "DW slide 26 draws week → quarter, which is not: a week can straddle two months or two quarters, so summing weeks "
                     "into quarters double-counts or splits days. Weeks roll up to years only after you fix a rule (for example ISO weeks)."},
        ]},
        {"id": "rollup", "title": "Roll-up and drill-down", "blocks": [
            {"type": "p", "text": "<b>Roll-up</b> climbs a hierarchy (city → country) or drops a dimension altogether; "
                                  "<b>drill-down</b> goes the other way. Both only regroup the same cells, so the grand total never changes."},
            {"type": "eq", "text": "C(i″, j, k)[country, q, item] = Σ_{city ∈ country} C(i, j, k)[city, q, item]",
             "where": [["i, i′, i″", "location at city, province and country level"], ["j, k", "time (quarter) and item, unchanged"],
                       ["Σ", "the measure's aggregate, here SUM"]]},
            {"type": "steps", "title": "How the server rolls location up", "items": [
                "Join the fact table to the location dimension so each row knows its province and country.",
                "GROUP BY the new level together with the dimensions that stay (quarter, item).",
                "SUM the sales in each group: one row per (country, quarter, item).",
                "Repeat one level higher; at ALL, the location column disappears.",
            ]},
            {"type": "figure", "figure": "olap-rollup", "title": "Sales at each level of location",
             "caption": "Pick the level. Bars are the groups of that level; they always add to the same total.",
             "controls": [{"k": "level", "label": "Level", "options": [["city", "city"], ["province", "province"], ["country", "country"], ["all", "ALL"]], "value": "country"}]},
            {"type": "compare", "items": [
                {"title": "Roll-up", "points": ["fewer, bigger groups", "climbs a hierarchy or removes a dimension", "GROUP BY fewer or coarser columns"]},
                {"title": "Drill-down", "points": ["more, smaller groups", "descends a hierarchy or adds a dimension", "slides 27–29: 3350 → items → item × store → rows"]},
            ]},
            {"type": "try", "label": "Replay the slide 27–29 drill-down", "params": {"op": "drill", "ds": "fact30", "dim_f30": "item", "path": "apex"},
             "note": f"3350 → {', '.join(str(v) for v in W.SLIDE_EXPECT['f30_items']['values'].values())} → item × store → the 30 fact rows."},
            {"type": "quiz", "q": "You roll the slide-18 cube up from city to country. How many cells does the result have?",
             "options": ["16", "32", "64"], "answer": 1,
             "why": "2 countries × 4 quarters × 4 items = 32 cells. Province would still give 64, because each province has only one city in the data."},
        ]},
        {"id": "dice", "title": "Slice and dice on the enrolment table", "blocks": [
            {"type": "p", "text": "DW slides 30 and 31 use G. K. Gupta's enrolment table: students by country and degree. A <b>slice</b> "
                                  "fixes one degree; a <b>dice</b> keeps a sub-cube, a set of members on each of several dimensions, "
                                  "and keeps every dimension."},
            {"type": "figure", "figure": "olap-enrol", "title": "Which cells each operation keeps",
             "caption": "Slice = slide 30 (BIT). Dice = slide 31 (BIT and BCom for India, Australia and Malaysia).",
             "controls": [{"k": "op", "label": "Operation", "options": [["slice", "Slice BIT"], ["dice", "Dice (slide 31)"], ["none", "Whole table"]], "value": "slice"}]},
            {"type": "quiz", "q": "Which of these is a dice rather than a slice?",
             "options": ["degree = 'BIT'", "country = 'India' AND degree = 'BIT'", "degree IN ('BIT','BCom') AND country IN ('India','Malaysia')"],
             "answer": 2, "why": "A slice fixes single members (one or more dimensions collapse to a point). The third keeps two members "
                                 "on each dimension, so both dimensions survive: a sub-cube."},
            {"type": "try", "label": "Open the slide-31 dice", "params": {"op": "dice", "ds": "enrol"},
             "note": "Click row and column labels to change the dice."},
        ]},
        {"id": "pivot", "title": "Pivot and window functions", "blocks": [
            {"type": "p", "text": "<b>Pivot</b> (rotate) changes which dimension runs down and which runs across, without changing a cell. "
                                  "In SQL it turns rows into columns (slide 32): SQL Server spells it PIVOT, PostgreSQL crosstab(), and "
                                  "the portable form is one SUM(CASE WHEN month = … THEN amount END) per column."},
            {"type": "figure", "figure": "olap-pivot", "title": "The lab-supplies table after the pivot",
             "caption": "One row per item, one column per month, as on slide 32."},
            {"type": "p", "text": "<b>Window functions</b> (slides 33–34) aggregate over a frame of neighbouring rows but keep every row. "
                                  "A tumbling window cuts time into blocks; a sliding window moves one row at a time."},
            {"type": "eq", "text": "AVG(amount) OVER (PARTITION BY item ORDER BY month ROWS BETWEEN n−1 PRECEDING AND CURRENT ROW)\n"
                                   "LAG(amount) OVER (PARTITION BY item ORDER BY month)\nRANK() OVER (PARTITION BY month ORDER BY amount DESC)",
             "where": [["PARTITION BY", "restart the window for each item (or month)"], ["ROWS … PRECEDING", "the frame: this row and the n−1 before it"],
                       ["LAG", "the previous row's value in the same partition (NULL for the first)"], ["RANK", "1 for the largest; ties share a rank"]]},
            {"type": "figure", "figure": "olap-window", "title": "Moving average of CSCL sales",
             "caption": "Change the frame. Sliding frames smooth; tumbling frames give one value per block.",
             "controls": [{"k": "kind", "label": "Window", "options": [["sliding", "Sliding"], ["tumbling", "Tumbling"]], "value": "sliding"},
                          {"k": "size", "label": "Months", "min": 2, "max": 4, "step": 1, "value": 3}]},
        ]},
        {"id": "notation", "title": "Cuboids, the lattice and SQL", "blocks": [
            {"type": "p", "text": "Every operation lands on a cuboid. DW slide 22 names them C(i,j,k) with 0 for ALL; with hierarchies, "
                                  "a prime marks a coarser level (i′ = province, i″ = country). All the combinations form a lattice."},
            {"type": "figure", "figure": "olap-lattice", "title": "All 24 cuboids of the slide-18 cube",
             "caption": "Bottom: the base cuboid. Top: the grand total. Each edge rolls one dimension up one level."},
            {"type": "list", "items": [
                "Roll-up / drill-down: move along an edge; the SQL changes its GROUP BY.",
                "Slice / dice: stay on a node and read part of it; the SQL adds a WHERE.",
                "Pivot: stay on the node and reshape the output; the SQL only changes the SELECT list.",
            ]},
        ]},
        {"id": "slides", "title": "What the slides get wrong", "blocks": [
            {"type": "callout", "tone": "warn", "title": "Corrections made in the live view",
             "text": "Slide 17's '2-D cube' is the Vancouver slice of slide 18, not a cube of its own. Slide 23 queries "
                     "<code>storedID = 0</code>: the column is storeID and the materialised cube (slide 24) marks ALL with NULL, "
                     "so the query must use <code>IS NULL</code> and <code>itemID IS NOT NULL</code>. Slide 24 prints the grand-total "
                     "row with an empty store cell; it is NULL like the others. Slide 26 rolls weeks up into quarters, which is not "
                     "a strict hierarchy. Slide 30 shades one column of a semester × country × degree cube, but only the 'ALL semesters' "
                     "face is given, so the BIT slice computed here is a vector of seven countries."},
            {"type": "callout", "tone": "info", "title": "Limits of the demo",
             "text": "The cubes are a few dozen cells, so every cuboid is computed on demand. Real OLAP servers precompute and index "
                     "them (see the lattice page) because a single GROUP BY over billions of rows is too slow for interactive use."},
        ]},
    ],
    "takeaways": [
        "A data cube stores one measure for every combination of dimension members; cuboids summarise it at coarser levels.",
        "Roll-up and drill-down move between cuboids and never change the total; slice and dice select cells; pivot only reshapes.",
        f"Slide 17 is the Vancouver slice; the slide-30 BIT slice totals {_BIT['after_total']}; the slide-31 dice totals {_DICE['after_total']}.",
        "Concept hierarchies multiply the cuboids: the slide-18 cube has 4 × 3 × 2 = 24.",
        "Window functions aggregate over neighbouring rows without collapsing them: moving sums, LAG and RANK.",
    ],
    "glossary": [
        ["Measure", "The numeric value stored in each cell, here SUM(sales)."],
        ["Dimension", "An axis of the cube: location, time, item."],
        ["Concept hierarchy", "The levels of a dimension, e.g. city → province → country → ALL."],
        ["Cuboid", "The cube aggregated at one level of each dimension; C(0,0,0) is the apex, C(i,j,k) the base."],
        ["Slice", "Fix one member of a dimension; that dimension disappears."],
        ["Dice", "Keep a set of members on several dimensions; all dimensions stay."],
        ["Pivot", "Swap the axes of the displayed table."],
        ["Window function", "An aggregate computed over a frame of rows, returned on every row."],
    ],
}, order=10)


# ===================================================================== CUBE: SQL grouping operators
_G9 = W.grouping("fact9", "cube", ["store", "item", "cust"])
_R9 = W.grouping("fact9", "rollup", ["item", "store", "cust"])
_N9 = W.grouping("fact9", "cube", ["store", "item", "cust"], with_null=True)
_BUGS = W.deck_bugs()
_SCHEMA = W.schema()


@figure("cube-schema")
def _fig_schema(p):
    kind = p.choice("kind", "star", ["star", "snowflake"])
    sc = _SCHEMA[kind]
    pos = ({"Sales": (230, 90), "Store": (20, 10), "Item": (440, 10), "Customer": (440, 180)} if kind == "star" else
           {"Sales": (230, 90), "Store": (20, 10), "City": (20, 110), "County": (20, 210), "Item": (440, 10), "Category": (440, 110),
            "Customer": (440, 210)})
    nodes = [{"id": t, "x": x, "y": y, "w": 140, "h": 70 if t != "Sales" else 84, "title": t,
              "sub": ", ".join(c for c in cols if not c.endswith("ID"))[:22] or ", ".join(cols)[:22],
              "value": f"{len(cols)} cols", "hot": t == "Sales"} for t, (x, y) in pos.items() for cols in [sc["tables"][t]]]
    edges = [{"from": a, "to": b, "label": k, "width": 2} for a, b, k in sc["edges"]]
    return {"kind": "flow", "height": 290, "nodes": nodes, "edges": edges,
            "notes": [f"{'Star' if kind == 'star' else 'Snowflake'}: {len(sc['tables'])} tables, {len(sc['edges'])} foreign keys.",
                      f"GROUP BY state needs {sc['joins']['state']} join{'s' if sc['joins']['state'] > 1 else ''} from Sales; "
                      f"GROUP BY category needs {sc['joins']['category']}."]}


@figure("cube-sets")
def _fig_sets(p):
    op = p.choice("op", "cube", ["groupby", "rollup", "cube"])
    order = ["item", "store", "cust"] if op == "rollup" else ["store", "item", "cust"]
    g = W.grouping("fact9", op, order)
    return {"kind": "bars", "labels": [s["label"] for s in g["sets"]], "series": [{"name": "rows", "color": "s1", "values": [s["rows"] for s in g["sets"]]}],
            "ylabel": "rows in the result", "height": 240,
            "notes": [f"{op.upper() if op != 'groupby' else 'GROUP BY'}({', '.join(order)}) on the 9-row table: {len(g['sets'])} grouping set"
                      f"{'s' if len(g['sets']) > 1 else ''}, {g['row_count']} rows" + (f", apex {g['apex']}." if g["apex"] else "."),
                      "Slide 13 shows the 33-row CUBE and slide 24 the 19-row ROLLUP; both are reproduced exactly."]}


@figure("cube-lattice")
def _fig_cube_lattice(p):
    op = p.choice("op", "rollup", ["rollup", "cube", "gsets"])
    order = ["item", "store", "cust"]
    g = W.grouping("fact9", op, order)
    lat = g["lattice"]
    nodes = [{"id": n["id"], "x": n["x"] * 600 - 70, "y": 10 + n["y"] * 210, "w": 140, "h": 44, "title": n["label"],
              "sub": f"{n['rows']} rows", "hot": n["on"]} for n in lat["nodes"]]
    edges = [{"from": a, "to": b, "width": 1.2} for a, b in lat["edges"]]
    return {"kind": "flow", "height": 270, "nodes": nodes, "edges": edges,
            "notes": [f"{op.upper() if op != 'gsets' else 'GROUPING SETS (every pair)'} on ({', '.join(order)}) uses "
                      f"{sum(n['on'] for n in lat['nodes'])} of the 8 nodes: " + ", ".join(s["label"] for s in g["sets"]) + ".",
                      "ROLLUP walks one path from the base to the apex; CUBE takes every node."]}


@figure("cube-rotations")
def _fig_rotations(p):
    cov = [W.rotated_rollup_coverage(n) for n in range(2, 7)]
    return {"kind": "bars", "labels": [f"n = {c['n']}" for c in cov],
            "series": [{"name": "2ⁿ grouping sets (CUBE)", "color": "s1", "values": [c["all"] for c in cov]},
                       {"name": "covered by n rotated ROLLUPs", "color": "s2", "values": [c["covered"] for c in cov]}],
            "ylabel": "grouping sets", "height": 240,
            "notes": [f"For 3 dimensions the rotations cover all {cov[1]['all']} sets, which is why slide 25 works.",
                      f"For 4 they cover {cov[2]['covered']} of {cov[2]['all']}: {', '.join(cov[2]['missing'])} are never a prefix of a rotation."]}


@figure("cube-null")
def _fig_null(p):
    rows = _N9["rows"]
    naive = sum(1 for r in rows if r["vals"][2] is None)
    real_sub = sum(1 for r in rows if r["grouping"][2] == 1)
    return {"kind": "stats", "items": [
        {"label": "Rows without / with the NULL sale", "value": f"{_G9['row_count']} → {_N9['row_count']}", "note": "one extra group per cuboid that contains cust"},
        {"label": "Rows that print identically", "value": f"{len(_N9['ambiguous'])} pairs", "tone": "bad", "note": "same NULLs, different meanings"},
        {"label": "WHERE custid IS NULL finds", "value": str(naive), "note": "subtotals plus the unknown customer's groups", "tone": "warn"},
        {"label": "WHERE GROUPING(custid) = 1 finds", "value": str(real_sub), "note": "exactly the customer subtotals", "tone": "good"}],
        "notes": [f"Adding one sale of {W.FACTS['fact9'].null_row[1]} with an unknown customer makes the grand total "
                  f"{_N9['fact']['total']} and creates a second all-NULL row worth {W.FACTS['fact9'].null_row[1]}.",
                  "GROUPING(col) is 1 only where the operator aggregated col away, so it separates the two."]}


@figure("cube-bugs")
def _fig_bugs(p):
    rows = [b for b in _BUGS if isinstance(b["deck_rows"], int)]
    return {"kind": "bars", "labels": [b["query"][:18] for b in rows],
            "series": [{"name": "rows from the deck's query", "color": "s2", "values": [b["deck_rows"] for b in rows]},
                       {"name": "rows from the corrected query", "color": "s3", "values": [b["fixed_rows"] for b in rows]}],
            "ylabel": "rows returned", "height": 250,
            "notes": [f"{len(rows)} queries run but return the wrong rows; {len(_BUGS) - len(rows)} more do not run at all "
                      "(an undefined alias, a missing comma, a misspelt column).",
                      "Most errors come from forgetting that a materialised cube mixes several cuboids: a cuboid is selected by "
                      "IS NULL on the dropped columns and IS NOT NULL on the kept ones."]}


explainer("bigdata/cube", {
    "title": "Building a data cube in SQL: GROUP BY, ROLLUP, CUBE",
    "kicker": KICKER,
    "summary": f"A cube is many GROUP BYs stacked into one table. On the 9-row sales table of the slides, CUBE produces {_G9['row_count']} "
               f"rows from 8 grouping sets and ROLLUP {_R9['row_count']} rows from 4. This page builds them three ways, explains the NULL "
               "that marks 'all', and corrects the deck's SQL.",
    "sections": [
        {"id": "schema", "title": "Star and snowflake schemas", "blocks": [
            {"type": "p", "text": "A relational warehouse keeps the measures in a <b>fact table</b>, Sales(storeID, itemID, custID, price, qty) "
                                  "on SQL slide 5, and the descriptions in <b>dimension tables</b> (Store, Item, Customer). Drawn around "
                                  "the fact table they make a star."},
            {"type": "figure", "figure": "cube-schema", "title": "The Sales schema",
             "caption": "Switch to the snowflake: Store and Item are normalised into chains, as slide 4 does for Han & Kamber's schema.",
             "controls": [{"k": "kind", "label": "Schema", "options": [["star", "Star"], ["snowflake", "Snowflake"]], "value": "star"}]},
            {"type": "compare", "items": [
                {"title": "Star", "points": ["dimensions denormalised: one table each", "every attribute one join away", "repeats values (state per store)"]},
                {"title": "Snowflake", "points": ["dimensions normalised into chains", f"GROUP BY state needs {_SCHEMA['snowflake']['joins']['state']} joins",
                                                  "less redundancy, easier updates"]},
            ]},
        ]},
        {"id": "union", "title": "A cube is a union of GROUP BYs", "blocks": [
            {"type": "p", "text": "With three dimensions there are 2³ = 8 ways to group: by all three, by each pair, by each one alone and "
                                  "by nothing. SQL slides 7–12 compute them one by one and stack them with UNION, writing NULL in the "
                                  "columns a query does not group by."},
            {"type": "steps", "title": "The eight queries", "items": [
                "GROUP BY store, item, cust: the 9 base cells.",
                "GROUP BY each pair: store × item, store × cust, item × cust (the floor and the two walls).",
                "GROUP BY each dimension alone: the three axes.",
                "No GROUP BY: the grand total, 735.",
                "UNION ALL the eight results; the missing columns are NULL.",
            ]},
            {"type": "figure", "figure": "cube-sets", "title": "Rows per grouping set",
             "caption": "Pick the operator: the bars are the grouping sets it runs on the 9-row table.",
             "controls": [{"k": "op", "label": "Operator", "options": [["cube", "CUBE"], ["rollup", "ROLLUP"], ["groupby", "GROUP BY"]], "value": "cube"}]},
            {"type": "eq", "text": "CUBE(a, b, c) = GROUPING SETS ((a,b,c), (a,b), (a,c), (b,c), (a), (b), (c), ())\nrows = Σ over sets of the number of distinct groups",
             "where": [["GROUPING SETS", "the list of GROUP BYs to run in one statement"], ["()", "the empty grouping: one row, the grand total"]]},
        ]},
        {"id": "operators", "title": "ROLLUP, CUBE and GROUPING SETS", "blocks": [
            {"type": "p", "text": "Writing eight queries is tedious and scans the table eight times. SQL has operators that generate the sets: "
                                  "<b>ROLLUP</b> keeps the prefixes of its list (a hierarchy: item, item × store, item × store × cust); "
                                  "<b>CUBE</b> keeps every subset; <b>GROUPING SETS</b>, beyond the slides, takes any list you write."},
            {"type": "figure", "figure": "cube-lattice", "title": "Which nodes of the lattice each operator uses",
             "caption": "ROLLUP is one path from the base to the apex; CUBE is the whole lattice.",
             "controls": [{"k": "op", "label": "Operator", "options": [["rollup", "ROLLUP"], ["cube", "CUBE"], ["gsets", "GROUPING SETS (pairs)"]], "value": "rollup"}]},
            {"type": "cards", "items": [
                {"title": "ROLLUP(item, store, cust)", "text": f"4 sets, {_R9['row_count']} rows on slide 24. The order matters."},
                {"title": "CUBE(store, item, cust)", "text": f"8 sets, {_G9['row_count']} rows on slide 13. The order does not matter."},
                {"title": "GROUPING SETS", "text": "Only the sets you need: cheaper than CUBE when most subsets are never queried."},
            ]},
            {"type": "try", "label": "Run ROLLUP(item, store, cust)", "params": {"op": "rollup", "fact": "fact9", "order": "item,store,cust"}},
            {"type": "quiz", "q": "ROLLUP(item, store, cust) and ROLLUP(cust, store, item): do they return the same rows?",
             "options": ["Yes, ROLLUP ignores the order", "No: they share only the base and the apex sets", "No: the second has twice as many rows"],
             "answer": 1, "why": "ROLLUP keeps prefixes. The first gives (item, store), (item); the second (cust, store), (cust). Only the full "
                                 "set and () are common."},
        ]},
        {"id": "rotations", "title": "CUBE from three ROLLUPs (slide 25)", "blocks": [
            {"type": "p", "text": "MySQL long had ROLLUP but no CUBE. Slide 25 builds CUBE from three ROLLUPs whose column lists are rotations "
                                  "of each other, glued with UNION. It works because every subset of three columns is a prefix of one rotation."},
            {"type": "figure", "figure": "cube-rotations", "title": "Rotations stop covering the cube after three dimensions",
             "caption": "Grouping sets produced by n rotated ROLLUPs, against the 2ⁿ that CUBE needs."},
            {"type": "callout", "tone": "info", "title": "UNION, not UNION ALL",
             "text": f"The three ROLLUPs all contain the base set and the apex, so UNION ALL returns {_G9['constructions']['rot_union_all']} rows "
                     f"instead of {_G9['row_count']}. UNION removes the repeats, which costs a sort, and it would also merge two genuinely "
                     "different rows that happen to print the same."},
        ]},
        {"id": "null", "title": "NULL means ALL, until the data has a NULL", "blocks": [
            {"type": "p", "text": "The cube marks 'all customers' with NULL. That is ambiguous as soon as the data itself contains a NULL, "
                                  "for example a sale whose customer is unknown. The deck never shows this case."},
            {"type": "figure", "figure": "cube-null", "title": "One sale with an unknown customer",
             "caption": "Computed on the 9-row table plus one row (s2, i1, NULL, 20)."},
            {"type": "eq", "text": "GROUPING(cust) = 1  ⇔  cust was aggregated away in this row\nGROUPING_ID(store, item, cust) = 4·g_store + 2·g_item + g_cust",
             "where": [["GROUPING(col)", "0 for a real value or a real NULL, 1 for the ALL marker"],
                       ["GROUPING_ID", "the bits as one number: 0 for the base cuboid, 7 for the apex"]]},
            {"type": "callout", "tone": "warn", "title": "Do not test for subtotals with IS NULL",
             "text": "WHERE custid IS NULL returns the subtotals and also the unknown customer's own groups. Use GROUPING(custid) = 1, "
                     "or COALESCE the data's NULLs to a label such as 'unknown' before cubing."},
            {"type": "try", "label": "Add the NULL sale in the live view", "params": {"nul": True, "op": "cube", "fact": "fact9"},
             "note": "Then switch on 'Label rolled-up columns ALL' to see GROUPING() fix the display."},
            {"type": "quiz", "q": "After adding the NULL sale, how many rows print as (NULL, NULL, NULL)?",
             "options": ["1", "2", "4"], "answer": 1,
             "why": "The apex (everything, 755) and the cust-only group of the unknown customer (20). Their GROUPING bits are 111 and 110."},
        ]},
        {"id": "deck", "title": "The deck's SQL, corrected", "blocks": [
            {"type": "figure", "figure": "cube-bugs", "title": "Rows returned by the deck's queries and by the corrected ones",
             "caption": "All run against the materialised 33-row cube of the 9-row table."},
            {"type": "list", "items": [
                "Slide 18: 'item axis' tests storeid IS NULL AND custid IS NULL but not itemid IS NOT NULL, so it also returns the apex; "
                "'item, store' mixes four cuboids; 'item, store, customer' has no WHERE at all; and ORDER BY uses a column storedid that does not exist.",
                "Slide 26: the slice custid = 'cust2' on the materialised cube returns rows from four cuboids.",
                "Slides 28 and 30: the literals 'store1 ' and 'item2 ' end in a space. On PostgreSQL text that matches nothing; MySQL and SQL Server ignore trailing spaces, so the bug hides there.",
                "Slide 30: SELECT f.itemid … GROUP BY f.itemid uses an alias f that the FROM clause never defines.",
                "Slide 10: SELECT itemID,storeID,custID sum(price) is missing a comma.",
                "DW slide 23: storedID = 0 misspells the column and uses 0 for ALL, while the stored cube uses NULL.",
            ]},
            {"type": "callout", "tone": "warn", "title": "The rule behind the fixes",
             "text": "A materialised cube is several cuboids in one table. To read one cuboid, put IS NULL on every dropped column and "
                     "IS NOT NULL on every kept one (or, better, filter on GROUPING_ID)."},
        ]},
    ],
    "takeaways": [
        f"CUBE over n dimensions runs 2ⁿ GROUP BYs; on the 9-row table that is {_G9['row_count']} rows with grand total {_G9['apex']}.",
        f"ROLLUP keeps the n + 1 prefixes of its list: {_R9['row_count']} rows for ROLLUP(item, store, cust). The order matters.",
        "CUBE equals the UNION of three rotated ROLLUPs only for three dimensions.",
        "NULL marks ALL in the output; GROUPING() is the only reliable way to tell it from a NULL in the data.",
        "A star schema needs one join per dimension; a snowflake trades longer joins for less redundancy.",
    ],
    "glossary": [
        ["Fact table", "The table of measurements, one row per event (a sale)."],
        ["Dimension table", "Descriptions of one dimension: Store(city, county, state)."],
        ["Grouping set", "One GROUP BY list inside a ROLLUP, CUBE or GROUPING SETS query."],
        ["ROLLUP", "GROUP BY every prefix of the list, down to ()."],
        ["CUBE", "GROUP BY every subset of the list."],
        ["GROUPING()", "1 if the column was aggregated away in this row, 0 otherwise."],
        ["Materialised cube", "The full CUBE result stored as a table, so queries read it instead of the facts."],
    ],
}, order=11)


# ===================================================================== LATTICE
_L4 = W.lattice("hk", 4, 5, ["item", "location"], where=True)
_L4b = W.lattice("hk", 4, 5, ["item", "location"], where=False)


@figure("lat-lattice")
def _fig_lat(p):
    n = p.num("n", 4, 2, 4, integer=True)
    L = W.lattice("hk", n, 0)
    lat = L["lattice"]
    widest = max(sum(1 for x in lat["nodes"] if x["y"] == y) for y in {x["y"] for x in lat["nodes"]})
    w = min(170, 600 / widest - 8)
    H = 60 + (n + 1) * 62
    nodes = [{"id": x["id"], "x": x["x"] * 600 - w / 2, "y": 10 + x["y"] * (H - 60), "w": w, "h": 44, "title": x["label"],
              "sub": f"{x['rows']} rows", "hot": not x["dims"] or len(x["dims"]) == n} for x in lat["nodes"]]
    edges = [{"from": a, "to": b, "width": 1.2} for a, b in lat["edges"]]
    biggest = max((x for x in lat["nodes"] if 0 < len(x["dims"]) < n), key=lambda x: x["rows"], default=None)
    return {"kind": "flow", "height": H, "nodes": nodes, "edges": edges,
            "notes": [f"{n} dimensions ({', '.join(L['dims'])}): 2^{n} = {L['cuboids']} cuboids, base {L['base_rows']} rows."
                      + (" T, I, L, S = time, item, location, supplier." if n == 4 else ""),
                      (f"The largest cuboid below the base is ({biggest['label']}) with {biggest['rows']} rows; the apex has 1." if biggest else
                       "Each child has one dimension fewer than its parent.")]}


@figure("lat-hier")
def _fig_hier(p):
    ns = [2, 3, 4, 5]
    hs = [W.hierarchy_lattice("hk", n) for n in ns]
    return {"kind": "bars", "labels": [f"n = {n}" for n in ns],
            "series": [{"name": "2ⁿ (no hierarchies)", "color": "s1", "values": [h["flat"] for h in hs]},
                       {"name": "∏(Lᵢ + 1) with hierarchies", "color": "s3", "values": [h["count"] for h in hs]}],
            "ylabel": "cuboids", "height": 240,
            "notes": [f"With time (quarter, year) and location (city, province, country), 5 dimensions give {hs[-1]['count']} cuboids instead of {hs[-1]['flat']}.",
                      f"Their sizes add up to {hs[-1]['total_rows']:,} rows, computed from the data."]}


@figure("lat-layers")
def _fig_layers(p):
    n = p.num("n", 4, 2, 5, integer=True)
    L = W.lattice("hk", n, 0)
    layers = sorted({x["layer"] for x in L["layers"]})
    reads = [sum(x["parent_rows"] for x in L["layers"] if x["layer"] == k) for k in layers]
    naive = [L["fact_rows"] * sum(1 for x in L["layers"] if x["layer"] == k) for k in layers]
    return {"kind": "bars", "labels": [f"MR{k}" for k in layers],
            "series": [{"name": "rows read, smallest parent", "color": "s3", "values": reads},
                       {"name": "rows read, each from the fact file", "color": "s2", "values": naive}],
            "ylabel": "rows read", "height": 250,
            "notes": [f"{n} dimensions: {len(layers)} layers. Reading each cuboid's smallest parent costs {L['layer_reads']:,} rows in all; "
                      f"rescanning the fact file for every GROUP BY costs {L['naive_reads']:,}.",
                      "Only MR1 touches the fact file; every later layer reads the much smaller output of the layer before."]}


@figure("lat-router")
def _fig_router(p):
    where = p.choice("where", "yes", ["yes", "no"]) == "yes"
    L = _L4 if where else _L4b
    q = L["query"]
    return {"kind": "stats", "items": [
        {"label": "Query", "value": "GROUP BY item, location" + (" WHERE time = 'Q1'" if where else ""), "note": "Lattice slides 21–22"},
        {"label": "Cuboid needed", "value": "(" + ", ".join(q["needs"]) + ")", "note": "GROUP BY columns + WHERE columns"},
        {"label": "Rows read", "value": f"{q['route']['cost']}", "note": f"vs {q['base_cost']} in the base cuboid", "tone": "good"},
        {"label": "Speed-up", "value": f"{q['speedup']:.0f}×", "note": "full cube materialised, as in Kylin"}],
        "notes": [f"{'With' if where else 'Without'} the WHERE on time, the smallest cuboid that can answer is ({', '.join(q['route']['chosen'])}) with "
                  f"{q['route']['cost']} rows.",
                  "A filter column must still be in the cuboid, so a WHERE pushes the query down the lattice to a bigger node."]}


@figure("lat-hru")
def _fig_hru(p):
    ck = p.choice("ds", "hk", ["hk", "fact30"])
    n = p.num("n", 4, 3, 5, integer=True)
    L = W.lattice(ck, n, 10, mode="hru")
    c = L["hru"]["curve"]
    return {"kind": "lines", "x": list(range(len(c))), "xlabel": "views materialised besides the base (k)", "ylabel": "total query cost (rows)",
            "series": [{"name": "HRU greedy", "color": "s1", "y": c},
                       {"name": "everything materialised", "color": "s2", "y": [L["hru"]["all_cost"]] * len(c), "dash": True}],
            "ymin": 0,
            "notes": [f"{'H&K' if ck == 'hk' else 'FACT30'} with {n} dimensions: answering all {L['cuboids']} cuboids from the base alone costs {c[0]:,} rows; "
                      f"{len(c) - 1} views bring it to {c[-1]:,} (the full cube would cost {L['hru']['all_cost']:,}).",
                      f"The first view saves {c[0] - c[1]:,} rows, the last only {c[-2] - c[-1]:,}: the benefits shrink every round."]}


@figure("lat-benefit")
def _fig_benefit(p):
    L = W.lattice("hk", 4, 5, mode="hru")
    views = sorted((v for v in L["sizes"] if len(v["dims"]) < 4), key=lambda v: (v["rows"], v["dims"]))
    rounds = L["hru"]["rounds"]
    top = max(rounds[0]["benefits"].values())
    cells = []
    for v in views:
        k = ",".join(v["dims"])
        cells += [None if r["benefits"].get(k) is None else r["benefits"][k] / top for r in rounds]
    return {"kind": "grid", "cols": len(rounds), "rows": len(views), "cells": cells, "palette": "seq", "height": 300,
            "xlabel": "HRU round", "ylabel": "candidate view",
            "xticks": [[i, str(i + 1)] for i in range(len(rounds))],
            "yticks": [[i, "".join(W.LSHORT[d] for d in v["dims"]) or "apex"] for i, v in enumerate(views) if i % 2 == 0],
            "notes": [f"Round 1 picks ({', '.join(rounds[0]['pick'])}), benefit {rounds[0]['benefit']:,}; then "
                      + ", ".join(f"({', '.join(r['pick'])}) {r['benefit']:,}" for r in rounds[1:]) + ".",
                      "Once a view is chosen its row is blank; the benefits of its neighbours drop because they are now cheaper to answer."]}


explainer("bigdata/lattice", {
    "title": "The lattice of cuboids: computing and choosing what to store",
    "kicker": KICKER,
    "summary": f"With four dimensions a cube has 16 cuboids; with concept hierarchies, far more. This page computes their sizes from the "
               f"course data, builds them layer by layer as Kylin's MapReduce jobs do, routes a query to the smallest cuboid that can "
               f"answer it ({_L4['query']['route']['cost']} rows instead of {_L4['base_rows']} for the slide-21 query), and uses the HRU "
               "algorithm to choose which few cuboids are worth materialising.",
    "sections": [
        {"id": "lattice", "title": "From a cube to a lattice", "blocks": [
            {"type": "p", "text": "Each subset of the dimensions defines one cuboid: GROUP BY that subset. Ordered by 'has one dimension "
                                  "fewer', the cuboids form a lattice with the base cuboid at the bottom and the apex (the grand total) "
                                  "at the top (Lattice slides 3–7)."},
            {"type": "figure", "figure": "lat-lattice", "title": "The H&K sales lattice",
             "caption": "Each node shows the number of rows (distinct groups) of its cuboid. Dimensions: time, item, location, then supplier.",
             "controls": [{"k": "n", "label": "Dimensions", "min": 2, "max": 4, "step": 1, "value": 4}]},
            {"type": "cards", "items": [
                {"title": "Node", "text": "A cuboid: the measure aggregated over one combination of dimensions."},
                {"title": "Edge", "text": "Parent → child: the child drops one dimension and can be computed from the parent."},
                {"title": "Layer", "text": "All cuboids with the same number of dimensions: C(n, k) of them."},
            ]},
            {"type": "quiz", "q": "How many cuboids does a 4-dimensional cube have, without hierarchies?",
             "options": ["4", "15", "16", "24"], "answer": 2, "why": "One per subset of the 4 dimensions, including the empty set (the apex): 2⁴ = 16."},
        ]},
        {"id": "hierarchies", "title": "Concept hierarchies multiply the cuboids", "blocks": [
            {"type": "p", "text": "Slide 7 asks whether a cube is 'a set of cuboids for each possible subset of the given dimensions'. "
                                  "Only without hierarchies. If dimension i has Lᵢ levels, it can sit at any of them or at ALL:"},
            {"type": "eq", "text": "number of cuboids = ∏ᵢ (Lᵢ + 1)",
             "where": [["Lᵢ", "levels of dimension i, not counting ALL (location: city, province, country → 3)"],
                       ["+ 1", "the ALL level"], ["∏", "product over the dimensions; with every Lᵢ = 1 this is 2ⁿ"]]},
            {"type": "figure", "figure": "lat-hier", "title": "Cuboids with and without hierarchies",
             "caption": "time has 2 levels, location 3, the others 1. Counts and sizes are enumerated on the server."},
        ]},
        {"id": "layers", "title": "Layer by layer: MR1 to MR5", "blocks": [
            {"type": "p", "text": "Lattice slides 12–13 compute the cube in layers. The first MapReduce job (or Spark stage) reads the fact "
                                  "file and builds the base cuboid; each later job builds the next layer up from the one below."},
            {"type": "steps", "title": "One layer", "items": [
                "For each child cuboid, pick its parent with the fewest rows (any parent works; the smallest is cheapest).",
                "Map: for each parent row, emit key = the child's dimension values, value = the measure.",
                "Shuffle groups equal keys together.",
                "Reduce: SUM the values of each key; the output is the child cuboid.",
            ]},
            {"type": "figure", "figure": "lat-layers", "title": "Rows read per layer",
             "caption": "Smallest-parent cubing against running every GROUP BY on the fact file.",
             "controls": [{"k": "n", "label": "Dimensions", "min": 2, "max": 5, "step": 1, "value": 4}]},
            {"type": "callout", "tone": "tip", "title": "The key-value question on the slides",
             "text": "For the child (time, item) of (time, item, location): key = (time, item), value = sales. All rows with the same "
                     "(time, item) meet at one reducer, which adds them. The same mapper can emit for several children at once."},
        ]},
        {"id": "router", "title": "Answering a query: the Kylin example", "blocks": [
            {"type": "p", "text": "Kylin precomputes the cuboids and sends each query to the smallest one that contains every column the "
                                  "query uses, in GROUP BY or in WHERE (Lattice slides 21–22)."},
            {"type": "figure", "figure": "lat-router", "title": "GROUP BY item, location, with and without WHERE time",
             "caption": "With the filter, the query needs (time, item, location); without it, (item, location) suffices.",
             "controls": [{"k": "where", "label": "WHERE on time", "options": [["yes", "With WHERE"], ["no", "Without"]], "value": "yes"}]},
            {"type": "callout", "tone": "warn", "title": "Slide 21's SQL",
             "text": "The slide writes <code>GROUP BY item, location WHERE time = '2016-10-01'</code>. WHERE must come before GROUP BY; "
                     "as printed the query is a syntax error. The live view shows it in the right order."},
            {"type": "try", "label": "Route the slide-21 query", "params": {"ds": "hk", "n": 4, "mode": "all", "where": True},
             "note": "Click other nodes of the lattice to send different GROUP BYs."},
        ]},
        {"id": "hru", "title": "Which cuboids to store: HRU greedy selection", "blocks": [
            {"type": "p", "text": "Materialising everything is fast to query but costly to build and store. Slide 14 says the cube 'may "
                                  "not materialize in full'; it does not say which part to keep. A classic answer, beyond the slides, is "
                                  "the greedy algorithm of Harinarayan, Rajaraman and Ullman (1996)."},
            {"type": "eq", "text": "cost(w | S) = min { |v| : v ∈ S, w ⊆ v }\nbenefit(v | S) = Σ_{w ⊆ v} max(0, cost(w | S) − |v|)",
             "where": [["|v|", "rows in cuboid v (the linear cost model: answering a query costs a scan of the cuboid used)"],
                       ["S", "the materialised set; it always contains the base cuboid"], ["w ⊆ v", "every query v can answer"]]},
            {"type": "steps", "items": [
                "Start with S = {base cuboid}.",
                "For every other cuboid v, add up how many rows it would save for each query it can answer.",
                "Materialise the v with the largest benefit.",
                "Repeat k times.",
            ]},
            {"type": "figure", "figure": "lat-hru", "title": "Total query cost as views are added",
             "caption": "The curve falls steeply for the first few views and then flattens.",
             "controls": [{"k": "ds", "label": "Data", "options": [["hk", "H&K sales"], ["fact30", "FACT30"]], "value": "hk"},
                          {"k": "n", "label": "Dimensions", "min": 3, "max": 5, "step": 1, "value": 4}]},
            {"type": "figure", "figure": "lat-benefit", "title": "The benefit table, round by round",
             "caption": "Rows are the candidate views (smallest first), columns the HRU rounds; darker = more benefit."},
            {"type": "try", "label": "Let HRU choose 3 views", "params": {"mode": "hru", "k": 3, "n": 4, "ds": "hk"}},
            {"type": "quiz", "q": "Why does the benefit of the best view never increase from one round to the next?",
             "options": ["Because views get bigger", "Because each pick makes queries cheaper, so there is less left to save", "It can increase"],
             "answer": 1, "why": "Adding a view only lowers cost(w | S), and the benefit of every candidate is a sum of max(0, cost − size): "
                                 "it can only shrink. This 'diminishing returns' is why greedy is guaranteed at least 63% of the optimum."},
        ]},
        {"id": "limits", "title": "Limits and the synthetic dimensions", "blocks": [
            {"type": "callout", "tone": "warn", "title": "Supplier and channel are synthetic",
             "text": f"The slides give the H&K cube in three dimensions. For 4 and 5 dimensions each (city, quarter, item) cell is split "
                     f"among SUP1–SUP3 (the supplier dimension of DW slide 20) and between shop and online, with integer weights from "
                     f"numpy.random.default_rng({W.SEED}); every sum is unchanged. FACT30 rows get a seeded supplier and channel."},
            {"type": "compare", "items": [
                {"title": "ROLAP", "points": ["cuboids stored as relational tables (slide 16)", "bitmap indexes on dimensions", "scales with the RDBMS"]},
                {"title": "MOLAP", "points": ["cuboids stored as multidimensional arrays", "fast and compact for dense cubes", "needs care with sparse data"]},
            ]},
            {"type": "p", "text": "The linear cost model ignores indexes and assumes every query is equally likely. Real systems weight "
                                  "queries by frequency, cap the storage (HRU has a space-bounded variant) and refresh views incrementally."},
        ]},
    ],
    "takeaways": [
        "A cube over n dimensions has 2ⁿ cuboids; with concept hierarchies, ∏(Lᵢ + 1).",
        "Each cuboid can be computed from any parent; using the smallest parent, layer by layer, reads far fewer rows.",
        f"A query is answered by the smallest materialised cuboid containing its GROUP BY and WHERE columns: {_L4['query']['route']['cost']} rows for the slide-21 query.",
        "HRU materialises views greedily by benefit; benefits shrink every round, so a few views give most of the speed-up.",
    ],
    "glossary": [
        ["Cuboid", "The cube aggregated over one subset of the dimensions."],
        ["Lattice", "All cuboids ordered by 'can be computed from'."],
        ["Apex / base", "The grand total (no dimensions) / the full-detail cuboid (all dimensions)."],
        ["Materialise", "Precompute and store a cuboid so queries can read it."],
        ["Linear cost model", "Answering a query costs the number of rows in the cuboid it reads."],
        ["HRU", "Greedy view selection by benefit (Harinarayan, Rajaraman, Ullman 1996)."],
        ["ROLAP / MOLAP", "Cuboids kept in relational tables / in multidimensional arrays."],
    ],
}, order=12)
