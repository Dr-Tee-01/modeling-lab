"""Explainers for the Big Data Lab's storage and batch views: HDFS, the MapReduce engine, MR joins.

Every figure is computed from modlab.bigdata.hdfs / mapreduce, the same code the live views call.
"""
from __future__ import annotations

import functools

from ..bigdata import datasets as D
from ..bigdata import hdfs as H
from ..bigdata import mapreduce as M
from .registry import explainer, figure


def _mb(v: float) -> str:
    return f"{v / 1024:.2f} GB" if v >= 1024 else (f"{v:.0f} MB" if float(v).is_integer() else f"{v:.1f} MB")


def _big(n: float) -> str:
    for d, s in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if n >= d:
            return f"{n / d:.1f}{s}"
    return f"{n:.0f}"


@functools.lru_cache(maxsize=128)
def _sim(*args):
    return H.simulate(*args)


@functools.lru_cache(maxsize=128)
def _job(*args):
    return M.run_job(*args)


@functools.lru_cache(maxsize=128)
def _join(*args):
    return M.run_join(*args)


# ===================================================================== HDFS figures
@figure("hdfs-split")
def _fig_split(p):
    size = p.num("size_mb", 500, 1, 2048)
    block = float(p.choice("block_mb", "128", ["64", "128", "256"]))
    bl = H.split_blocks(1, size, block)
    return {
        "kind": "bars", "labels": [b["id"] for b in bl] if len(bl) <= 40 else [str(i + 1) for i in range(len(bl))],
        "series": [{"name": "Block size (MB)", "color": "s1", "values": [b["mb"] for b in bl]}],
        "ylabel": "MB", "hlines": [{"y": block, "label": f"block size {block:.0f} MB", "color": "neutral"}],
        "notes": [f"{_mb(size)} with {block:.0f} MB blocks is {len(bl)} block{'s' if len(bl) > 1 else ''}: "
                  + (f"{len(bl) - 1} full and a last one of {_mb(bl[-1]['mb'])}." if bl[-1]["mb"] < block else "all full."),
                  f"With replication 3 the file uses {3 * size:,.0f} MB ({3 * size / 1024:.2f} GiB) of disk; the last block takes only its real size, not a full {block:.0f} MB."],
    }


@figure("hdfs-blockmap")
def _fig_blockmap(p):
    r = p.num("replication", 3, 1, 4, integer=True)
    aware = p.choice("policy", "aware", ["aware", "naive"]) == "aware"
    s = _sim(3, 384, 128.0, r, 6, 2, aware, 0, False, 1.0, 100.0, 1)
    nodes = s["nodes"]["place"]
    ids = [b["id"] for b in s["blocks"]]
    cells = []
    for i, nd in enumerate(nodes):
        rack = 1 if nd["rack"] == "rack A" else 2
        cells += [rack if b in nd["blocks"] else 0 for b in ids]
    loads = [nd["mb"] for nd in nodes]
    return {
        "kind": "grid", "cols": len(ids), "rows": len(nodes), "cells": cells, "palette": "cat",
        "colors": ["neutral", "s1", "s2"], "height": 240, "xlabel": "Block", "ylabel": "DataNode",
        "xticks": [[i, b] for i, b in enumerate(ids)], "yticks": [[i, nd["id"]] for i, nd in enumerate(nodes)],
        "legend": [{"name": "Copy in rack A", "color": "s1"}, {"name": "Copy in rack B", "color": "s2"}],
        "notes": [f"{s['failure']['single_rack_blocks']} of {len(ids)} blocks have every copy in one rack"
                  + (" (with one copy that is unavoidable)." if r == 1 else "." if not aware else ": rack-aware placement always spans two racks."),
                  f"DataNodes hold between {_mb(min(loads))} and {_mb(max(loads))}."],
    }


@figure("hdfs-failure")
def _fig_failure(p):
    kill = p.num("kill", 1, 1, 6, integer=True)
    s = _sim(3, 384, 128.0, 3, 6, 2, True, kill, False, 1.0, 100.0, 1)
    nodes = []
    for i, nd in enumerate(s["nodes"]["heal"]):
        dead = nd["id"] in s["dead"]
        x = 20 + (i % 3) * 200
        y = 40 if nd["rack"] == "rack A" else 150
        nodes.append({"id": nd["id"], "x": x, "y": y, "w": 160, "h": 62, "title": nd["id"] + (" (down)" if dead else ""),
                      "sub": f"{nd['rack']} · {len(nd['blocks'])} blocks", "fill": 0 if dead else nd["mb"] / 1200,
                      "fill_color": "s2" if dead else "s1", "hot": dead})
    edges = [{"from": e["from"], "to": e["to"], "label": _mb(e["mb"]), "width": 1.5 + e["mb"] / 64, "flow": True, "color": "s3"}
             for e in s["rereplication"]["by_pair"]]
    rr = s["rereplication"]
    return {
        "kind": "flow", "height": 240, "nodes": nodes, "edges": edges,
        "texts": [{"x": 300, "y": 28, "text": "rack A", "cls": "n-sub"}, {"x": 300, "y": 232, "text": "rack B", "cls": "n-sub"}],
        "notes": [f"Killing DN{kill} leaves {s['failure']['under']} blocks under-replicated and {s['failure']['lost']} lost.",
                  f"The NameNode orders {rr['copies']} copies ({_mb(rr['mb'])}, of which {_mb(rr['cross_rack_mb'])} crosses racks) "
                  "straight between DataNodes."],
    }


@figure("hdfs-rackloss")
def _fig_rackloss(p):
    t = H.rack_loss_table(6, 2, 3, 384, 128)
    return {
        "kind": "bars", "labels": [f"r = {r}" for r in t["replication"]], "ylabel": "Blocks lost (worst rack)",
        "series": [{"name": "Rack-aware", "color": "s3", "values": t["aware"]}, {"name": "Not rack-aware", "color": "s2", "values": t["naive"]}],
        "notes": [f"With 3 copies, a whole-rack failure loses {t['naive'][2]} of 9 blocks without rack awareness and {t['aware'][2]} with it.",
                  "Rack-aware placement makes any r ≥ 2 survive a rack failure; without it you need r larger than the nodes per rack."],
    }


@figure("hdfs-readtime")
def _fig_readtime(p):
    mbps = p.num("mbps", 100, 50, 1000)
    one, hundred = H.read_time(1, mbps, 1), H.read_time(100, mbps, 1)
    xs = one["curve"]["readers"]
    return {
        "kind": "lines", "x": xs, "xlog": True, "ylog": True, "xlabel": "Disks reading in parallel", "ylabel": "Hours",
        "series": [{"name": "1 TB", "color": "s1", "y": [one["hours_one"] / k for k in xs]},
                   {"name": "100 TB", "color": "s2", "y": [hundred["hours_one"] / k for k in xs]}],
        "hlines": [{"y": 1, "label": "1 hour", "color": "neutral"}],
        "notes": [f"At {mbps:.0f} MB/s one disk reads 1 TB in {one['hours_one']:.2f} h and 100 TB in {hundred['hours_one']:.0f} h ({hundred['days_one']:.1f} days).",
                  f"1000 disks read 100 TB in {hundred['hours_one'] * 60 / 1000:.1f} minutes: that is why HDFS spreads blocks over many machines."],
    }


@figure("hdfs-slide-check")
def _fig_slide_check(p):
    copies = H.slide7_copies()
    two = sum(1 for v in copies.values() if v == 2)
    r = H.read_time(1, 100, 1)
    spark = H.split_blocks(1, 500, 128)
    return {
        "kind": "stats", "items": [
            {"label": "L1 slide 12 as written", "value": f"{r['slide_formula_value']:,.0f} h", "note": "2⁴⁰ / (100·60·60): bytes divided by MB/s", "tone": "bad"},
            {"label": "L1 slide 12 corrected", "value": f"{r['hours_one']:.2f} h", "note": "2⁴⁰ / (100·2²⁰·3600)", "tone": "good"},
            {"label": "Slide 7 copies drawn", "value": f"{two} of {len(copies)} blocks ×2", "note": "the rest are elided with “…”; the default is 3", "tone": "warn"},
            {"label": "Spark deck p4 blocks", "value": " + ".join(f"{b['mb']:.0f}" for b in spark) + " MB", "note": "“four blocks of 128 MB” is really three and a 116 MB tail"},
        ],
        "notes": [f"The slide formula is off by a factor of 2²⁰ = {2 ** 20:,}; the ‘approx. 3 hours’ answer is right.",
                  "500 MB × 3 = 1500 MB is 1.46 GiB in binary units, the ‘1.5 GB’ on Spark deck p4."],
    }


# ===================================================================== MapReduce figures
@figure("mr-flow")
def _fig_mr_flow(p):
    job = p.choice("job", "avg", ["avg", "wordcount", "index", "topn"])
    comb = p.choice("combiner", "none", ["none", "proper"])
    R = {"avg": 2, "wordcount": 3, "index": 3, "topn": 1}[job]
    r = _job(job, comb, "range", R, True, True, 3)
    st = r["stages"]
    Mn = st["M"]
    nodes, edges = [], []
    for i in range(Mn):
        out = len((st["combine"] or st["map"])[i])
        nodes.append({"id": f"m{i}", "x": 10, "y": 20 + i * 70, "w": 170, "h": 56, "title": f"Mapper {i + 1}",
                      "sub": f"{len(st['splits'][i])} in → {out} out", "fill": out / max(1, len(st["map"][i])), "fill_color": "s1"})
    for j in range(R):
        h = 200 / R
        nodes.append({"id": f"r{j}", "x": 420, "y": 20 + j * h + (h - 56) / 2, "w": 170, "h": 56, "title": f"Reducer {j + 1}",
                      "sub": f"{r['partitions'][j]} · {r['load']['reducer_records'][j]} rec", "hot": True,
                      "fill": r["load"]["reducer_records"][j] / max(1, r["shuffle"]["records"]), "fill_color": "s2"})
    for i in range(Mn):
        for j in range(R):
            t = st["traffic"][i][j]["records"]
            if t:
                edges.append({"from": f"m{i}", "to": f"r{j}", "label": str(t), "width": 1.5 + t / 2.5, "flow": True})
    sh = r["shuffle"]
    return {
        "kind": "flow", "height": 230, "nodes": nodes, "edges": edges,
        "notes": [f"{M.JOBS[job]['label']}: {sh['records']} records ({sh['bytes']} bytes) cross the network"
                  + (f", against {sh['records_no_combiner']} ({sh['bytes_no_combiner']} bytes) without combining." if comb != "none" else "."),
                  f"Reducer loads {' / '.join(map(str, r['load']['reducer_records']))}; output {r['check']['n_ok']} of {r['check']['n']} correct."],
    }


@figure("mr-avg-bug")
def _fig_avg_bug(p):
    a = M.avg_of_averages()
    return {
        "kind": "bars", "labels": [f"dno {d}" for d in a["dno"]], "ylabel": "Average salary",
        "series": [{"name": "True average", "color": "s3", "values": a["true"]},
                   {"name": "Average of per-node averages", "color": "s2", "values": a["naive"]}],
        "notes": [f"dno 3: true {a['true'][1]:,.1f}, average of averages {a['naive'][1]:,.2f}; dno 5: {a['true'][3]:,.2f} against {a['naive'][3]:,.2f}.",
                  "Departments that live on one node only (dno 1 and dno 4) come out right by luck: there is nothing to average twice."],
    }


@figure("mr-combiner")
def _fig_combiner(p):
    jobs = ["avg", "wordcount", "index", "topn"]
    R = {"avg": 2, "wordcount": 3, "index": 3, "topn": 1}
    rs = [_job(j, "proper", "range", R[j], True, True, 3) for j in jobs]
    return {
        "kind": "bars", "labels": [M.JOBS[j]["label"] for j in jobs], "ylabel": "Records shuffled",
        "series": [{"name": "No combiner", "color": "neutral", "values": [r["shuffle"]["records_no_combiner"] for r in rs]},
                   {"name": "Correct combiner", "color": "s1", "values": [r["shuffle"]["records"] for r in rs]}],
        "notes": [f"The average job drops from {rs[0]['shuffle']['records_no_combiner']} to {rs[0]['shuffle']['records']} records "
                  f"({rs[0]['shuffle']['bytes_no_combiner']} → {rs[0]['shuffle']['bytes']} bytes), word count from "
                  f"{rs[1]['shuffle']['records_no_combiner']} to {rs[1]['shuffle']['records']}.",
                  f"The inverted index saves only {rs[2]['shuffle']['saved_records']} record: combiners help when keys repeat inside one mapper."],
    }


@figure("mr-skew")
def _fig_skew(p):
    R = p.num("reducers", 2, 1, 6, integer=True)
    h, rg = _job("avg", "none", "hash", R, True, True, 3), _job("avg", "none", "range", R, True, True, 3)
    return {
        "kind": "bars", "labels": [f"R{j + 1}" for j in range(R)], "ylabel": "Records received",
        "series": [{"name": "Hash mod R", "color": "s2", "values": h["load"]["reducer_records"]},
                   {"name": "Range (sampled split points)", "color": "s1", "values": rg["load"]["reducer_records"]}],
        "notes": [f"Hash: busiest reducer gets {max(h['load']['reducer_records'])} of 30 records (skew {h['load']['skew']:.2f}×); "
                  f"range: {max(rg['load']['reducer_records'])} (skew {rg['load']['skew']:.2f}×).",
                  "Key 5 holds 22 of the 30 rows, and one key always goes to one reducer, so no partitioner can do better than 22."],
    }


@figure("mr-topn")
def _fig_topn(p):
    n = p.num("n", 3, 1, 9, integer=True)
    rows = [r for s in D.TOPN_SPLITS for r in s]
    top = {c for c, _ in M.expected_topn(n)}
    r = _job("topn", "proper", "range", 1, True, True, n)
    return {
        "kind": "bars", "labels": [c for c, _ in rows], "ylabel": "Order total",
        "series": [{"name": f"Top {n}", "color": "s1", "values": [a if c in top else 0 for c, a in rows]},
                   {"name": "Dropped by a min-heap", "color": "neutral", "values": [0 if c in top else a for c, a in rows]}],
        "notes": [f"Top {n}: " + ", ".join(f"{c} ({a:,})" for c, a in M.expected_topn(n)) + ".",
                  f"Each mapper sends at most {n} records; the single reducer receives {r['shuffle']['records']} instead of {r['shuffle']['records_no_combiner']}."],
    }


@figure("mr-secsort")
def _fig_secsort(p):
    items = []
    for g, nat, label in ((True, True, "Grouping on dno, partition on dno"), (False, True, "No grouping comparator"),
                          (True, False, "Partition on <dno, salary>")):
        r = _job("secsort", "none", "hash", 2, g, nat, 3)
        items.append({"label": label, "value": f"{sum(r['load']['reduce_calls'])} calls",
                      "note": f"{len(r['output'])} output rows · {'correct' if r['check']['all_ok'] else 'wrong'}",
                      "tone": "good" if r["check"]["all_ok"] else "bad"})
    return {"kind": "stats", "items": items,
            "notes": ["Sorting alone is not enough: the grouping comparator decides which sorted keys share one reduce call.",
                      "With 4 departments, the correct job makes exactly 4 reduce calls, each with salaries already in descending order."]}


# ===================================================================== join figures
@figure("mrj-flow")
def _fig_join_flow(p):
    algo = p.choice("algo", "standard", list(M.JOIN_ALGOS))
    r = _join(algo, "inner", 3, False, True, True, "hash")
    nodes = [{"id": "c", "x": 10, "y": 30, "w": 140, "h": 56, "title": "customers", "sub": f"{len(D.MR_CUSTOMERS)} rows", "fill": 0.3, "fill_color": "s1"},
             {"id": "o", "x": 10, "y": 150, "w": 140, "h": 56, "title": "orders", "sub": f"{len(D.MR_ORDERS)} rows", "fill": 0.8, "fill_color": "s2"}]
    edges = []
    if algo == "broadcast":
        for i in range(3):
            nodes.append({"id": f"m{i}", "x": 230, "y": 20 + i * 68, "w": 150, "h": 52, "title": f"Mapper {i + 1}", "sub": "HR = customers"})
            edges += [{"from": "c", "to": f"m{i}", "label": "copy", "width": 2, "color": "s1", "flow": True},
                      {"from": "o", "to": f"m{i}", "width": 2.5, "color": "s2"}]
        nodes.append({"id": "out", "x": 450, "y": 90, "w": 140, "h": 60, "title": "Output", "value": str(r["check"]["rows"]), "hot": True})
        edges += [{"from": f"m{i}", "to": "out", "width": 2} for i in range(3)]
    else:
        nodes.append({"id": "m", "x": 220, "y": 85, "w": 130, "h": 66, "title": "Mappers", "sub": "(cid, tag)" if algo == "improved" else "cid → tagged"})
        edges += [{"from": "c", "to": "m", "width": 2, "color": "s1"}, {"from": "o", "to": "m", "width": 3, "color": "s2"}]
        for j, n in enumerate(r["stages"]["stats"]["reducer_records"]):
            nodes.append({"id": f"r{j}", "x": 430, "y": 15 + j * 72, "w": 160, "h": 56, "title": f"Reducer {j + 1}",
                          "sub": f"{n} rec · {r['stages']['stats']['reduce_calls'][j]} calls", "hot": True, "fill": n / 12, "fill_color": "s4"})
            if n:
                edges.append({"from": "m", "to": f"r{j}", "label": str(n), "width": 1.5 + n / 2, "flow": True})
    net = r["network"]
    return {
        "kind": "flow", "height": 230, "nodes": nodes, "edges": edges,
        "notes": [f"{algo.capitalize()}: {net['shuffled_records']} records ({net['shuffled_bytes']} B) shuffled, "
                  f"{net['broadcast_bytes']} B broadcast; {r['check']['rows']} joined rows.",
                  f"Peak memory in one task: {r['memory']['peak_records']} records ({r['memory']['where']})."],
    }


@figure("mrj-jointypes")
def _fig_jointypes(p):
    extra = p.choice("extra", "on", ["on", "off"]) == "on"
    series = []
    for algo, color in (("standard", "s2"), ("improved", "s1"), ("broadcast", "s3")):
        series.append({"name": algo.capitalize(), "color": color,
                       "values": [_join(algo, j, 3, extra, True, True, "hash")["check"]["rows"] for j in M.JOIN_TYPES]})
    exp = [len(M.expected_join(j, extra)) for j in M.JOIN_TYPES]
    return {
        "kind": "bars", "labels": ["Inner", "Left (orders kept)", "Right (customers kept)", "Full outer"], "ylabel": "Rows",
        "series": series, "hlines": [],
        "notes": [f"Correct row counts: inner {exp[0]}, left {exp[1]}, right {exp[2]}, full {exp[3]}"
                  + ("." if extra else " (all equal: every key has both sides in the slide data)."),
                  "The broadcast join " + ("falls short on right and full outer: no mapper can know that a customer matched no order anywhere."
                                           if extra else "gives the same 9 rows; turn on the unmatched rows to see where it falls short.")],
    }


@figure("mrj-scale")
def _fig_scale(p):
    lo, nodes = p.num("log_orders", 8, 2, 9), p.num("nodes", 100, 1, 1000, integer=True)
    s = M.join_scale(0, lo, nodes, 1024)
    c = s["curve"]
    cross = s["crossover_customers"]
    return {
        "kind": "lines", "x": c["customers"], "xlog": True, "ylog": True, "xlabel": "Customers (rows)", "ylabel": "GB on the network",
        "series": [{"name": "Repartition (shuffle both)", "color": "s2", "y": c["repartition_gb"]},
                   {"name": "Broadcast customers", "color": "s1", "y": c["broadcast_gb"]}],
        "markers": ([{"x": cross, "label": "crossover"}] if cross and cross < c["customers"][-1] else [])
                   + ([{"x": s["max_broadcast_customers"], "label": "1 GB heap full"}] if s["max_broadcast_customers"] < c["customers"][-1] else []),
        "notes": [f"{_big(s['orders'])} orders make {s['now']['maps']:,} map tasks on {nodes} nodes, so the customers table is copied {s['now']['copies']}×.",
                  (f"Broadcasting is cheaper below about {_big(cross)} customers." if cross else "Broadcasting is cheaper at every size here.")
                  + f" A 1 GB task heap holds at most {_big(s['max_broadcast_customers'])} customers."],
    }


@figure("mrj-memory")
def _fig_memory(p):
    lo = p.num("log_orders", 6, 1, 9)
    s = M.join_scale(3, lo, 100, 1024)
    n = s["now"]
    kb = lambda b: b / 1024
    return {
        "kind": "bars", "labels": ["Repartition", "Improved", "Broadcast"], "ylabel": "KB held by one task",
        "series": [{"name": "Peak memory (KB)", "color": "s4", "values": [kb(n["standard_mem"]), kb(n["improved_mem"]), kb(n["broadcast_mem"])]}],
        "notes": [f"With {_big(s['customers'])} customers and {_big(s['orders'])} orders, a repartition reducer buffers "
                  f"{s['orders'] / s['customers']:,.0f} orders per key ({kb(n['standard_mem']):,.1f} KB); the improved join holds one {s['bytes_per_record']['customer_tagged']:.0f}-byte customer.",
                  f"The broadcast join holds all customers in every mapper ({kb(n['broadcast_mem']):,.0f} KB), however many orders there are."],
    }


@figure("mrj-grouping")
def _fig_grouping(p):
    items = []
    for g, nat, label in ((True, True, "Composite key + grouping comparator"), (False, True, "No grouping comparator (as on the slides)"),
                          (True, False, "Partition on (cid, tag)")):
        r = _join("improved", "inner", 3, False, g, nat, "hash")
        items.append({"label": label, "value": f"{r['check']['rows']} rows", "note": f"{r['reduce_calls']} reduce calls · "
                      + ("correct" if r["check"]["ok"] else f"{len(r['check']['missing'])} rows missing"), "tone": "good" if r["check"]["ok"] else "bad"})
    return {"kind": "stats", "items": items,
            "notes": ["Without the grouping comparator, (111, C) and (111, O) are different keys: the customer and its orders never share a reduce call.",
                      "Partitioning on the whole composite key can send (cid, C) and (cid, O) to different reducers."]}


# ===================================================================== HDFS explainer
_S7 = H.simulate()
_R1 = H.read_time(1, 100, 1)
_KILL = H.simulate(kill=1)
_RACK = H.rack_loss_table(6, 2, 3, 384, 128)

explainer("bigdata/hdfs", {
    "title": "HDFS: how a big file lives on many machines",
    "kicker": "Big Data Lab · IE494 L1, L3 · Spark implementation deck",
    "summary": "One disk reads about 100 MB a second, so reading a terabyte takes hours. HDFS cuts every file into large blocks, "
               "stores three copies of each block on different DataNodes, and lets a central NameNode remember where everything is. "
               "This page follows a file from its first byte to a DataNode failure and the repair that follows.",
    "sections": [
        {"id": "why", "title": "Why one disk is not enough", "blocks": [
            {"type": "p", "text": "L1 slide 12 asks how long it takes just to <i>read</i> 1 TB from a hard disk at 100 MB/s. "
                                  f"The answer is {_R1['hours_one']:.2f} hours, and 100 TB takes {H.read_time(100, 100, 1)['days_one']:.1f} days, "
                                  "before any computation. The only way out is to read many disks at once."},
            {"type": "eq", "text": "time = bytes / (MB/s × 2²⁰) / disks\n1 TB at 100 MB/s = 2⁴⁰ / (100 × 2²⁰) s = 10 486 s ≈ 2.91 h",
             "where": [["bytes", "1 TB = 2⁴⁰ bytes"], ["MB/s × 2²⁰", "bytes per second (1 MB = 2²⁰ bytes)"],
                       ["disks", "how many disks read different parts of the data at the same time"]]},
            {"type": "figure", "figure": "hdfs-readtime", "title": "Read time against parallel disks",
             "caption": "Log–log axes: every tenfold increase in disks cuts the time tenfold. Change the disk speed.",
             "controls": [{"k": "mbps", "label": "Disk throughput", "min": 50, "max": 1000, "step": 10, "value": 100, "unit": "MB/s"}]},
            {"type": "try", "label": "100 TB over 1000 disks in the live calculator", "params": {"read_tb": 100, "mbps": 100, "readers": 1000}},
        ]},
        {"id": "blocks", "title": "Blocks", "blocks": [
            {"type": "p", "text": f"HDFS stores a file as a sequence of fixed-size <b>blocks</b>, {D.HDFS_BLOCK_MB} MB by default "
                                  "(64 MB in Hadoop 1). Blocks are huge compared with the 4 KB of an ordinary file system, so the "
                                  "NameNode has few entries to track and every read is a long sequential scan."},
            {"type": "figure", "figure": "hdfs-split", "title": "One file, cut into blocks",
             "caption": "The last block holds the remainder and takes only its real size on disk.",
             "controls": [{"k": "size_mb", "label": "File size", "min": 1, "max": 2048, "step": 1, "value": 500, "unit": "MB"},
                          {"k": "block_mb", "label": "Block size", "options": [["64", "64 MB"], ["128", "128 MB"], ["256", "256 MB"]], "value": "128"}]},
            {"type": "cards", "items": [
                {"title": "NameNode", "text": "Keeps the namespace (folders, files) and the block map: which blocks make each file and which DataNodes hold them. No file data."},
                {"title": "DataNodes", "text": "Store the blocks on local disks, send a heartbeat every few seconds and a block report listing what they hold."},
                {"title": "Client", "text": "Asks the NameNode where the blocks are, then reads and writes block data directly with the DataNodes."},
            ]},
        ]},
        {"id": "replicas", "title": "Replicas and where they go", "blocks": [
            {"type": "p", "text": "Each block is copied to <b>r</b> DataNodes (r = 3 by default). Copies on the same rack share a switch "
                                  "and a power supply, so Hadoop's default policy is <b>rack-aware</b>."},
            {"type": "steps", "title": "Hadoop's default placement for r = 3", "items": [
                "Copy 1 goes on the writer's own DataNode (or a random one if the client is outside the cluster).",
                "Copy 2 goes on a DataNode in a <i>different</i> rack.",
                "Copy 3 goes on another DataNode in the <i>same</i> rack as copy 2, which saves cross-rack bandwidth during the write pipeline.",
                "Any further copies go to the least-loaded nodes, with at most (r − 1)/racks + 2 copies per rack.",
            ]},
            {"type": "figure", "figure": "hdfs-blockmap", "title": "Block map on 6 DataNodes in 2 racks",
             "caption": "Rows are DataNodes, columns are the nine blocks of slide 7's three files. Colour shows the rack of each copy.",
             "controls": [{"k": "replication", "label": "Replication", "min": 1, "max": 4, "step": 1, "value": 3},
                          {"k": "policy", "label": "Placement", "options": [["aware", "Rack-aware"], ["naive", "Next nodes in order"]], "value": "aware"}]},
            {"type": "quiz", "q": "With rack-aware placement and r = 3, how many racks hold a copy of each block?",
             "options": ["One", "Two", "Three"], "answer": 1,
             "why": "Copy 1 is on the writer's rack; copies 2 and 3 share a second rack. Two racks survive any single rack failure while keeping cross-rack writes to one."},
        ]},
        {"id": "failure", "title": "When a DataNode dies", "blocks": [
            {"type": "p", "text": "A DataNode that misses heartbeats for about ten minutes is declared dead. Every block it held is now "
                                  "<b>under-replicated</b>. The NameNode does not copy anything itself: it tells a surviving holder of each "
                                  "block to send a copy to another DataNode."},
            {"type": "figure", "figure": "hdfs-failure", "title": "Re-replication after a failure",
             "caption": "Pick the DataNode to kill. Arrows are the copies the NameNode schedules, labelled with the bytes they move.",
             "controls": [{"k": "kill", "label": "Kill DataNode", "min": 1, "max": 6, "step": 1, "value": 1}]},
            {"type": "p", "text": f"On slide 7's cluster (4 DataNodes, 2 racks), killing DN1 leaves {_KILL['failure']['under']} blocks "
                                  f"under-replicated and none lost; {_KILL['rereplication']['copies']} copies ({_mb(_KILL['rereplication']['mb'])}) restore them."},
            {"type": "figure", "figure": "hdfs-rackloss", "title": "Blocks lost when a whole rack fails",
             "caption": "Worst case over both racks, on 6 DataNodes holding slide 7's nine blocks."},
            {"type": "quiz", "q": "Replication is 3 but placement ignores racks. Can a single rack failure lose data?",
             "options": ["No, three copies are always enough", "Yes, if all three copies sit in that rack", "Only if the NameNode also fails"], "answer": 1,
             "why": f"Without rack awareness the copies can all land in one rack: here {_RACK['naive'][2]} of 9 blocks are lost when the worst rack goes down."},
            {"type": "try", "label": "Take a rack down without rack awareness", "params": {"datanodes": 6, "racks": 2, "rack_aware": False, "kill": 1, "fail_rack": True, "stage": "fail"}},
        ]},
        {"id": "costs", "title": "What replication costs", "blocks": [
            {"type": "compare", "items": [
                {"title": "More copies", "points": ["Survives more simultaneous failures", "More DataNodes can serve each block to readers", "r times the disk and r − 1 extra network copies per write"]},
                {"title": "Fewer copies", "points": ["Cheaper storage", "A failure loses data at r = 1", "Hot blocks have fewer places to be read from"]},
            ]},
            {"type": "p", "text": "A 500 MB file with replication 3 uses 1500 MB of disk (Spark deck p4). Re-replication after a failure "
                                  "moves one block-sized copy per missing replica, so large blocks mean few, long transfers."},
        ]},
        {"id": "slides", "title": "What the slides get wrong", "blocks": [
            {"type": "figure", "figure": "hdfs-slide-check", "title": "Slide numbers, checked"},
            {"type": "callout", "tone": "warn", "title": "Slide errors and limits of this model",
             "text": "L1 slide 12 computes 2⁴⁰/(100·60·60): dividing bytes by MB/s without the 2²⁰ factor gives about 3 million, not 3 hours. "
                     "The answer of about 3 hours is right; the formula is not. "
                     "L3 slide 7 says the default replication is 3, but draws most blocks only twice (the rest are hidden by “…”), and its "
                     "legend colours replicas when the colours actually follow files. "
                     "Spark deck p4 calls the blocks of a 500 MB file “four blocks of 128 MB”, but the last is 116 MB. "
                     "This model places blocks deterministically (real HDFS picks random nodes within the policy), ignores the write pipeline "
                     "and balancer, and treats a dead node as dead at once rather than after the heartbeat timeout."},
            {"type": "try", "label": "Open the slide 7 layout", "params": {"files": 3, "size_mb": 384, "block_mb": 128, "replication": 3, "datanodes": 4, "racks": 2, "stage": "place"}},
        ]},
    ],
    "takeaways": [
        "Reading 1 TB at 100 MB/s takes 2⁴⁰/(100·2²⁰) s ≈ 2.9 h; only parallel reads make big data practical.",
        "HDFS splits files into 128 MB blocks; the last block holds the remainder.",
        "Each block has r copies (default 3); rack-aware placement keeps them in two racks.",
        "When a DataNode dies, the NameNode schedules copies from survivors until every block is back to r.",
        "A block is lost only when every copy is gone: rack awareness makes that much rarer.",
    ],
    "glossary": [
        ["Block", "A fixed-size piece of a file (128 MB by default), the unit of storage and replication."],
        ["NameNode", "The master that stores the namespace and the block map, but no file data."],
        ["DataNode", "A worker machine that stores blocks and reports to the NameNode."],
        ["Replication factor", "How many copies of each block HDFS keeps (default 3)."],
        ["Heartbeat", "The periodic ‘I am alive’ message from a DataNode to the NameNode."],
        ["Under-replicated", "A block with fewer live copies than the replication factor."],
        ["Rack awareness", "Placing copies so that a single rack failure cannot remove them all."],
    ],
}, order=1)


# ===================================================================== MapReduce explainer
_AVG = M.avg_of_averages()
_WC = M.run_job("wordcount", "proper", "range", 3)

explainer("bigdata/mapreduce", {
    "title": "MapReduce: map, shuffle, reduce",
    "kicker": "Big Data Lab · IE494 L3 · MapReduce programming 1 and 2",
    "summary": "MapReduce runs a job as many small map tasks next to the data, a shuffle that brings every key to one reducer, and "
               "reduce tasks that finish the work. This page runs the course's own examples (the department average over 30 employees, "
               "WordCount, Top-N, secondary sort) through a real engine and shows what moves where.",
    "sections": [
        {"id": "model", "title": "The programming model", "blocks": [
            {"type": "p", "text": "You write two functions. <b>map</b> turns one input record into zero or more (key, value) pairs. "
                                  "<b>reduce</b> receives one key with the list of all its values and writes the result. The framework does the rest."},
            {"type": "eq", "text": "map:    (k1, v1)        → list(k2, v2)\nreduce: (k2, list(v2))  → list(k3, v3)",
             "where": [["k1, v1", "an input record, e.g. (line offset, employee row)"], ["k2, v2", "an intermediate pair, e.g. (dno, salary)"],
                       ["list(v2)", "every value emitted for k2 by any mapper"], ["k3, v3", "an output row, e.g. (dno, average salary)"]]},
            {"type": "steps", "title": "SELECT dno, AVG(salary) GROUP BY dno, step by step", "items": [
                f"Split: the {sum(len(n) for n in D.EMP30_NODES)} employee rows sit on {len(D.EMP30_NODES)} DataNodes, one map task each.",
                "Map: each row becomes (dno, salary).",
                "Partition: each pair is assigned to a reducer by its key.",
                "Shuffle and sort: each reducer pulls its pairs from every mapper, sorts them by key and groups equal keys.",
                "Reduce: one call per dno adds the salaries, counts them and divides.",
            ]},
            {"type": "figure", "figure": "mr-flow", "title": "Who sends what to whom",
             "caption": "Arrows are records sent from each mapper to each reducer. Switch the job and the combiner.",
             "controls": [{"k": "job", "label": "Job", "options": [["avg", "Average salary"], ["wordcount", "Word count"], ["index", "Inverted index"], ["topn", "Top-3"]], "value": "avg"},
                          {"k": "combiner", "label": "Combiner", "options": [["none", "None"], ["proper", "Correct combiner"]], "value": "none"}]},
        ]},
        {"id": "shuffle", "title": "Partitioning and skew", "blocks": [
            {"type": "p", "text": "The <b>partitioner</b> decides which reducer owns a key. Hadoop's default hashes the key: "
                                  "(key.hashCode() & Integer.MAX_VALUE) mod R. A range partitioner gives each reducer a contiguous range "
                                  "instead, which is how the WordCount figure (MR deck 1 p18) sends A–I, J–Q and R–Z to three reducers."},
            {"type": "figure", "figure": "mr-skew", "title": "Reducer load for the department average",
             "caption": "Change the number of reducers. The range partitioner picks split points from a sample of keys.",
             "controls": [{"k": "reducers", "label": "Reducers R", "min": 1, "max": 6, "step": 1, "value": 2}]},
            {"type": "callout", "tone": "info", "title": "A hot key cannot be split",
             "text": f"Department 5 has {_AVG['counts'][5]} of the 30 employees. Every value of one key must meet in one reduce call, so one "
                     "reducer gets at least 22 records whatever the partitioner does. This is <b>skew</b>, and it sets the job's finishing time."},
            {"type": "try", "label": "See the hash partitioner's skew", "params": {"job": "avg", "partitioner": "hash", "reducers": 2, "stage": "partition"}},
        ]},
        {"id": "combiner", "title": "Combiners", "blocks": [
            {"type": "p", "text": "A <b>combiner</b> is a mini-reducer that runs on each mapper's own output before the shuffle. "
                                  "For WordCount it sums the 1s per word locally, so mapper 1 sends (Map, 4) instead of four (Map, 1)s."},
            {"type": "figure", "figure": "mr-combiner", "title": "Records shuffled, with and without a combiner"},
            {"type": "compare", "items": [
                {"title": "Without a combiner", "points": ["Every map output pair is written to disk and sent", "Reducers receive long lists of small values", "Always correct"]},
                {"title": "With a combiner", "points": ["Each mapper pre-aggregates its own keys", "Fewer records and bytes cross the network", "Correct only if reduce is associative and commutative over the combiner's output"]},
            ]},
            {"type": "p", "text": f"With combiners the WordCount reducers receive Hadoop [2, 1] and Map [4, 3] (MR deck 1 p30); the shuffle "
                                  f"carries {_WC['shuffle']['records']} records instead of {_WC['shuffle']['records_no_combiner']}, and the counts are unchanged."},
            {"type": "try", "label": "WordCount with a combiner", "params": {"job": "wordcount", "combiner": "proper", "partitioner": "range", "reducers": 3, "stage": "shuffle"}},
        ]},
        {"id": "avg", "title": "The average-of-averages bug", "blocks": [
            {"type": "p", "text": "It is tempting to reuse the reducer as the combiner (setCombinerClass(Reducer.class)). That only works "
                                  "when reduce is <b>associative and commutative</b> (MR deck 1 p32). SUM is; AVG is not: the average of "
                                  "per-node averages weights each node equally, whatever its count."},
            {"type": "figure", "figure": "mr-avg-bug", "title": "True averages against the average of node averages"},
            {"type": "eq", "text": "map:     (dno, <salary, 1>)\ncombine: (dno, <Σ salary, Σ count>)\nreduce:  avg = Σ sums / Σ counts",
             "where": [["<sum, count>", "a pair value; adding pairs is associative, so it can be combined any number of times"],
                       ["avg", f"for dno 5: {sum(s for n in D.EMP30_NODES for d, s in n if d == 5):,} / {_AVG['counts'][5]} = {_AVG['true'][3]:,.2f}"]]},
            {"type": "quiz", "q": "Which of these reduce functions can safely be reused as its own combiner?",
             "options": ["AVG", "MAX", "MEDIAN"], "answer": 1,
             "why": "MAX of MAXes is the MAX: it is associative and commutative. AVG and MEDIAN of partial results are not the overall AVG or MEDIAN."},
            {"type": "try", "label": "Run the buggy combiner", "params": {"job": "avg", "combiner": "reducer", "partitioner": "range", "reducers": 2, "stage": "reduce"}},
        ]},
        {"id": "patterns", "title": "Two more patterns: Top-N and secondary sort", "blocks": [
            {"type": "p", "text": "<b>Top-N</b> (MR deck 2 p24–29): each mapper keeps a <b>min-heap</b> of size N: when it grows to N + 1, the "
                                  "smallest is dropped, so the N largest remain. In cleanup() it emits them with key NULL, so they all reach one "
                                  "reducer, which merges the heaps the same way."},
            {"type": "figure", "figure": "mr-topn", "title": "Top-N customers by order total",
             "caption": "Change N. The two input splits are those of MR deck 2 p24.",
             "controls": [{"k": "n", "label": "N", "min": 1, "max": 9, "step": 1, "value": 3}]},
            {"type": "p", "text": "<b>Secondary sort</b> (MR deck 1 p46–47): to get salaries in descending order inside each department, make "
                                  "the key &lt;dno, salary&gt;, sort by dno then salary descending, partition on dno only, and add a "
                                  "<b>grouping comparator</b> on dno so reduce is called once per department."},
            {"type": "figure", "figure": "mr-secsort", "title": "Secondary sort, three ways"},
            {"type": "quiz", "q": "Why does Top-N use a min-heap rather than a max-heap?",
             "options": ["It is faster to build", "The smallest of the kept N is at the root, so it is the one to evict", "Hadoop sorts keys ascending"], "answer": 1,
             "why": "When a new record arrives you must drop the smallest of the N + 1. A min-heap gives it in O(log N); MR deck 2 p25's ‘max heap’ would evict the largest."},
        ]},
        {"id": "slides", "title": "Slide errors and limits", "blocks": [
            {"type": "callout", "tone": "warn", "title": "Corrected here",
             "text": "L3 slide 16 and MR deck 1 p3 give different department numbers for the same employees than slide 17 / p4–6; the lab uses "
                     "the consistent p4–6 data. MR deck 1 p32's driver sets the combiner for a job named ‘Average Salary’ although the class sums. "
                     "MR deck 2 p25 says ‘max heap’ where p26–29 (correctly) use a min-heap, and p28 compares the amount as a string; p29 "
                     "indexes the heap as if it were sorted. Secondary sort needs the grouping comparator of p46, which several examples omit. "
                     "Limits: the engine runs in one process, so ‘bytes’ are the text size of each key⇥value line, not Hadoop's serialized, "
                     "compressed spill files, and it ignores task failures and speculative execution."},
        ]},
    ],
    "takeaways": [
        "Map runs where the data is; the shuffle brings every value of a key to one reducer.",
        f"The 30-employee job gives dno 1 = {_AVG['true'][0]:,.1f}, dno 3 = {_AVG['true'][1]:,.1f}, dno 4 = {_AVG['true'][2]:,.0f}, dno 5 ≈ {_AVG['true'][3]:,.1f}.",
        "A combiner cuts shuffle traffic but must be associative and commutative: send <sum, count>, not averages.",
        "Partitioners balance keys, not records: a hot key (22 of 30 rows) always lands on one reducer.",
        "Top-N uses a min-heap per mapper and one reducer; secondary sort needs a composite key, a natural-key partitioner and a grouping comparator.",
    ],
    "glossary": [
        ["Input split", "The part of the input one map task reads, usually one HDFS block."],
        ["Partitioner", "Function from an intermediate key to a reducer number."],
        ["Shuffle", "Moving every map output pair to the reducer that owns its key, then sorting."],
        ["Combiner", "An optional reduce run on one mapper's output before the shuffle."],
        ["Skew", "Uneven reducer load; here the busiest reducer's records over the mean."],
        ["Grouping comparator", "Decides which sorted keys go to the same reduce call."],
        ["Min-heap", "A tree whose root is its smallest element, used to keep the N largest values."],
    ],
}, order=2)


# ===================================================================== Joins explainer
_J = M.run_join()
_SC = M.join_scale(3, 8, 100, 1024)

explainer("bigdata/mr-joins", {
    "title": "Joins in MapReduce",
    "kicker": "Big Data Lab · IE494 MapReduce programming 2, p8–22",
    "summary": f"Join {len(D.MR_CUSTOMERS)} customers with {len(D.MR_ORDERS)} orders when both files are spread over a cluster. "
               "The reduce-side (repartition) join shuffles both tables by the join key; the improved version sorts so the reducer "
               "barely needs memory; the broadcast join copies the small table to every mapper and skips the shuffle. "
               "Which wins depends on the table sizes, so the last part scales them up.",
    "sections": [
        {"id": "problem", "title": "The problem", "blocks": [
            {"type": "p", "text": "SELECT * FROM customers c JOIN orders o ON c.cid = o.cid (MR deck 2 p11). A customer and its orders "
                                  "can be in different files on different machines. MapReduce has one way to bring records together: give "
                                  "them the same key, so they meet in one reduce call."},
            {"type": "cards", "items": [
                {"title": "R: customers", "text": f"{len(D.MR_CUSTOMERS)} rows (cid, country, state): one row per key. The slides call it the reference table."},
                {"title": "L: orders", "text": f"{len(D.MR_ORDERS)} rows (ordrno, cid, amount): many rows per key. The log table."},
                {"title": "Answer", "text": f"{len(M.expected_join('inner'))} joined rows (cno, country, state, ordrno, amount), as on p12."},
            ]},
        ]},
        {"id": "repartition", "title": "Repartition join", "blocks": [
            {"type": "steps", "title": "Standard repartition join (p9–15)", "items": [
                "Map: read a record from either file, tag it C (customer) or O (order), and emit (cid, tagged record).",
                "Shuffle: all records with the same cid meet at one reducer, in no particular order.",
                "Reduce: walk the list, keep the customer, buffer every order, then emit one joined row per order.",
            ]},
            {"type": "figure", "figure": "mrj-flow", "title": "Records on the move",
             "caption": "Switch the algorithm. Repartition joins shuffle all 12 records; the broadcast join shuffles none.",
             "controls": [{"k": "algo", "label": "Algorithm", "options": [["standard", "Repartition"], ["improved", "Improved"], ["broadcast", "Broadcast"]], "value": "standard"}]},
            {"type": "p", "text": "Because map tasks finish in any order, the customer record can be anywhere in the list: for key 112 the "
                                  "reducer receives (O,2), (O,12), (C,112), (O,17), exactly as on p12. So the reducer must buffer orders until "
                                  f"it has seen the customer: up to {_J['memory']['peak_records']} records here, and every order of the busiest customer at scale."},
            {"type": "try", "label": "Step through the standard join", "params": {"algo": "standard", "join": "inner", "stage": "shuffle"}},
        ]},
        {"id": "types", "title": "Inner or outer? Read the reducer", "blocks": [
            {"type": "p", "text": "The p15 reducer starts with crec_rest = \"NULL,NULL\" and, if a cid has no orders, still writes the "
                                  "customer padded with NULLs. Orders with no customer keep the NULL default and are written too. "
                                  "So it computes a <b>full outer</b> join, although the SQL on p11 asks for an inner JOIN."},
            {"type": "figure", "figure": "mrj-jointypes", "title": "Rows produced by each join type",
             "caption": "With the unmatched rows on, one customer has no orders and one order has no customer.",
             "controls": [{"k": "extra", "label": "Unmatched rows", "options": [["on", "Added"], ["off", "Slide data only"]], "value": "on"}]},
            {"type": "quiz", "q": "On the slide's data, why do the inner and full outer joins give the same 9 rows?",
             "options": ["MapReduce ignores NULLs", "Every customer has orders and every order has a customer", "The reducer sorts the rows"], "answer": 1,
             "why": "Outer joins only add rows for keys that appear on one side only. In the p12 data every cid appears in both files, so the bug is invisible."},
            {"type": "try", "label": "See the full outer join with unmatched rows", "params": {"algo": "standard", "join": "full", "extra": True, "stage": "output"}},
        ]},
        {"id": "improved", "title": "Improved repartition join", "blocks": [
            {"type": "p", "text": "Make the map key the composite (cid, tag) with tag C < O. Sorting then puts each customer before its "
                                  "orders, so the reducer keeps one record and streams the orders past it (p16–20)."},
            {"type": "eq", "text": "map key       = (cid, tag),  tag C < O\nsort by       (cid, tag)\npartition on  hash(cid) mod R\ngroup by      cid",
             "where": [["(cid, tag)", "the composite key"], ["partition on cid", "so a customer and its orders reach the same reducer (p19)"],
                       ["group by cid", "the grouping comparator: one reduce call per customer, not per (cid, tag)"]]},
            {"type": "figure", "figure": "mrj-grouping", "title": "The comparator the slides forget"},
            {"type": "quiz", "q": "The improved join sorts on (cid, tag) but has no grouping comparator. What does the inner join return?",
             "options": ["The same 9 rows", "Nothing: customer and orders arrive in different reduce calls", "9 rows in a different order"], "answer": 1,
             "why": "Without grouping on cid, (111, C) and (111, O) are separate keys, so reduce never sees a customer with its orders."},
        ]},
        {"id": "broadcast", "title": "Broadcast (map-side) join", "blocks": [
            {"type": "p", "text": "If customers are small, copy the whole table to every mapper (p21–22). map_init() builds a hash table "
                                  "HR on cid; map() looks up each order and writes the joined row. There is no shuffle and no reducer."},
            {"type": "compare", "items": [
                {"title": "Repartition join", "points": ["Any table sizes", "Shuffles both tables over the network", "Any join type, including full outer", "Reducer buffers orders (standard) or one record (improved)"]},
                {"title": "Broadcast join", "points": ["The small table must fit in memory", "Copies the small table to every node, shuffles nothing", "Inner and left outer only", "Each mapper holds the whole small table"]},
            ]},
            {"type": "figure", "figure": "mrj-memory", "title": "Peak memory in one task",
             "caption": "3,000 customers; scale the orders.",
             "controls": [{"k": "log_orders", "label": "Orders (log₁₀ of 9 × 10^y)", "min": 1, "max": 9, "step": 0.5, "value": 6}]},
        ]},
        {"id": "scale", "title": "When broadcasting wins", "blocks": [
            {"type": "p", "text": "A repartition join moves both tables once. A broadcast join moves the small table once per node that runs "
                                  "map tasks. Broadcasting wins while the copies cost less than shuffling the big table, and while the small "
                                  "table fits in a task's memory."},
            {"type": "figure", "figure": "mrj-scale", "title": "Network bytes against the size of the customers table",
             "caption": "Log–log axes. Move the number of orders and nodes; the crossing moves with them.",
             "controls": [{"k": "log_orders", "label": "Orders (log₁₀ of 9 × 10^y)", "min": 2, "max": 9, "step": 0.5, "value": 8},
                          {"k": "nodes", "label": "Nodes", "min": 1, "max": 1000, "step": 1, "value": 100}]},
            {"type": "p", "text": f"With {_big(_SC['orders'])} orders on 100 nodes, broadcasting is cheaper up to about "
                                  f"{_big(_SC['crossover_customers'])} customers, and a 1 GB heap holds {_big(_SC['max_broadcast_customers'])}."},
            {"type": "try", "label": "Scale both tables up", "params": {"log_cust": 7, "log_orders": 8.5, "nodes": 100}},
        ]},
        {"id": "slides", "title": "Slide errors and limits", "blocks": [
            {"type": "callout", "tone": "warn", "title": "Corrected here",
             "text": "p15's reducer is a full outer join, not the inner join of p11 (and its no-orders branch drops a comma: ‘GujaratNULL’). "
                     "p10's generic reducer never resets rm per key, so a key with only orders reuses a stale customer. "
                     "p14's customer mapper reads record[0] without splitting the row. p17's mapper output still shows plain cid keys, "
                     "p18's comment should say R sorts before all values of L, and p20's reducer needs a grouping comparator on the join key "
                     "that no slide mentions; it also never checks the tag. p22 uses V.join_key for v.join_key. p12 lists output in descending "
                     "order number, which nothing in the code produces. Limits: the scale-up is a cost model built from the measured record "
                     "sizes with orders spread evenly over customers; real skew makes the standard join's buffers larger still."},
        ]},
    ],
    "takeaways": [
        "Repartition join: tag records, key by the join attribute, and let the shuffle bring matches together.",
        "The slide's reducer computes a full outer join; inner, left, right and full differ only on unmatched keys.",
        "The improved join needs a composite key, a partitioner on the join key, and a grouping comparator.",
        "A broadcast join shuffles nothing but needs the small table in every mapper's memory and cannot do right or full outer.",
        "Broadcast while |R| × copies < |L| and R fits in memory; otherwise repartition.",
    ],
    "glossary": [
        ["Repartition join", "Reduce-side join: both tables are shuffled on the join key."],
        ["Tag", "A marker added to each record saying which table it came from."],
        ["Composite key", "A key made of several fields, e.g. (cid, tag)."],
        ["Grouping comparator", "Decides which sorted keys share one reduce call."],
        ["Broadcast join", "Map-side join: the small table is copied to every mapper."],
        ["Outer join", "A join that also keeps rows with no match, padded with NULLs."],
    ],
}, order=3)
