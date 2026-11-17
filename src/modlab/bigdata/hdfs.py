"""HDFS blocks, replica placement, DataNode failure and re-replication (L3 slide 7, Spark deck p4).

A small, deterministic model of what the NameNode does:

* **split**: every file is cut into fixed-size blocks; the last block holds the remainder
  (a 500 MB file with 128 MB blocks is 128 + 128 + 128 + 116 MB, Spark deck p4).
* **place**: each block gets ``replication`` copies on distinct DataNodes. With rack awareness
  the default Hadoop policy is used (1st copy on the writer's node, 2nd on a node in another
  rack, 3rd on another node in the 2nd copy's rack, further copies on the least-loaded nodes
  with at most ``(r - 1) // racks + 2`` copies per rack). Without it, the writer pipelines the
  block to the next ``r - 1`` nodes in order, which ignores racks.
* **fail**: one DataNode (or its whole rack) stops sending heartbeats. Blocks that lost a copy
  are *under-replicated*; blocks with no copy left are *lost*.
* **heal**: the NameNode schedules one copy per missing replica from a surviving holder to the
  least-loaded live node that does not hold the block (preferring a new rack), so the cluster
  returns to the target replication. The bytes moved are the re-replication traffic.

The read-time calculator does the L1 slide 12 arithmetic correctly: 1 TB = 2^40 bytes and
100 MB/s = 100 * 2^20 bytes/s, so one disk needs 2^40 / (100 * 2^20) s = 2.91 h.
"""
from __future__ import annotations

import math

from .datasets import HDFS_BLOCK_MB, HDFS_FILES, HDFS_LAYOUT, HDFS_REPLICATION

MB = 2 ** 20
TB = 2 ** 40
STAGES = ("split", "place", "fail", "heal")


def rack_of(i: int, n: int, racks: int) -> int:
    """Nodes are spread over racks in contiguous runs (DN1-DN3 in rack A, DN4-DN6 in rack B, ...)."""
    return i * racks // n


def rack_name(r: int) -> str:
    return "rack " + chr(ord("A") + r)


def split_blocks(files: int, size_mb: float, block_mb: float) -> list[dict]:
    """Cut ``files`` files of ``size_mb`` each into blocks b1, b2, ... (numbered across files, as on slide 7)."""
    out, k = [], 0
    for f in range(files):
        full, rest = divmod(size_mb, block_mb)
        sizes = [block_mb] * int(full) + ([round(rest, 6)] if rest > 1e-9 else [])
        for j, s in enumerate(sizes):
            k += 1
            out.append({"id": f"b{k}", "file": f"file{f + 1}", "index": j, "mb": float(s)})
    return out


def _least(cands: list[int], load: list[float], start: int, n: int) -> int:
    """Least-loaded candidate; ties go to the first node at or after ``start`` (round robin)."""
    return min(cands, key=lambda i: (load[i], (i - start) % n))


def place(blocks: list[dict], n: int, replication: int, racks: int, rack_aware: bool) -> list[list[int]]:
    """Replica holders (node indices) for every block."""
    r = min(replication, n)
    racks = max(1, min(racks, n))
    load = [0.0] * n
    out = []
    for j, b in enumerate(blocks):
        w = j % n                                    # the writer (client co-located with this DataNode)
        if not rack_aware:
            chosen = [(w + t) % n for t in range(r)]
        else:
            chosen = [w]
            cap = (r - 1) // racks + 2
            rk = lambda i: rack_of(i, n, racks)
            if r >= 2:
                other = [i for i in range(n) if rk(i) != rk(w)] or [i for i in range(n) if i != w]
                chosen.append(_least(other, load, w + 1, n))
            if r >= 3:
                same = [i for i in range(n) if rk(i) == rk(chosen[1]) and i not in chosen]
                if not same:
                    same = [i for i in range(n) if i not in chosen]
                chosen.append(_least(same, load, w + 2, n))
            while len(chosen) < r:
                per = {}
                for i in chosen:
                    per[rk(i)] = per.get(rk(i), 0) + 1
                cands = [i for i in range(n) if i not in chosen and per.get(rk(i), 0) < cap] or \
                        [i for i in range(n) if i not in chosen]
                chosen.append(_least(cands, load, w + len(chosen), n))
        for i in chosen:
            load[i] += b["mb"]
        out.append(chosen)
    return out


def heal(blocks, holders, alive, n, racks, target, rack_aware):
    """Re-replicate every under-replicated block. Returns (new holders, list of copy transfers)."""
    racks = max(1, min(racks, n))
    load = [0.0] * n
    for b, hs in zip(blocks, holders):
        for i in hs:
            load[i] += b["mb"]
    live_nodes = [i for i in range(n) if alive[i]]
    goal = min(target, len(live_nodes))
    moves, new = [], []
    for j, (b, hs) in enumerate(zip(blocks, holders)):
        hs = list(hs)
        if not hs:                                   # lost: nothing to copy from
            new.append(hs)
            continue
        while len(hs) < goal:
            src = hs[0]
            cands = [i for i in live_nodes if i not in hs]
            if not cands:
                break
            if rack_aware:
                used = {rack_of(i, n, racks) for i in hs}
                fresh = [i for i in cands if rack_of(i, n, racks) not in used]
                cands = fresh or cands
            dst = _least(cands, load, j, n)
            load[dst] += b["mb"]
            hs.append(dst)
            moves.append({"block": b["id"], "from": src, "to": dst, "mb": b["mb"],
                          "cross_rack": rack_of(src, n, racks) != rack_of(dst, n, racks)})
        new.append(hs)
    return new, moves


def read_time(read_tb: float, mbps: float, readers: int) -> dict:
    """L1 slide 12 done right: seconds = bytes / (MB/s * 2^20), divided over parallel readers."""
    seconds = read_tb * TB / (mbps * MB)
    par = seconds / readers
    slide_value = read_tb * TB / (mbps * 60 * 60)   # the slide's 2^40 / (100 * 60 * 60): the 2^20 is missing
    curve_n = sorted({int(round(v)) for v in [10 ** (k / 8) for k in range(0, 33)]} | {readers})
    return {
        "read_tb": read_tb, "mbps": mbps, "readers": readers,
        "bytes": read_tb * TB, "seconds_one": seconds, "hours_one": seconds / 3600, "days_one": seconds / 86400,
        "seconds": par, "hours": par / 3600, "minutes": par / 60,
        "slide_formula_value": slide_value, "slide_formula_factor": MB,
        "curve": {"readers": curve_n, "hours": [seconds / k / 3600 for k in curve_n]},
    }


def simulate(files: int = 3, size_mb: float = 384, block_mb: float = HDFS_BLOCK_MB, replication: int = HDFS_REPLICATION,
             datanodes: int = 4, racks: int = 2, rack_aware: bool = True, kill: int = 0, fail_rack: bool = False,
             read_tb: float = 1, mbps: float = 100, readers: int = 1) -> dict:
    n = datanodes
    racks = max(1, min(racks, n))
    blocks = split_blocks(files, size_mb, block_mb)
    r = min(replication, n)
    holders = place(blocks, n, r, racks, rack_aware)

    dead = set()
    if 1 <= kill <= n:
        k = kill - 1
        dead = {i for i in range(n) if rack_of(i, n, racks) == rack_of(k, n, racks)} if fail_rack else {k}
    alive = [i not in dead for i in range(n)]
    after = [[i for i in hs if alive[i]] for hs in holders]
    healed, moves = heal(blocks, after, alive, n, racks, r, rack_aware) if dead else (after, [])

    def status(hs):
        return "lost" if not hs else ("under" if len(hs) < r else "ok")

    block_rows = []
    for b, h0, h1, h2 in zip(blocks, holders, after, healed):
        block_rows.append({**b, "replicas": [f"DN{i + 1}" for i in h0], "live": [f"DN{i + 1}" for i in h1],
                           "healed": [f"DN{i + 1}" for i in h2], "racks": sorted({rack_name(rack_of(i, n, racks)) for i in h0}),
                           "status_fail": status(h1), "status_heal": status(h2) if dead else "ok",
                           "single_rack": len({rack_of(i, n, racks) for i in h0}) == 1 and racks > 1})

    def node_view(hl):
        rows = []
        for i in range(n):
            ids = [b["id"] for b, hs in zip(blocks, hl) if i in hs]
            rows.append({"id": f"DN{i + 1}", "rack": rack_name(rack_of(i, n, racks)), "alive": alive[i] if hl is not holders else True,
                         "blocks": ids, "mb": sum(b["mb"] for b, hs in zip(blocks, hl) if i in hs)})
        return rows

    # parallel read of the stored files: each block read once, from the least-busy live holder
    busy = [0.0] * n
    for b, hs in zip(blocks, healed if dead else holders):
        if hs:
            i = min(hs, key=lambda x: (busy[x], x))
            busy[i] += b["mb"]
    live_n = sum(alive)
    under = sum(1 for b in block_rows if b["status_fail"] == "under")
    lost = sum(1 for b in block_rows if b["status_fail"] == "lost")
    logical = files * size_mb
    traffic = {
        "copies": len(moves), "mb": sum(m["mb"] for m in moves),
        "cross_rack_mb": sum(m["mb"] for m in moves if m["cross_rack"]),
        "by_pair": _pairs(moves, n),
    }
    return {
        "params": {"files": files, "size_mb": size_mb, "block_mb": block_mb, "replication": replication, "datanodes": n,
                   "racks": racks, "rack_aware": rack_aware, "kill": kill, "fail_rack": fail_rack},
        "effective_replication": r, "replication_capped": replication > n,
        "blocks": block_rows, "n_blocks": len(blocks),
        "files": [{"file": f"file{f + 1}", "mb": size_mb, "blocks": [b["id"] for b in blocks if b["file"] == f"file{f + 1}"]}
                  for f in range(files)],
        "nodes": {"place": node_view(holders), "fail": node_view(after), "heal": node_view(healed)},
        "dead": [f"DN{i + 1}" for i in sorted(dead)], "live_nodes": live_n,
        "storage": {"logical_mb": logical, "raw_mb": logical * r, "overhead": r,
                    "last_block_mb": blocks[-1]["mb"] if blocks else 0,
                    "max_node_mb": max((x["mb"] for x in node_view(holders)), default=0),
                    "min_node_mb": min((x["mb"] for x in node_view(holders)), default=0)},
        "failure": {"under": under, "lost": lost, "ok": len(blocks) - under - lost,
                    "lost_mb": sum(b["mb"] for b in block_rows if b["status_fail"] == "lost"),
                    "single_rack_blocks": sum(1 for b in block_rows if b["single_rack"])},
        "rereplication": {"moves": [{**m, "from": f"DN{m['from'] + 1}", "to": f"DN{m['to'] + 1}"} for m in moves], **traffic},
        "parallel_read": {"node_mb": busy, "max_node_mb": max(busy) if busy else 0,
                          "seconds": (max(busy) if busy else 0) / mbps, "serial_seconds": logical / mbps},
        "read": read_time(read_tb, mbps, readers),
    }


def _pairs(moves, n):
    agg = {}
    for m in moves:
        k = (m["from"], m["to"])
        agg[k] = agg.get(k, 0) + m["mb"]
    return [{"from": f"DN{a + 1}", "to": f"DN{b + 1}", "mb": v} for (a, b), v in sorted(agg.items())]


def rack_loss_table(datanodes: int = 6, racks: int = 2, files: int = 3, size_mb: float = 384, block_mb: float = 128) -> dict:
    """Blocks lost when a whole rack fails, by replication factor, with and without rack awareness."""
    n = datanodes
    racks = max(1, min(racks, n))
    blocks = split_blocks(files, size_mb, block_mb)
    reps = list(range(1, min(5, n) + 1))
    out = {"replication": reps, "aware": [], "naive": []}
    for r in reps:
        for key, aware in (("aware", True), ("naive", False)):
            holder_racks = [{rack_of(i, n, racks) for i in hs} for hs in place(blocks, n, r, racks, aware)]
            # a block is lost when the failed rack is the only rack holding it
            out[key].append(max(sum(1 for hr in holder_racks if hr == {k}) for k in range(racks)))
    return out


SLIDE7 = {"block_mb": HDFS_BLOCK_MB, "replication": HDFS_REPLICATION, "layout": HDFS_LAYOUT, "files": HDFS_FILES}


def slide7_copies() -> dict:
    """How many copies of each block slide 7 actually draws (the figure elides some with '...')."""
    counts = {}
    for blocks in HDFS_LAYOUT.values():
        for b in blocks:
            counts[b] = counts.get(b, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: int(kv[0][1:])))


def blocks_needed(size_mb: float, block_mb: float) -> int:
    return max(1, math.ceil(size_mb / block_mb))
