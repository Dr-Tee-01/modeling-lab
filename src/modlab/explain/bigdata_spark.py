"""Explainers for the Big Data Lab's RPC, RDD and Spark-stage views. Every figure is computed by the
same simulations as the live views (modlab.bigdata.rpc and modlab.bigdata.spark_core)."""
from __future__ import annotations

import functools

from ..bigdata import rpc as R
from ..bigdata import spark_core as S
from ..bigdata.datasets import EMP8, SPARK_CUSTOMERS, SPARK_ORDERS
from .registry import explainer, figure

POL_COL = {"none": "s4", "retry": "s2", "idempotent": "s3"}
NAME = R.POLICY_NAMES
D = dict(calls=10, balance=2000.0, amount=100.0, latency=50.0, jitter=30.0, dup=0.05, timeout=250.0, retries=3)


def _money(v: float) -> str:
    return f"{v:,.0f}"


def _signed(v: float) -> str:
    return f"{'+' if v > 0 else '−' if v < 0 else ''}{abs(v):,.0f}"


@functools.lru_cache(maxsize=64)
def _run(policy: str, loss: float = 0.2, dup: float = 0.05, seed: int = 4, retries: int = 3) -> dict:
    p = {**D, "dup": dup, "retries": retries}
    return R.simulate(p["calls"], p["balance"], p["amount"], p["latency"], p["jitter"], loss, dup, p["timeout"],
                      retries, policy, seed)


# ===================================================================== RPC figures
@functools.lru_cache(maxsize=8)
def _scenario_seed(case: str) -> tuple[int, float, float]:
    """Find a seed whose single call shows the case on its first attempt (searched, not scripted)."""
    loss, dup = (0.0, 0.5) if case == "duplicate" else (0.0, 0.0) if case == "ok" else (0.35, 0.0)
    for s in range(2000):
        r = R.simulate(1, 2000.0, 100.0, 50.0, 30.0, loss, dup, 250.0, 3, "retry", s)
        tl = r["timeline"]
        req = [m for m in tl if m["kind"] == "request"]
        rep = [m for m in tl if m["kind"] == "response"]
        if case == "ok" and len(tl) == 2:
            return s, loss, dup
        if case == "lost_request" and req[0]["lost"] and len(req) == 2 and not req[1]["lost"] and all(not m["lost"] for m in rep):
            return s, loss, dup
        if case == "lost_reply" and not req[0]["lost"] and rep and rep[0]["lost"] and len(req) == 2 and not req[1]["lost"] \
                and len(rep) == 2 and not rep[1]["lost"]:
            return s, loss, dup
        if case == "duplicate" and len(req) == 2 and req[1]["copy"]:
            return s, loss, dup
    raise RuntimeError(case)


@figure("rpc-one-call")
def _fig_one_call(p):
    case = p.choice("case", "lost_reply", ["ok", "lost_request", "lost_reply", "duplicate"])
    policy = p.choice("policy", "retry", list(R.POLICIES))
    seed, loss, dup = _scenario_seed(case)
    r = R.simulate(1, 2000.0, 100.0, 50.0, 30.0, loss, dup, 250.0, 3, policy, seed)
    tl = r["timeline"]
    req = [m for m in tl if m["kind"] == "request"]
    rep = [m for m in tl if m["kind"] == "response"]
    pc = r["per_call"][0]
    w = lambda k: 1.5 + 2.5 * k
    nodes = [
        {"id": "C", "x": 10, "y": 70, "w": 150, "h": 90, "title": "Client", "value": f"{pc['attempts']} attempt{'s' if pc['attempts'] > 1 else ''}",
         "sub": f"result: {pc['status']}"},
        {"id": "N", "x": 225, "y": 70, "w": 150, "h": 90, "title": "Network", "value": f"{r['lost']} lost · {r['duplicates']} copied",
         "sub": f"{len(tl)} messages", "fill": r["lost"] / max(1, len(tl)), "fill_color": "s2"},
        {"id": "S", "x": 440, "y": 70, "w": 150, "h": 90, "title": "Server", "value": f"balance {_money(r['final_balance'])}",
         "sub": f"{pc['executions']} debit{'s' if pc['executions'] != 1 else ''}" + (f" · {pc['dedup']} from table" if policy == "idempotent" else ""),
         "hot": True, "fill": r["final_balance"] / 2000, "fill_color": "s3" if r["difference"] == 0 else "s2"},
    ]
    edges = [
        {"from": "C", "to": "N", "label": f"{len(req)} request{'s' if len(req) > 1 else ''}", "width": w(len(req)), "flow": True, "color": "s1"},
        {"from": "N", "to": "S", "label": f"{sum(not m['lost'] for m in req)} arrive", "width": w(sum(not m['lost'] for m in req)), "flow": True, "color": "s1"},
        {"from": "S", "to": "N", "below": f"{len(rep)} repl{'ies' if len(rep) != 1 else 'y'}", "width": w(len(rep)), "color": "s3", "curve": [0, 40]},
        {"from": "N", "to": "C", "below": f"{sum(not m['lost'] for m in rep)} arrive", "width": w(sum(not m['lost'] for m in rep)), "color": "s3", "curve": [0, 40]},
    ]
    story = {"ok": "Nothing goes wrong: one request, one debit, one reply.",
             "lost_request": "The first request is lost before it reaches the server, so retrying is safe: the withdrawal runs once.",
             "lost_reply": "The server debits, but its reply is lost. The client cannot tell this from a lost request.",
             "duplicate": "The network delivers the request twice."}[case]
    effect = (f"The server ran withdraw {pc['executions']} time{'s' if pc['executions'] != 1 else ''}"
              + (f" and answered {pc['dedup']} repeat{'s' if pc['dedup'] != 1 else ''} from its idempotency table" if pc["dedup"] else "")
              + f"; the balance is {_money(r['final_balance'])} against the expected {_money(r['expected_balance'])}.")
    return {"kind": "flow", "height": 230, "nodes": nodes, "edges": edges,
            "texts": [{"x": 300, "y": 30, "text": f"{NAME[policy]} · seed {seed}", "cls": "n-title"}],
            "notes": [story, effect]}


@figure("rpc-sweep")
def _fig_sweep(p):
    retries = p.num("retries", 3, 0, 6, integer=True)
    sw = R.sweep(D["calls"], D["balance"], D["amount"], D["latency"], D["jitter"], D["dup"], D["timeout"], retries)
    x = [round(v * 100) for v in sw["loss"]]
    series = []
    for pol in R.POLICIES:
        series.append({"name": f"{NAME[pol]}: double debits", "color": POL_COL[pol], "y": sw[pol]["double_debits"]})
        series.append({"name": f"{NAME[pol]}: never ran", "color": POL_COL[pol], "y": sw[pol]["missed"], "dash": True, "thin": True})
    i20 = x.index(20)
    return {"kind": "lines", "x": x, "xlabel": "Message loss (%)", "ylabel": "Withdrawals per run of 10 calls", "ymin": 0,
            "series": series,
            "notes": [f"At 20% loss with {retries} retr{'y' if retries == 1 else 'ies'}, retrying without a key double-debits "
                      f"{sw['retry']['double_debits'][i20]:.2f} withdrawals per run on average (over {sw['seeds']} seeds); with a key, "
                      f"{sw['idempotent']['double_debits'][i20]:.2f}.",
                      f"Without retries {sw['none']['missed'][i20]:.2f} withdrawals per run never happen; with {retries} retr{'y' if retries == 1 else 'ies'} "
                      f"and a key, {sw['idempotent']['missed'][i20]:.2f}. More retries trade missing debits for duplicate ones unless the server deduplicates."]}


@figure("rpc-attempts")
def _fig_attempts(p):
    loss = p.num("loss", 0.2, 0.0, 0.6)
    ks = list(range(1, 8))
    q = (1 - loss) ** 2
    analytic = [(1 - q) ** k for k in ks]
    sims = []
    for k in ks:
        runs = [R.simulate(10, 2000.0, 100.0, 50.0, 30.0, loss, 0.0, 250.0, k - 1, "retry", s, detail=False) for s in range(12)]
        sims.append(sum(r["unknown"] for r in runs) / (10 * len(runs)))
    debits = sum(R.simulate(10, 2000.0, 100.0, 50.0, 30.0, loss, 0.0, 250.0, 3, "retry", s, detail=False)["executions"]
                 for s in range(12)) / 120
    return {"kind": "lines", "x": ks, "xlabel": "Attempts allowed (1 + retries)", "ylabel": "Share of calls left unanswered",
            "ymin": 0, "ymax": 1,
            "series": [{"name": "Formula (1 − (1 − p)²)ᵏ", "color": "s1", "y": analytic},
                       {"name": "Simulation (120 calls)", "color": "s2", "y": sims, "dash": True}],
            "notes": [f"With {loss:.0%} loss a round trip succeeds with probability (1 − p)² = {q:.2f}; after 4 attempts "
                      f"{analytic[3]:.1%} of calls are still unanswered (simulated: {sims[3]:.1%}).",
                      f"Retrying without a key, each call debits {debits:.2f} times on average with 3 retries: every attempt whose "
                      "request arrives runs withdraw, whether or not its reply gets back."]}


@figure("rpc-timeline")
def _fig_timeline(p):
    policy = p.choice("policy", "retry", list(R.POLICIES))
    r = _run(policy)
    lines, pts = [], []
    for m in r["timeline"]:
        y0 = 1 if m["from"] == "client" else 0
        if m["lost"]:
            lines.append({"x": [m["t_send"], m["t_lost"]], "y": [y0, 0.5], "color": "bad", "name": f"call {m['call']} {m['kind']} lost"})
            pts.append({"x": m["t_lost"], "y": 0.5, "color": "bad", "name": f"Call {m['call']}, attempt {m['attempt']}: {m['kind']} lost"})
        else:
            lines.append({"x": [m["t_send"], m["t_arrive"]], "y": [y0, 1 - y0], "color": "s1" if m["kind"] == "request" else "s3",
                          "dash": m["copy"], "name": f"call {m['call']} {m['kind']}"})
            if m["kind"] == "request":
                pts.append({"x": m["t_arrive"], "y": 0, "color": {"debit": "s2", "dedup": "good"}.get(m["outcome"], "bad"),
                            "name": f"Call {m['call']}, attempt {m['attempt']}: {m['outcome']}"})
    return {"kind": "scatter", "points": pts, "lines": lines, "xlabel": "Time (ms)", "ylabel": "server (0) · client (1)", "height": 300,
            "legend": [{"name": "Request", "color": "s1"}, {"name": "Reply", "color": "s3"}, {"name": "Lost", "color": "bad"},
                       {"name": "Debit", "color": "s2"}, {"name": "Answered from table", "color": "good"}],
            "notes": [f"{NAME[policy]}: {r['messages']} messages, {r['lost']} lost, {r['retries']} retries; "
                      f"{r['executions']} debits for {r['calls']} calls.",
                      f"Final balance {_money(r['final_balance'])} against the expected {_money(r['expected_balance'])} ({_signed(r['difference'])})."]}


@figure("rpc-bytes")
def _fig_bytes(p):
    loss = p.num("loss", 0.2, 0.0, 0.6)
    labels = [NAME[pol] for pol in R.POLICIES]
    runs = {pol: R.simulate(10, 2000.0, 100.0, 50.0, 30.0, loss, 0.05, 250.0, 3, pol, 4, detail=False) for pol in R.POLICIES}
    useful = [runs[pol]["bytes_useful"] for pol in R.POLICIES]
    total = [runs[pol]["bytes"] for pol in R.POLICIES]
    m = R.marshal()
    return {"kind": "bars", "labels": labels, "ylabel": "Bytes on the wire (10 calls)",
            "series": [{"name": "One request + one reply per call", "color": "s1", "values": useful},
                       {"name": "Actually sent (retries, losses, copies)", "color": "s2", "values": total}],
            "notes": [f"One marshalled request as JSON is {m['json_bytes']} bytes; the same two arguments in binary are "
                      f"{m['binary_bytes']} bytes (int32 + float64).",
                      f"At {loss:.0%} loss the retry policies put {total[1]:,} and {total[2]:,} bytes on the wire against "
                      f"{useful[1]:,} for a clean run: {total[1] / useful[1]:.2f}× the traffic."]}


@figure("rpc-endian")
def _fig_endian(p):
    account = p.num("account", R.ACCOUNT, 0, 99999, integer=True)
    m = R.marshal(account, 100.0)
    rows = [m["int32"]["be"], m["int32"]["le"], m["float64"]["be"], m["float64"]["le"]]
    cells = []
    for r in rows:
        cells += [int(b, 16) for b in r] + [None] * (8 - len(r))
    return {"kind": "grid", "cols": 8, "rows": 4, "cells": cells, "palette": "seq", "height": 200,
            "xlabel": "Byte position on the wire", "ylabel": "",
            "xticks": [[i, str(i)] for i in range(8)],
            "yticks": [[0, "int BE"], [1, "int LE"], [2, "f64 BE"], [3, "f64 LE"]],
            "notes": [f"account = {account} is {' '.join(m['int32']['be'])} in big-endian (network) order and "
                      f"{' '.join(m['int32']['le'])} in little-endian order. Read with the wrong order it becomes {m['int32']['misread']:,}.",
                      f"amount = 100.0 as an IEEE 754 double is {' '.join(m['float64']['be'])}; read backwards it is "
                      f"{m['float64']['misread']:.3e}. Darker cells are larger byte values."]}


@figure("rpc-compare")
def _fig_compare(p):
    loss = p.num("loss", 0.2, 0.0, 0.6)
    rs = {pol: _run(pol, round(loss, 2)) for pol in R.POLICIES}
    items = [{"label": NAME[pol], "value": _money(rs[pol]["final_balance"]),
              "note": f"{rs[pol]['semantics']} · expected {_money(rs[pol]['expected_balance'])} · {rs[pol]['double_debits']} double, "
                      f"{rs[pol]['missed']} missing", "tone": "good" if rs[pol]["difference"] == 0 else "bad"} for pol in R.POLICIES]
    return {"kind": "stats", "items": items,
            "notes": [f"Same seed and same network for all three: {rs['retry']['lost']} of {rs['retry']['messages']} messages lost under retries.",
                      "Only the idempotency key keeps the balance right when replies are lost." if rs["idempotent"]["difference"] == 0
                      else "Even with a key, a call whose every attempt is lost never runs: more retries or a longer timeout are needed."]}


_R0 = {pol: _run(pol) for pol in R.POLICIES}
_M0 = R.marshal()

explainer("bigdata/rpc", {
    "title": "Remote procedure calls when the network misbehaves",
    "kicker": "Big Data Lab · IE494 L4",
    "summary": "A remote call looks like a local one, but it crosses a network that can delay, lose or duplicate messages. "
               "This page follows withdraw(account, amount) through a simulated network and shows why retrying alone "
               "double-debits accounts, and how an idempotency key fixes it.",
    "sections": [
        {"id": "call", "title": "A call that leaves the machine", "blocks": [
            {"type": "p", "text": "In a local call, caller and callee share a stack and a heap: arguments are passed by value or by "
                                  "reference, and the return value comes back through a register. In a <b>remote procedure call</b> "
                                  "the client and the server are different processes on different machines. Arguments have to be "
                                  "copied into a message, sent, decoded, and the result sent back the same way."},
            {"type": "compare", "items": [
                {"title": "Local call", "points": ["Same address space", "Pass by value or by reference", "Returns or throws, nothing in between", "Nanoseconds"]},
                {"title": "Remote call", "points": ["Separate heaps and stacks", "Only by value: pointers mean nothing on the other machine",
                                                    "May succeed, fail, or leave the caller not knowing", f"Milliseconds: about {2 * D['latency'] + R.SERVER_MS:.0f} ms per round trip here"]},
            ]},
            {"type": "cards", "items": [
                {"title": "Request–reply", "text": "The client sends a request message and waits for a reply. Nothing else tells it what happened."},
                {"title": "Timeout", "text": f"If no reply arrives within the timeout ({D['timeout']:.0f} ms here) the client must decide: give up or try again."},
                {"title": "Semantics", "text": "What the system promises about how many times the procedure runs: at most once, at least once, or exactly once in effect."},
            ]},
        ]},
        {"id": "marshal", "title": "Marshalling: agreeing on the bytes", "blocks": [
            {"type": "p", "text": "Before an argument can travel it must be turned into bytes in a format both sides understand. "
                                  "<b>Serialization</b> turns an object into a portable form such as JSON; <b>marshalling</b> also fixes "
                                  "the encoding of each basic type. Integers are the classic trap: x86 stores the low byte first "
                                  "(little-endian), the network standard is high byte first (big-endian)."},
            {"type": "figure", "figure": "rpc-endian", "title": "One integer and one double, in both byte orders",
             "caption": "Change the account number and watch which byte changes in each order.",
             "controls": [{"k": "account", "label": "account (int32)", "min": 0, "max": 99999, "step": 1, "value": R.ACCOUNT}]},
            {"type": "p", "text": f"As JSON the whole request is <code>{_M0['json']}</code>: {_M0['json_bytes']} bytes of text "
                                  f"against {_M0['binary_bytes']} bytes of binary, but any machine can read it without agreeing on byte order."},
        ]},
        {"id": "failures", "title": "Four things that can happen to one call", "blocks": [
            {"type": "p", "text": "The simulation sends one withdraw(1001, 100) through a network with random delays and losses. "
                                  "Pick what goes wrong on the first attempt and which policy the client uses. Each case is a real "
                                  "run with a seed found by searching, not a drawing."},
            {"type": "figure", "figure": "rpc-one-call", "title": "One call, four fates",
             "caption": "Arrow widths count messages. The server box shows the balance after the call.",
             "controls": [{"k": "case", "label": "What goes wrong", "options": [["ok", "Nothing"], ["lost_request", "Request lost"],
                                                                                   ["lost_reply", "Reply lost"], ["duplicate", "Request duplicated"]], "value": "lost_reply"},
                          {"k": "policy", "label": "Policy", "options": [[k, NAME[k]] for k in R.POLICIES], "value": "retry"}]},
            {"type": "callout", "tone": "info", "title": "The client cannot tell a lost request from a lost reply",
             "text": "In both cases it sees the same thing: silence until the timeout. Only the server knows whether withdraw ran."},
            {"type": "quiz", "q": "The client times out and retries with a fresh request id. When does this cause a double debit?",
             "options": ["When the request was lost", "When the reply was lost", "Never: the server checks the balance first"], "answer": 1,
             "why": "If the request was lost, withdraw never ran and the retry runs it once. If the reply was lost, withdraw already ran, "
                    "and the server cannot recognise the retry as the same call, so it runs again."},
        ]},
        {"id": "semantics", "title": "At-most-once, at-least-once, exactly-once", "blocks": [
            {"type": "steps", "title": "How the three policies behave", "items": [
                "<b>No retry</b>: send once, give up at the timeout. The procedure runs at most once, but a timed-out call may or may not have run (the client's result is <i>unknown</i>).",
                "<b>Retry on timeout</b>: keep resending until a reply arrives or the retries run out. The procedure runs at least once if any request gets through, and once per arriving request.",
                "<b>Retry with an idempotency key</b>: every attempt of one call carries the same key. The server stores the reply for each key and answers repeats from that table without running withdraw again.",
                "Retries give at-least-once <i>delivery</i>; the server's table turns it into an exactly-once <i>effect</i>.",
            ]},
            {"type": "figure", "figure": "rpc-timeline", "title": "Ten calls on the wire",
             "caption": "Requests go down from the client to the server, replies go up. Dots on the server line mark what each arriving request did.",
             "controls": [{"k": "policy", "label": "Policy", "options": [[k, NAME[k]] for k in R.POLICIES], "value": "retry"}]},
            {"type": "figure", "figure": "rpc-compare", "title": "Final balance under each policy",
             "caption": "The same seed for all three, so they see identical losses.",
             "controls": [{"k": "loss", "label": "Message loss", "min": 0, "max": 0.6, "step": 0.05, "value": 0.2}]},
            {"type": "try", "label": "Open the lab with retries and no key", "params": {"policy": "retry", "loss": 0.2, "seed": 4}},
        ]},
        {"id": "idempotency", "title": "Idempotency: making a retry harmless", "blocks": [
            {"type": "p", "text": "An operation is <b>idempotent</b> if doing it twice has the same effect as doing it once. "
                                  "\"Set the balance to 900\" is idempotent; \"subtract 100\" is not. A withdrawal cannot be "
                                  "rewritten as a set, so the server makes it idempotent by remembering which calls it has already run."},
            {"type": "eq", "text": "server(key, amount):\n  if key in table: return table[key]\n  r = withdraw(amount)\n  table[key] = r\n  return r",
             "where": [["key", "a unique id the client picks once per logical call and reuses on every retry"],
                       ["table", "the server's record of replies already sent, kept at least as long as the client may retry"],
                       ["withdraw", "the real, non-idempotent operation"]]},
            {"type": "figure", "figure": "rpc-sweep", "title": "Double debits and missing debits as the network worsens",
             "caption": "Averages over many seeds. Solid lines: double debits per run; dashed lines: withdrawals that never ran.",
             "controls": [{"k": "retries", "label": "Max retries", "min": 0, "max": 6, "step": 1, "value": 3}]},
            {"type": "quiz", "q": "With an idempotency key, can the balance still be wrong?",
             "options": ["No, never", "Yes, if every attempt of a call is lost", "Yes, whenever a reply is lost"], "answer": 1,
             "why": "The key prevents running a call twice. It cannot make a call run if none of its requests reaches the server, "
                    "so the client still needs enough retries (or to report the failure)."},
            {"type": "try", "label": "Same network, with an idempotency key", "params": {"policy": "idempotent", "loss": 0.2, "seed": 4}},
        ]},
        {"id": "cost", "title": "What retries cost", "blocks": [
            {"type": "figure", "figure": "rpc-attempts", "title": "Unanswered calls against the number of attempts",
             "caption": "The formula and the simulation agree: each extra attempt multiplies the failure share by 1 − (1 − p)².",
             "controls": [{"k": "loss", "label": "Message loss", "min": 0, "max": 0.6, "step": 0.05, "value": 0.2}]},
            {"type": "eq", "text": "P(answered in one try) = (1 − p)²\nP(unanswered after k tries) = (1 − (1 − p)²)ᵏ",
             "where": [["p", "probability that one message (request or reply) is lost"], ["k", "attempts allowed: 1 + retries"]]},
            {"type": "figure", "figure": "rpc-bytes", "title": "Bytes on the wire",
             "caption": "Retries, lost messages and network copies all cost bandwidth.",
             "controls": [{"k": "loss", "label": "Message loss", "min": 0, "max": 0.6, "step": 0.05, "value": 0.2}]},
        ]},
        {"id": "slides", "title": "Notes on the slides and on this simulation", "blocks": [
            {"type": "callout", "tone": "warn", "title": "A slide error and the model's limits",
             "text": "L4 p3's client code <code>do_withdraw()</code> calls <code>deposit(accno, amount)</code>, while the server procedure "
                     "is <code>withdraw</code>: as written it would add money. The lab calls withdraw. The network here is a seeded random "
                     "model (fixed latency plus uniform jitter, independent loss and duplication); real networks have bursts, reordering "
                     "and congestion. Calls are sequential: the client waits for each one before starting the next. Message sizes are "
                     "the JSON text in UTF-8."},
            {"type": "list", "items": [
                "Service discovery, security (TLS, tokens) and error propagation (p9–p11) are not simulated.",
                "A real idempotency table needs an expiry; it must outlive the client's last possible retry.",
            ]},
        ]},
    ],
    "takeaways": [
        "A remote call can succeed, fail, or leave the caller not knowing; a timeout cannot tell a lost request from a lost reply.",
        f"Retrying with fresh ids gives at-least-once semantics: in the default run it double-debits {_R0['retry']['double_debits']} withdrawals "
        f"and ends at {_money(_R0['retry']['final_balance'])} instead of {_money(_R0['retry']['expected_balance'])}.",
        "An idempotency key plus a server-side reply table makes retries harmless: exactly-once effect.",
        "No retry is at-most-once only if the network never duplicates; true at-most-once also needs the server's table.",
        "Marshalling fixes byte order and types: 1001 in big-endian is 00 00 03 E9, and read backwards it is a large negative number.",
    ],
    "glossary": [
        ["RPC", "Remote procedure call: calling a function in another process as if it were local."],
        ["Marshalling", "Encoding arguments and results into a standard byte format for the wire."],
        ["Endianness", "The order of bytes in a multi-byte number: big-endian (high byte first) or little-endian."],
        ["At-most-once", "The procedure runs zero or one times per call."],
        ["At-least-once", "The procedure runs one or more times per call, if any request arrives."],
        ["Idempotent", "Doing it twice has the same effect as doing it once."],
        ["Idempotency key", "A client-chosen id, reused on every retry, that lets the server recognise repeats."],
    ],
}, order=4)


# ===================================================================== RDD figures
_AVG = ("avg_reduce", "avg_group", "avg_aggregate")


@figure("rdd-shuffle")
def _fig_rdd_shuffle(p):
    labels = [f"{n} partition{'s' if n > 1 else ''}" for n in range(1, 5)]
    vals = {k: [S.rdd_lab(k, n, False, 1)["shuffled_records"] for n in range(1, 5)] for k in _AVG}
    return {"kind": "bars", "labels": labels, "ylabel": "Records written to the shuffle",
            "series": [{"name": "reduceByKey", "color": "s1", "values": vals["avg_reduce"]},
                       {"name": "groupByKey", "color": "s2", "values": vals["avg_group"]},
                       {"name": "aggregateByKey", "color": "s3", "values": vals["avg_aggregate"]}],
            "notes": [f"groupByKey always ships all {len(EMP8)} employee records. reduceByKey ships one partial (sum, count) per "
                      f"department per partition: {', '.join(map(str, vals['avg_reduce']))} records for 1 to 4 partitions.",
                      "aggregateByKey combines on the map side exactly like reduceByKey, so it shuffles the same number of records."]}


@figure("rdd-steps")
def _fig_rdd_steps(p):
    pipe = p.choice("pipe", "avg_reduce", ["avg_reduce", "avg_group", "join", "regions"])
    parts = p.num("parts", 2, 1, 4, integer=True)
    r = S.rdd_lab(pipe, parts, False, 1)
    labels = [st["var"] for st in r["steps"]]
    cols = ["s1", "s3", "s4", "s2"]
    series = [{"name": f"P{j}", "color": cols[j], "values": [st["partitions"][j]["count"] if st["partitions"] and j < st["n"] else 0
                                                              for st in r["steps"]]} for j in range(parts)]
    shuf = [st for st in r["steps"] if st["kind"] == "shuffle"]
    return {"kind": "bars", "labels": labels, "ylabel": "Records in the partition", "series": series,
            "notes": [f"{r['label']} with {parts} partition{'s' if parts > 1 else ''}: the shuffle step{'s' if len(shuf) > 1 else ''} "
                      f"({', '.join(st['var'] for st in shuf)}) {'write' if len(shuf) > 1 else 'writes'} {r['shuffled_records']} records "
                      f"({r['shuffled_bytes']} B) out of {r['shuffle_input']} entering.",
                      f"Result: {r['result']}."]}


@figure("rdd-mappartitions")
def _fig_mp(p):
    parts = p.num("parts", 3, 1, 4, integer=True)
    r = S.rdd_lab("mappartitions", parts, False, 1)
    out = r["steps"][-1]["partitions"]
    vals = [int(x) for pp in out for x in pp["records"]]
    which = [pp["p"] for pp in out for _ in pp["records"]]
    cols = ["s1", "s3", "s4", "s2"]
    series = [{"name": f"P{j}", "color": cols[j], "values": [v if w == j else 0 for v, w in zip(vals, which)]} for j in range(parts)]
    return {"kind": "bars", "labels": [str(i) for i in range(1, 10)], "xlabel": "Input element", "ylabel": "Output value", "series": series,
            "notes": [f"With {parts} partition{'s' if parts > 1 else ''} the output is {vals}.",
                      "The counter i restarts at 1 in every partition, so which elements are multiplied by 100 depends on where the partitions are cut."]}


@figure("rdd-recompute")
def _fig_recompute(p):
    pipe = p.choice("pipe", "accumulator", ["accumulator", "avg_reduce", "join"])
    xs = [1, 2, 3]
    no = [S.rdd_lab(pipe, 2, False, a) for a in xs]
    yes = [S.rdd_lab(pipe, 2, True, a) for a in xs]
    series = [{"name": "Partition computations, no persist", "color": "s2", "y": [r["partitions_computed"] for r in no]},
              {"name": "Partition computations, persist()", "color": "s3", "y": [r["partitions_computed"] for r in yes]}]
    notes = [f"Three actions without persist compute {no[-1]['partitions_computed']} partitions ({no[-1]['recomputations']} of them again); "
             f"with persist(), {yes[-1]['partitions_computed']}."]
    if pipe == "accumulator":
        series.append({"name": "Accumulator value, no persist", "color": "bad", "y": [r["counter"] for r in no], "dash": True})
        series.append({"name": "Accumulator value, persist()", "color": "good", "y": [r["counter"] for r in yes], "dash": True})
        notes.append(f"The accumulator should read 4. Without persist it reads {', '.join(str(r['counter']) for r in no)} after 1, 2, 3 actions: "
                     "map() runs again on every action and adds again.")
    else:
        notes.append("Steps before the shuffle are not repeated even without persist: Spark keeps the shuffle files and skips that stage.")
    return {"kind": "lines", "x": xs, "xlabel": "Actions called", "ylabel": "Count", "ymin": 0, "series": series, "notes": notes}


@figure("rdd-flow")
def _fig_rdd_flow(p):
    method = p.choice("method", "reduceByKey", ["reduceByKey", "groupByKey"])
    r = S.rdd_lab("avg_reduce" if method == "reduceByKey" else "avg_group", 2, False, 1)
    sh = r["steps"][3]["shuffles"][0]
    red = r["steps"][3]["partitions"]
    nodes, edges = [], []
    for i, m in enumerate(sh["map"]):
        nodes.append({"id": f"m{i}", "x": 20, "y": 30 + i * 110, "w": 170, "h": 84, "title": f"Map task P{m['p']}",
                      "value": f"{m['records_in']} in → {m['records']} out", "sub": f"{m['bytes']} B written", "fill": m["records"] / 4, "fill_color": "s1"})
    for j, pp in enumerate(red):
        nodes.append({"id": f"r{j}", "x": 410, "y": 30 + j * 110, "w": 170, "h": 84, "title": f"Reduce P{j}",
                      "value": ", ".join(x.split(",")[0].strip("(' ") for x in pp["records"]) or "empty", "sub": f"{pp['count']} key{'s' if pp['count'] != 1 else ''}",
                      "hot": True})
    for i, row in enumerate(sh["matrix"]):
        for j, n in enumerate(row):
            if n:
                edges.append({"from": f"m{i}", "to": f"r{j}", "label": f"{n} rec", "width": 1.5 + 1.5 * n, "flow": True,
                              "color": "s2" if i != j else "s3"})
    return {"kind": "flow", "height": 250, "nodes": nodes, "edges": edges,
            "notes": [f"{method} with 2 partitions writes {sh['records']} records ({sh['bytes']} B) to the shuffle; "
                      f"{sh['cross_records']} of them go to the other executor.",
                      "Keys are placed by String.hashCode mod 2: " + ", ".join(
                          f"'{k}' (hash {S.java_hash(k)}) to P{S.java_hash(k) % 2}" for k in dict.fromkeys(str(d) for _, _, d, _ in EMP8)) + "."]}


@figure("rdd-stats")
def _fig_rdd_stats(p):
    r = S.rdd_lab("avg_aggregate", 2, False, 1, True)
    agg = [x for pp in r["steps"][3]["partitions"] for x in pp["records"]]
    items = []
    for rec in agg:
        k = rec.split(",")[0].strip("('")
        body = rec[rec.index("Stats("):]
        vals = dict(kv.split("=") for kv in body[6:-2].split(", "))
        items.append({"label": f"Department {k}", "value": f"{float(vals['sum']) / int(vals['count']):,.2f}",
                      "note": f"sum {float(vals['sum']):,.0f} · count {vals['count']} · min {int(vals['min']):,} · max {int(vals['max']):,}"})
    return {"kind": "stats", "items": items,
            "notes": ["The zero value Stats(0.0, 0, Integer.MAX_VALUE, Integer.MIN_VALUE) is the identity for every field: adding it changes nothing.",
                      f"Averages match the slide: {', '.join(i['label'].split()[-1] + ' → ' + i['value'] for i in items)}."]}


_RR = S.rdd_lab("avg_reduce", 2, False, 1)
_RG = S.rdd_lab("avg_group", 2, False, 1)
_RJ = S.rdd_lab("join", 2, False, 1)
_ACC = S.rdd_lab("accumulator", 2, False, 2)
_DEPT = {}
for _e, _n, _d, _s in EMP8:
    _DEPT.setdefault(_d, []).append(_s)

explainer("bigdata/rdd", {
    "title": "RDDs: partitions, shuffles and lazy evaluation",
    "kicker": "Big Data Lab · IE494 RDD decks 1–3",
    "summary": "An RDD is a read-only collection split into partitions across machines, built by transformations and computed "
               "only when an action asks. This page runs the course's pipelines on a small RDD engine and shows what every "
               "partition holds, what crosses the network in a shuffle, and what is computed again.",
    "sections": [
        {"id": "rdd", "title": "What an RDD is", "blocks": [
            {"type": "p", "text": "A <b>resilient distributed dataset</b> is an immutable collection of records cut into "
                                  "<b>partitions</b>. Each partition lives on one executor, and each task works on one partition. "
                                  "You never change an RDD: every <b>transformation</b> (map, filter, reduceByKey, join…) returns a new "
                                  "one and remembers which RDD it came from. That chain of parents is the <b>lineage</b>."},
            {"type": "cards", "items": [
                {"title": "Partitions", "text": f"The {len(EMP8)} employees in 2 partitions: rows 1–4 in P0, rows 5–8 in P1, cut the way parallelize cuts a list."},
                {"title": "Transformations", "text": "Lazy: they describe a new RDD and return immediately."},
                {"title": "Actions", "text": "collect, count, reduce, save: they run a job and bring a result to the driver."},
            ]},
            {"type": "figure", "figure": "rdd-steps", "title": "Records per partition at each step",
             "caption": "Pick a pipeline and the number of partitions; every bar is computed by running it.",
             "controls": [{"k": "pipe", "label": "Pipeline", "options": [["avg_reduce", "reduceByKey"], ["avg_group", "groupByKey"],
                                                                          ["join", "join"], ["regions", "regions"]], "value": "avg_reduce"},
                          {"k": "parts", "label": "Partitions", "min": 1, "max": 4, "step": 1, "value": 2}]},
        ]},
        {"id": "average", "title": "Average salary per department", "blocks": [
            {"type": "p", "text": "The course's running example: average salary by department over 8 employees. Summing is easy "
                                  "to split across partitions; averaging is not, because the average of two partial averages is wrong "
                                  "unless both cover the same number of employees. The fix is to carry <b>(sum, count)</b> pairs."},
            {"type": "eq", "text": "map:          (eno, name, dno, salary)  →  (dno, (salary, 1))\nreduceByKey:  (s₁, c₁) ⊕ (s₂, c₂) = (s₁ + s₂, c₁ + c₂)\nmap:          (dno, (s, c))  →  (dno, s / c)",
             "where": [["dno", "department number, field 2 of the line"], ["salary", "field 3"], ["⊕", "the combining function; associative and commutative, so order does not matter"]]},
            {"type": "steps", "title": "The slide's program, run on 2 partitions", "items": [
                "<code>lines = sc.textFile(\"employee.csv\")</code>: 4 lines in each partition.",
                "<code>emps = lines.map(split)</code> and <code>dnosals1 = emps.map(lambda rec: (rec[2], (int(rec[3]), 1)))</code>.",
                f"<code>reduceByKey</code> first combines inside each partition: P0 holds only department 5, P1 holds 1 and 4, so "
                f"{_RR['shuffled_records']} partial pairs are shuffled instead of 8 records.",
                "After the shuffle the partials for each department meet in one partition and are added: "
                + ", ".join(f"{d} → ({sum(v)}, {len(v)})" for d, v in _DEPT.items()) + ".",
                "The last map divides: " + ", ".join(f"{d} → {sum(v) / len(v):.1f}" for d, v in _DEPT.items()) + ", as on slide p31.",
            ]},
            {"type": "figure", "figure": "rdd-flow", "title": "Map tasks to reduce partitions",
             "caption": "Edge labels count the records each map task sends to each reduce partition.",
             "controls": [{"k": "method", "label": "Method", "options": [["reduceByKey", "reduceByKey"], ["groupByKey", "groupByKey"]], "value": "reduceByKey"}]},
            {"type": "try", "label": "Step through reduceByKey on 2 partitions", "params": {"pipe": "avg_reduce", "parts": 2, "actions": 1, "step": 4}},
        ]},
        {"id": "shuffle", "title": "groupByKey, reduceByKey and aggregateByKey", "blocks": [
            {"type": "compare", "items": [
                {"title": "groupByKey", "points": ["Ships every (key, value) record", "Builds the whole list of values per key on one machine",
                                                  f"{_RG['shuffled_records']} records shuffled here"]},
                {"title": "reduceByKey", "points": ["Combines inside each partition first (like a MapReduce combiner)",
                                                   "Ships one partial per key per partition", f"{_RR['shuffled_records']} records shuffled here"]},
                {"title": "aggregateByKey", "points": ["Zero value, seqFunc inside a partition, combFunc across partitions",
                                                      "The result type can differ from the value type: (sum, count) or Stats", "Shuffles like reduceByKey"]},
            ]},
            {"type": "figure", "figure": "rdd-shuffle", "title": "Records shuffled against the number of partitions",
             "caption": "More partitions mean more partial results, but never more than groupByKey's full 8."},
            {"type": "figure", "figure": "rdd-stats", "title": "aggregateByKey with Stats(sum, count, min, max)",
             "caption": "RDD deck 2 p12: one object carries four aggregates through seqFunc and combFunc."},
            {"type": "quiz", "q": "With 3 partitions, how many records does reduceByKey shuffle for the 8 employees?",
             "options": ["3", "4", "8"], "answer": 1,
             "why": "parallelize cuts 8 rows into 2, 3 and 3: P0 holds department 5 only (1 partial), P1 holds 5, 5 and 1 (2 partials), "
                    "P2 holds three of department 4 (1 partial). 1 + 2 + 1 = 4."},
        ]},
        {"id": "join", "title": "Joining customers and orders", "blocks": [
            {"type": "p", "text": f"The second example joins {len(SPARK_CUSTOMERS)} customers with {len(SPARK_ORDERS)} orders on the customer id and "
                                  "totals the order amounts per state. join is an inner join: customer 116 (Karnataka) and 117 "
                                  "(Gujarat) have no orders and drop out, so Gujarat's total comes from customer 111 alone."},
            {"type": "steps", "items": [
                "custRecs = (custId, state); orderRecs = (custId, amount).",
                f"join hash-partitions both sides on custId and shuffles them: {len(SPARK_CUSTOMERS)} + {len(SPARK_ORDERS)} = "
                f"{len(SPARK_CUSTOMERS) + len(SPARK_ORDERS)} records.",
                "stateOrders = (state, amount), then reduceByKey adds the amounts.",
                "Result: " + _RJ["result"].strip("[]") + ".",
            ]},
            {"type": "try", "label": "Open the join, inspecting the join step", "params": {"pipe": "join", "parts": 2, "actions": 1, "step": 5}},
        ]},
        {"id": "lazy", "title": "Laziness, actions and persist", "blocks": [
            {"type": "p", "text": "Calling a transformation runs nothing. Spark only records the lineage. When an action is called, "
                                  "the scheduler walks the lineage back to data it already has and runs every step in between. "
                                  "A second action runs the job again, unless an RDD on the way was <b>persisted</b>."},
            {"type": "figure", "figure": "rdd-recompute", "title": "Work done as actions are repeated",
             "caption": "Two partitions, the result RDD persisted or not.",
             "controls": [{"k": "pipe", "label": "Pipeline", "options": [["accumulator", "accumulator"], ["avg_reduce", "reduceByKey"], ["join", "join"]], "value": "accumulator"}]},
            {"type": "callout", "tone": "tip", "title": "Accumulators count recomputation too",
             "text": f"<code>counter.add(1)</code> inside map() runs every time the partition is computed. Two actions without persist "
                     f"leave the counter at {_ACC['counter']} for 4 elements. Spark only guarantees one update per task for accumulators "
                     "used inside actions such as foreach."},
            {"type": "figure", "figure": "rdd-mappartitions", "title": "mapPartitions with a per-partition counter",
             "caption": "Spark implementation deck p46: even positions within a partition are multiplied by 100, odd ones by 10.",
             "controls": [{"k": "parts", "label": "Partitions", "min": 1, "max": 4, "step": 1, "value": 3}]},
            {"type": "quiz", "q": "You call rdd2.collect() and then rdd2.count() on an RDD that is not persisted. How often does its map() function run per element?",
             "options": ["Once", "Twice", "It depends on the number of partitions"], "answer": 1,
             "why": "Each action is a separate job that recomputes rdd2 from its lineage, so map() runs once per action."},
            {"type": "try", "label": "See the accumulator double count", "params": {"pipe": "accumulator", "parts": 2, "actions": 2, "persist": False}},
        ]},
        {"id": "faults", "title": "Fault tolerance through lineage", "blocks": [
            {"type": "p", "text": "Spark does not replicate RDD partitions. If an executor dies, its partitions are rebuilt by running "
                                  "the lineage again for just those partitions, from the input (which HDFS does replicate) or from the "
                                  "nearest persisted or shuffled data. The Stages &amp; DAG lab shows exactly which partitions recompute."},
            {"type": "callout", "tone": "warn", "title": "Slide errors corrected here",
             "text": "RDD deck 1 p23 is titled \"MAX(SALARY)\" but its reduceByKey adds the salaries: a maximum needs "
                     "<code>(a, b) -> Math.max(a, b)</code>. p27–28 index <code>rec[2]</code> and <code>rec[3]</code>, which are the "
                     "date of birth and the gender in the 7-field file shown on p28; for that file the key is <code>rec[6]</code> (dno) and the "
                     "value <code>rec[4]</code> (salary), as p34 uses. This lab uses the 4-field layout (eno, name, dno, salary) where "
                     "rec[2] and rec[3] are right. RDD-more p8 says a lost partition is rebuilt \"on another replica node\": "
                     "RDDs are not replicated, the partition is recomputed from lineage. The engine is a plain-Python simulation of "
                     "Spark's behaviour, not PySpark; record sizes are the length of the record's text."},
        ]},
    ],
    "takeaways": [
        "An RDD is immutable and partitioned; transformations build lineage, actions run jobs.",
        f"reduceByKey and aggregateByKey combine inside each partition, so they shuffle {_RR['shuffled_records']} records on 2 partitions where groupByKey shuffles all {_RG['shuffled_records']}.",
        "Averages need (sum, count) pairs: 4 → 31000.0, 5 → 33250.0, 1 → 55000.0.",
        "The inner join drops customers without orders: Gujarat 3302, Texas 2223, Maharashtra 2188, Scotland 2333, Washington 2064.",
        "Without persist every action recomputes the lineage, and accumulators inside transformations count twice.",
        "Lost partitions are recomputed from lineage, not copied from a replica.",
    ],
    "glossary": [
        ["RDD", "Resilient distributed dataset: an immutable, partitioned collection with lineage."],
        ["Partition", "One slice of an RDD, processed by one task on one executor."],
        ["Lineage", "The chain of transformations that produced an RDD, used to recompute it."],
        ["Shuffle", "Redistributing records across partitions by key, through files and the network."],
        ["Map-side combine", "Merging values with the same key inside a partition before the shuffle."],
        ["persist / cache", "Keep computed partitions (in memory by default) for later actions."],
        ["Accumulator", "A shared counter that tasks add to and only the driver reads."],
        ["Broadcast variable", "A read-only value shipped once to each executor."],
    ],
}, order=5)


# ===================================================================== DAG figures
@figure("dag-stages")
def _fig_dag_stages(p):
    copart = p.choice("copart", "no", ["no", "yes"]) == "yes"
    r = S.dag_lab("quiz", 3, copart=copart)
    colW, nodes, edges = 118, [], []
    for x in r["rdds"]:
        nodes.append({"id": str(x["id"]), "x": 8 + x["depth"] * colW, "y": 30 + x["lane"] * 100, "w": 104, "h": 62,
                      "title": x["name"].split(" (")[0], "sub": f"stage {x['stage']} · {x['n']}P",
                      "hot": x["stage"] == r["n_stages"], "fill": 0.25 * x["stage"], "fill_color": ["s1", "s3", "s4", "s2"][(x["stage"] - 1) % 4]})
        for d in x["deps"]:
            edges.append({"from": str(d["parent"]), "to": str(x["id"]), "width": 3.5 if d["kind"] == "shuffle" else 1.6,
                          "color": "s2" if d["kind"] == "shuffle" else "neutral", "label": "shuffle" if d["kind"] == "shuffle" else None})
    for e in edges:
        if e["label"] is None:
            e.pop("label")
    return {"kind": "flow", "height": 240, "nodes": nodes, "edges": edges,
            "notes": [f"{r['n_stages']} stages and {r['tasks_total']} tasks: " + "; ".join(f"stage {s['id']} = {' → '.join(s['rdds'])}" for s in r["stages"]) + ".",
                      "Using mapValues keeps groupByKey's HashPartitioner, so join reads that side narrowly and two stages merge."
                      if copart else "groupByKey and join are the boundaries: each needs its inputs shuffled."]}


@figure("dag-iter")
def _fig_dag_iter(p):
    gb = p.num("data_gb", 100, 10, 5000)
    it = S.iterative_cost(10, gb)
    return {"kind": "lines", "x": it["iterations"], "xlabel": "Iterations", "ylabel": "Running time (s)", "ymin": 0,
            "series": [{"name": "Hadoop MapReduce", "color": "s2", "y": it["mapreduce"]},
                       {"name": "Spark", "color": "s1", "y": it["spark"], "area": True}],
            "markers": [{"x": 10, "label": f"{it['speedup_10']:.1f}× at 10"}],
            "notes": [f"{gb:,.0f} GB: MapReduce takes {it['per_iter_mr']:.0f} s per iteration; Spark takes {it['first']:.0f} s for the "
                      f"first and {it['per_iter_spark']:.1f} s for each later one ({it['cached_share']:.0%} of the data fits in the cluster's "
                      f"{it['cache_gb']:,.0f} GB of cache).",
                      f"At 10 iterations: {it['mapreduce'][9]:,.0f} s against {it['spark'][9]:,.0f} s, {it['speedup_10']:.1f}× faster, not the 100× on the slide."]}


@figure("dag-speedup")
def _fig_dag_speedup(p):
    it = S.iterative_cost(30, 100)
    ratio = [m / s for m, s in zip(it["mapreduce"], it["spark"])]
    return {"kind": "lines", "x": it["iterations"], "xlabel": "Iterations", "ylabel": "MapReduce time / Spark time", "ymin": 0,
            "series": [{"name": "Whole-job speed-up", "color": "s1", "y": ratio, "area": True}],
            "hlines": [{"y": 100, "label": "slide's claim: 100× at 10 iterations", "color": "bad"},
                       {"y": it["limit"], "label": f"limit t_disk / t_mem = {it['limit']:.0f}×", "color": "neutral"}],
            "notes": [f"The whole-job ratio is {ratio[0]:.1f}× at 1 iteration, {ratio[9]:.1f}× at 10 and {ratio[29]:.1f}× at 30.",
                      f"It approaches {it['limit']:.0f}× only as the first, disk-bound iteration becomes negligible: 100× holds per iteration, not per job."]}


@figure("dag-recovery")
def _fig_dag_recovery(p):
    lose = p.num("lose", 1, 0, 2, integer=True)
    pipe = p.choice("pipe", "quiz", ["quiz", "paper", "logmine"])
    r = S.dag_lab(pipe, 3, cache=pipe != "quiz", actions=1, lose=lose)
    rec: dict = {}
    for x in r["recovery"]["recomputed"]:
        rec[x["rdd"]] = rec.get(x["rdd"], 0) + 1
    labels = [x["name"].split(" (")[0] for x in r["rdds"]]
    return {"kind": "bars", "labels": labels, "ylabel": "Partitions",
            "series": [{"name": "All partitions", "color": "neutral", "values": [x["n"] for x in r["rdds"]]},
                       {"name": "Recomputed after the failure", "color": "bad", "values": [rec.get(x["name"], 0) for x in r["rdds"]]}],
            "notes": [f"Losing the executor of partition {lose} recomputes {len(r['recovery']['recomputed'])} of "
                      f"{r['recovery']['partitions_total']} partitions in {r['recovery']['tasks']} tasks.",
                      "Narrow steps rebuild only the matching partition. A shuffle needs the map outputs of every parent partition, but "
                      "only those written on the failed executor are gone, so only those map tasks re-run."]}


@figure("dag-resize")
def _fig_dag_resize(p):
    parts = p.num("parts", 6, 2, 6, integer=True)
    targets = list(range(1, 9))
    rep = [S.dag_lab("resize", parts, target=t, how="repartition") for t in targets]
    coa = [S.dag_lab("resize", parts, target=t, how="coalesce") for t in targets]
    return {"kind": "bars", "labels": [str(t) for t in targets], "xlabel": "Target partitions", "ylabel": "Records",
            "series": [{"name": "repartition: through shuffle files", "color": "s2", "values": [r["shuffle_records"] for r in rep]},
                       {"name": "repartition: change machine", "color": "s4", "values": [r["moved_records"] for r in rep]},
                       {"name": "coalesce: change machine", "color": "s3", "values": [r["moved_records"] for r in coa]}],
            "notes": [f"From {parts} partitions, repartition always writes all 12 records to shuffle files, even to grow to 8 partitions.",
                      f"coalesce cannot grow: asked for more than {parts} it keeps {parts}. Shrinking, it merges neighbours without shuffle files; "
                      f"coalesce(2) moves {coa[1]['moved_records']} records between machines."]}


@figure("dag-hash")
def _fig_dag_hash(p):
    n = p.num("n", 4, 1, 8, integer=True)
    r = S.dag_lab("hash", 3, target=n)
    sizes = [0] * n
    for pl in r["placement"]:
        sizes[pl["partition"]] += pl["records"]
    return {"kind": "bars", "labels": [f"P{i}" for i in range(n)], "ylabel": "Records", "series": [{"name": "Records", "color": "s1", "values": sizes}],
            "notes": ["; ".join(f"{pl['key']}: hashCode {pl['hash']:,} → P{pl['partition']}" for pl in r["placement"]) + ".",
                      f"{sum(1 for v in sizes if v == 0)} of {n} partition{'s' if n > 1 else ''} hold no records: 4 distinct keys can fill at most 4 partitions, and hash collisions mod {n} can leave fewer."]}


_Q = S.dag_lab("quiz", 3)
_QC = S.dag_lab("quiz", 3, copart=True)
_P20 = S.dag_lab("paper", 3, cache=True, actions=2, reuse=False)
_LM0 = S.dag_lab("logmine", 3, cache=False, actions=3)
_LM1 = S.dag_lab("logmine", 3, cache=True, actions=3)
_IT = S.iterative_cost(10, 100)

explainer("bigdata/dag", {
    "title": "How Spark turns a program into stages and tasks",
    "kicker": "Big Data Lab · IE494 Spark implementation",
    "summary": "An action becomes a job; the DAG scheduler cuts the job's lineage into stages wherever data must be shuffled, "
               "and runs one task per partition of each stage. This page works through the deck's stage quiz, caching, "
               "co-partitioned joins, recovery of a lost partition and the iterative-job comparison with MapReduce.",
    "sections": [
        {"id": "deps", "title": "Narrow and wide dependencies", "blocks": [
            {"type": "p", "text": "Every RDD knows its parents and how its partitions depend on theirs. A dependency is <b>narrow</b> "
                                  "when each parent partition feeds at most one child partition (map, filter, union, coalesce, a join "
                                  "of co-partitioned inputs). It is <b>wide</b> (a shuffle) when a child partition needs data from many "
                                  "parent partitions (groupByKey, reduceByKey, a join of inputs partitioned differently, repartition)."},
            {"type": "compare", "items": [
                {"title": "Narrow", "points": ["Pipelined: one task runs the whole chain on its partition", "No data leaves the machine",
                                              "A lost partition needs only the matching parent partition"]},
                {"title": "Wide (shuffle)", "points": ["Map tasks write files bucketed by key; reduce tasks fetch them over the network",
                                                      "Starts a new stage", "A lost partition needs map output from every parent partition"]},
            ]},
        ]},
        {"id": "stages", "title": "Cutting a job into stages: the quiz", "blocks": [
            {"type": "p", "text": "The deck's exercise (p21–22): textFile → map → filter on one side, hadoopFile → groupByKey → map on "
                                  "the other, then join → map. The scheduler starts from the final RDD and walks back: narrow parents "
                                  "join the current stage, a shuffle dependency starts a parent stage."},
            {"type": "steps", "items": [
                "Final map and join: the join shuffles both inputs, so the result stage holds join and the final map.",
                "Left input: textFile → map → filter are narrow, so they form one shuffle-map stage.",
                "Right input: map after groupByKey is narrow, but groupByKey shuffles, so groupByKey → map is a stage, and hadoopFile another.",
                f"Total: {_Q['n_stages']} stages and {_Q['tasks_total']} tasks with 3 partitions per RDD.",
            ]},
            {"type": "figure", "figure": "dag-stages", "title": "The quiz pipeline, cut by the scheduler",
             "caption": "Orange arrows are shuffles. Switch to mapValues to keep the partitioner after groupByKey.",
             "controls": [{"k": "copart", "label": "Keep the partitioner (mapValues)", "options": [["no", "map"], ["yes", "mapValues"]], "value": "no"}]},
            {"type": "eq", "text": "stages = 1 + number of shuffle dependencies in the lineage\ntasks(stage) = partitions of the stage's last RDD",
             "where": [["shuffle dependency", "each one ends a shuffle-map stage; a stage reached twice is counted once"],
                       ["last RDD", "for a shuffle-map stage, the RDD whose output is written to shuffle files"]]},
            {"type": "quiz", "q": f"With mapValues instead of map after groupByKey, how many stages does the quiz job have?",
             "options": ["4", str(_QC["n_stages"]), "2"], "answer": 1,
             "why": "mapValues keeps the HashPartitioner, so join reads that side through a narrow dependency: the groupByKey stage and the join stage merge."},
            {"type": "try", "label": "Open the quiz in the lab", "params": {"pipe": "quiz", "parts": 3, "copart": False, "stage": 1}},
        ]},
        {"id": "cache", "title": "Caching skips stages", "blocks": [
            {"type": "p", "text": "Before running a stage the scheduler checks whether its output already exists: a persisted RDD whose "
                                  "partitions are all in memory, or shuffle files from an earlier job. Everything upstream of such data "
                                  "is <b>skipped</b>. The paper's figure (p20) has B already in RAM, so only stages 2 and 3 run."},
            {"type": "list", "items": [
                f"Paper figure, B persisted, shuffle files dropped between jobs: job 1 runs {len([s for s in _P20['jobs'][0]['stages'] if not s['skipped']])} stages, "
                f"job 2 runs {len([s for s in _P20['jobs'][1]['stages'] if not s['skipped']])} and skips stage "
                f"{', '.join(str(s['id']) for s in _P20['jobs'][1]['stages'] if s['skipped'])}.",
                f"Log mining, three actions on errors: {_LM0['partitions_computed']} partition computations without cache, "
                f"{_LM1['partitions_computed']} with errors persisted.",
            ]},
            {"type": "callout", "tone": "info", "title": "Shuffle files are a cache too",
             "text": "Spark keeps map outputs on local disk. A later job over the same shuffle skips the map stage even without persist(); the Spark UI shows it as \"skipped\"."},
            {"type": "try", "label": "Paper figure with B in RAM", "params": {"pipe": "paper", "cache": True, "actions": 2, "reuse": False, "stage": 2}},
        ]},
        {"id": "join", "title": "Co-partitioned joins and partitioners", "blocks": [
            {"type": "p", "text": "A join must bring equal keys together. If both inputs are already hash-partitioned with the same "
                                  "partitioner, partition i of one side only needs partition i of the other: the join is narrow. "
                                  "<code>partitionBy(HashPartitioner(n))</code> places each key in partition hashCode(key) mod n."},
            {"type": "figure", "figure": "dag-hash", "title": "HashPartitioner(n) on the 12 customers' regions",
             "caption": "Java's String.hashCode of each region, made non-negative, mod n.",
             "controls": [{"k": "n", "label": "n", "min": 1, "max": 8, "step": 1, "value": 4}]},
            {"type": "figure", "figure": "dag-resize", "title": "repartition against coalesce",
             "caption": "repartition is a full shuffle; coalesce merges neighbouring partitions through a narrow dependency.",
             "controls": [{"k": "parts", "label": "Input partitions", "min": 2, "max": 6, "step": 1, "value": 6}]},
        ]},
        {"id": "recovery", "title": "Recovering a lost partition", "blocks": [
            {"type": "p", "text": "When an executor fails, its cached partitions and the shuffle files it wrote are gone. Spark resubmits "
                                  "tasks for exactly the missing partitions, following the lineage back to data that still exists."},
            {"type": "figure", "figure": "dag-recovery", "title": "What recomputes when one executor fails",
             "caption": "Partition k of every RDD lives on executor k mod 3.",
             "controls": [{"k": "lose", "label": "Lost partition", "min": 0, "max": 2, "step": 1, "value": 1},
                          {"k": "pipe", "label": "Pipeline", "options": [["quiz", "Quiz"], ["paper", "p20 (B cached)"], ["logmine", "Logs (errors cached)"]], "value": "quiz"}]},
            {"type": "quiz", "q": "A partition of a narrow-only chain (textFile → map → filter) is lost. What is recomputed?",
             "options": ["The whole RDD", "The same-numbered partition of each RDD in the chain", "Nothing: a replica is used"], "answer": 1,
             "why": "Each child partition depends on one parent partition, so only that path is re-run, from the replicated input file."},
            {"type": "try", "label": "Lose partition 1 in the quiz job", "params": {"pipe": "quiz", "lose": 1, "stage": 4}},
        ]},
        {"id": "iterative", "title": "Iterative jobs: why caching matters", "blocks": [
            {"type": "p", "text": "Logistic regression reads the same points every iteration. MapReduce writes and re-reads them from "
                                  "HDFS each time; Spark persists them in memory after the first pass (Spark intro p12–13: 100 GB on "
                                  "50 nodes, 110 s per MapReduce iteration, about 1 s per cached Spark iteration)."},
            {"type": "eq", "text": "T_MR(n) = n · t_disk\nT_Spark(n) = t_disk + (n − 1) · t_mem\nspeed-up(n) = n · t_disk / (t_disk + (n − 1) · t_mem)",
             "where": [["t_disk", "one pass that reads the input from disk: 110 s for 100 GB"], ["t_mem", "one pass over cached data: 1 s"],
                       ["n", "number of iterations"]]},
            {"type": "figure", "figure": "dag-iter", "title": "Running time against iterations",
             "caption": "Data beyond the cluster's memory is re-read from disk each iteration, so very large inputs lose most of the gain.",
             "controls": [{"k": "data_gb", "label": "Data size (GB)", "min": 10, "max": 5000, "step": 10, "value": 100}]},
            {"type": "figure", "figure": "dag-speedup", "title": "The speed-up never reaches 100× at 10 iterations",
             "caption": "Whole-job ratio with the slide's 110 s and 1 s."},
        ]},
        {"id": "slides", "title": "Notes on the slides and on this simulation", "blocks": [
            {"type": "callout", "tone": "warn", "title": "Slide errors and the model's limits",
             "text": f"Spark intro p13 claims \"100 time faster at 10 iterations\", but its own text and chart give about "
                     f"{_IT['speedup_10']:.0f}× (1,100 s against about 120 s); 100× is the per-iteration ratio after the first load. "
                     "The implementation deck never states the quiz's stage count (p22 only draws four outlines); it is 4. p12 says "
                     "reduceByKey \"may not be wide\" while p13 and p38 list it as a shuffle: it is wide unless its input is already "
                     "partitioned by the same partitioner. p13 lists coalesce as wide, but p50 is right that it is narrow by default. "
                     "p36 calls join a narrow dependency by looking only at the map after it; the cogroup underneath shuffles. "
                     "The engine is a plain-Python simulation of Spark's scheduling: stages, tasks, caching and recovery follow Spark's "
                     "rules, but there is no real cluster, timing or locality; repartition deals records round-robin from each "
                     "partition's own index instead of a random start."},
        ]},
    ],
    "takeaways": [
        "Narrow dependencies are pipelined inside a stage; every shuffle starts a new stage.",
        f"The quiz job has {_Q['n_stages']} stages; keeping the partitioner with mapValues makes the join narrow and gives {_QC['n_stages']}.",
        "Tasks per stage = partitions of the stage's last RDD.",
        "Persisted RDDs and kept shuffle files let later jobs skip whole stages.",
        "A lost partition is recomputed from lineage: narrow paths rebuild one partition, shuffles re-run only the lost map tasks.",
        f"For iterative jobs Spark is about {_IT['speedup_10']:.0f}× faster at 10 iterations, approaching 110× as iterations grow.",
    ],
    "glossary": [
        ["Job", "All the work triggered by one action."],
        ["Stage", "A set of tasks that run the same pipelined narrow transformations on different partitions."],
        ["Task", "One stage's work on one partition, run on one executor."],
        ["DAGScheduler", "Builds stages from the lineage, skips finished ones and resubmits failed ones."],
        ["TaskScheduler", "Launches tasks on executors and relaunches failed tasks."],
        ["Co-partitioned", "Two RDDs hash-partitioned with the same partitioner, so equal keys sit in partitions with the same index."],
        ["coalesce", "Reduce the number of partitions by merging neighbours, without a shuffle."],
    ],
}, order=6)
