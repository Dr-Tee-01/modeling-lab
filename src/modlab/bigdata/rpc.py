"""RPC over an unreliable network (IE494 L4): a seeded, deterministic simulation.

A client calls ``withdraw(account, amount)`` on a bank server N times, one call after another.
Every message crosses a network that delays it (latency + jitter), may lose it, and may
deliver it twice. The client waits ``timeout`` for the reply and then, depending on the
retry policy, gives up or sends the request again:

* ``none``        one attempt. At-most-once *from the client's side*: a timed-out call may or
                  may not have run on the server, and a network duplicate still runs twice.
* ``retry``       resend on timeout with a fresh request id. At-least-once: the server cannot
                  tell a retry from a new withdrawal, so a lost *reply* turns into a double debit.
* ``idempotent``  resend with the same idempotency key; the server remembers the reply for
                  each key and answers repeats from that table without debiting again. Retries
                  give at-least-once delivery and the table makes the effect exactly once.

Slide L4 p3's client ``do_withdraw`` calls ``deposit(accno, amount)``; the server procedure is
``withdraw``. The simulation calls ``withdraw`` (the slide's bug is noted in the explainer).
"""
from __future__ import annotations

import functools
import heapq
import json
import random
import struct

POLICIES = ("none", "retry", "idempotent")
SEMANTICS = {"none": "at-most-once", "retry": "at-least-once", "idempotent": "exactly-once effect"}
POLICY_NAMES = {"none": "No retry", "retry": "Retry on timeout", "idempotent": "Retry + idempotency key"}
ACCOUNT = 1001                 # the account number every call withdraws from
SERVER_MS = 2.0                # time the server spends running withdraw()
DUP_GAP_MS = (5.0, 60.0)       # a duplicated message arrives this much later than the original


def request_json(call: int, attempt: int, amount: float, policy: str) -> str:
    """The marshalled request as it goes on the wire (JSON text, UTF-8)."""
    body = {"id": f"c{call}-a{attempt}", "method": "withdraw", "params": {"account": ACCOUNT, "amount": float(amount)}}
    if policy == "idempotent":
        body["idempotency_key"] = f"k{call}"
    return json.dumps(body, separators=(",", ":"))


def response_json(rid: str, result: float | None, error: str | None) -> str:
    body = {"id": rid, "result": result} if error is None else {"id": rid, "error": error}
    return json.dumps(body, separators=(",", ":"))


def simulate(calls: int = 10, balance: float = 2000.0, amount: float = 100.0, latency: float = 50.0,
             jitter: float = 30.0, loss: float = 0.2, dup: float = 0.05, timeout: float = 250.0,
             retries: int = 3, policy: str = "retry", seed: int = 7, detail: bool = True) -> dict:
    """Run N sequential withdraw() calls through the network. Deterministic for a given seed."""
    if policy not in POLICIES:
        raise ValueError(policy)
    rng = random.Random(seed)
    lat = lambda: latency + rng.uniform(0, jitter)
    q: list = []
    seq = [0]

    def push(t, kind, **kw):
        seq[0] += 1
        heapq.heappush(q, (t, seq[0], kind, kw))

    msgs: list[dict] = []          # every message put on the wire
    marks: list[dict] = []         # client-side timeouts / give-ups / completions
    server = {"balance": float(balance), "table": {}, "log": []}
    st = [{"call": c + 1, "attempts": 0, "executions": 0, "status": None, "reply": None, "t_start": None,
           "t_end": None, "dedup": 0, "late": 0} for c in range(calls)]
    cur = {"c": 0}

    def send(t, src, kind, call, attempt, text, payload):
        """Put a message on the wire: maybe lose it, maybe duplicate it."""
        lost = rng.random() < loss
        m = {"i": len(msgs), "call": call, "attempt": attempt, "kind": kind, "from": src, "t_send": round(t, 2),
             "bytes": len(text.encode()), "lost": lost, "copy": False, "t_arrive": None, "outcome": None}
        msgs.append(m)
        if lost:
            m["t_lost"] = round(t + latency / 2, 2)
        else:
            m["t_arrive"] = round(t + lat(), 2)
            push(m["t_arrive"], "arrive", m=m["i"], payload=payload)
        if rng.random() < dup:                      # the network delivers a second copy
            d = dict(m, i=len(msgs), copy=True, lost=False, t_arrive=round(t + lat() + rng.uniform(*DUP_GAP_MS), 2),
                     outcome=None)
            d.pop("t_lost", None)
            msgs.append(d)
            push(d["t_arrive"], "arrive", m=d["i"], payload=payload)

    def start_call(t, c):
        s = st[c]
        s["t_start"] = round(t, 2)
        attempt(t, c)

    def attempt(t, c):
        s = st[c]
        s["attempts"] += 1
        a = s["attempts"]
        text = request_json(c + 1, a, amount, policy)
        send(t, "client", "request", c + 1, a, text, {"call": c, "attempt": a, "key": f"k{c + 1}"})
        push(t + timeout, "timeout", call=c, attempt=a)

    def finish(t, c, status, reply=None):
        s = st[c]
        s["status"], s["reply"], s["t_end"] = status, reply, round(t, 2)
        marks.append({"t": round(t, 2), "call": c + 1, "attempt": s["attempts"], "kind": status})
        if c + 1 < calls:
            cur["c"] = c + 1
            start_call(t, c + 1)

    start_call(0.0, 0)
    while q:
        t, _, kind, kw = heapq.heappop(q)
        if kind == "timeout":
            c, a = kw["call"], kw["attempt"]
            s = st[c]
            if s["status"] is not None or a != s["attempts"]:
                continue
            if policy == "none" or s["attempts"] > retries:
                marks.append({"t": round(t, 2), "call": c + 1, "attempt": a, "kind": "give up"})
                finish(t, c, "unknown")
            else:
                marks.append({"t": round(t, 2), "call": c + 1, "attempt": a, "kind": "retry"})
                attempt(t, c)
            continue
        m = msgs[kw["m"]]
        p = kw["payload"]
        if m["kind"] == "request":                 # the server receives a request
            c = p["call"]
            key = p["key"]
            if policy == "idempotent" and key in server["table"]:
                res, err = server["table"][key]
                m["outcome"] = "dedup"
                st[c]["dedup"] += 1
            else:
                if server["balance"] >= amount:
                    server["balance"] -= amount
                    res, err = server["balance"], None
                    m["outcome"] = "debit"
                    st[c]["executions"] += 1
                else:
                    res, err = None, "InsufficientBalance"
                    m["outcome"] = "error"
                if policy == "idempotent":
                    server["table"][key] = (res, err)
            server["log"].append({"t": round(t, 2), "call": c + 1, "attempt": p["attempt"], "outcome": m["outcome"],
                                  "balance": server["balance"], "copy": m["copy"]})
            rid = f"c{c + 1}-a{p['attempt']}"
            send(t + SERVER_MS, "server", "response", c + 1, p["attempt"], response_json(rid, res, err),
                 {"call": c, "attempt": p["attempt"], "result": res, "error": err})
        else:                                      # the client receives a reply
            c = p["call"]
            s = st[c]
            if s["status"] is None and cur["c"] == c:
                m["outcome"] = "accepted"
                finish(t, c, "error" if p["error"] else "ok", p["result"])
            else:
                m["outcome"] = "ignored (late)"
                s["late"] += 1

    funds = int(balance // amount) if amount > 0 else calls
    should = min(calls, funds)
    expected = balance - amount * should
    executions = sum(s["executions"] for s in st)
    doubles = sum(max(0, s["executions"] - 1) for s in st)
    missed = max(0, should - sum(1 for s in st if s["executions"] > 0))
    unknown = sum(1 for s in st if s["status"] == "unknown")
    believed = sum(1 for s in st if s["status"] == "ok")
    out = {
        "policy": policy, "semantics": SEMANTICS[policy], "calls": calls, "amount": amount, "start_balance": balance,
        "final_balance": server["balance"], "expected_balance": expected, "difference": server["balance"] - expected,
        "executions": executions, "double_debits": doubles, "missed": missed, "unknown": unknown,
        "client_ok": believed, "client_errors": sum(1 for s in st if s["status"] == "error"),
        "client_believes_balance": balance - amount * believed,
        "messages": len(msgs), "lost": sum(m["lost"] for m in msgs), "duplicates": sum(m["copy"] for m in msgs),
        "retries": sum(s["attempts"] - 1 for s in st), "dedup_hits": sum(s["dedup"] for s in st),
        "late_replies": sum(s["late"] for s in st),
        "bytes": sum(m["bytes"] for m in msgs), "bytes_useful": 0,
        "t_end": max([m.get("t_arrive") or m.get("t_lost") or 0 for m in msgs] + [mk["t"] for mk in marks]),
    }
    # bytes of one request and one reply on a clean network (what N calls would cost with no faults)
    out["bytes_useful"] = calls * (len(request_json(1, 1, amount, policy).encode())
                                   + len(response_json("c1-a1", balance - amount, None).encode()))
    if detail:
        out["timeline"] = msgs
        out["marks"] = marks
        out["per_call"] = [{k: s[k] for k in ("call", "attempts", "executions", "dedup", "late", "status", "reply",
                                               "t_start", "t_end")} for s in st]
        out["server_log"] = server["log"]
    return out


@functools.lru_cache(maxsize=128)
def sweep(calls: int, balance: float, amount: float, latency: float, jitter: float, dup: float, timeout: float,
          retries: int, seeds: int = 12) -> dict:
    """Average double debits, missed debits and unknown outcomes against the loss rate, per policy."""
    losses = [round(0.05 * i, 2) for i in range(11)]
    out = {"loss": losses, "seeds": seeds}
    for pol in POLICIES:
        dd, ms, un, wrong = [], [], [], []
        for L in losses:
            rs = [simulate(calls, balance, amount, latency, jitter, L, dup, timeout, retries, pol, s, detail=False)
                  for s in range(seeds)]
            dd.append(sum(r["double_debits"] for r in rs) / seeds)
            ms.append(sum(r["missed"] for r in rs) / seeds)
            un.append(sum(r["unknown"] for r in rs) / seeds)
            wrong.append(sum(r["final_balance"] != r["expected_balance"] for r in rs) / seeds)
        out[pol] = {"double_debits": dd, "missed": ms, "unknown": un, "wrong_balance": wrong}
    return out


# --------------------------------------------------------------------------- marshalling
def _hex(b: bytes) -> list[str]:
    return [f"{x:02X}" for x in b]


def marshal(account: int = ACCOUNT, amount: float = 100.0) -> dict:
    """How the two arguments look on the wire: binary (big- and little-endian) and as JSON."""
    i_be, i_le = struct.pack(">i", account), struct.pack("<i", account)
    f_be, f_le = struct.pack(">d", amount), struct.pack("<d", amount)
    j = json.dumps({"method": "withdraw", "params": {"account": account, "amount": float(amount)}}, separators=(",", ":"))
    return {
        "account": account, "amount": amount,
        "int32": {"be": _hex(i_be), "le": _hex(i_le), "misread": struct.unpack("<i", i_be)[0]},
        "float64": {"be": _hex(f_be), "le": _hex(f_le), "misread": struct.unpack("<d", f_be)[0],
                    "sign": (struct.unpack(">Q", f_be)[0] >> 63) & 1,
                    "exponent": ((struct.unpack(">Q", f_be)[0] >> 52) & 0x7FF) - 1023,
                    "mantissa_hex": f"{struct.unpack('>Q', f_be)[0] & ((1 << 52) - 1):013X}"},
        "float32": {"be": _hex(struct.pack(">f", amount)), "le": _hex(struct.pack("<f", amount)),
                    "roundtrip": struct.unpack(">f", struct.pack(">f", amount))[0]},
        "json": j, "json_bytes": len(j.encode()), "binary_bytes": 4 + 8,
    }
