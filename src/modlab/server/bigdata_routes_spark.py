"""/api/bigdata/{rpc,rdd,dag}: RPC over an unreliable network, the RDD playground and Spark stages.

Every response is computed here by plain-Python simulations (modlab.bigdata.rpc and
modlab.bigdata.spark_core); the browser only draws it.
"""
from __future__ import annotations

from ..bigdata import rpc as R
from ..bigdata import spark_core as S
from ..bigdata.datasets import EMP8, REGIONS12, SPARK_CUSTOMERS, SPARK_ORDERS
from .api import choice, endpoint, flag, num
from .api_bigdata import bp
from .catalog import section


@section("bigdata_spark")
def _catalog() -> dict:
    return {
        "rpc": {"policies": [[k, R.POLICY_NAMES[k]] for k in R.POLICIES], "semantics": R.SEMANTICS,
                "account": R.ACCOUNT, "server_ms": R.SERVER_MS},
        "rdd": {"pipes": [[k, v["label"]] for k, v in S.RDD_PIPES.items()],
                "info": S.RDD_PIPES, "emp8": len(EMP8), "customers": len(SPARK_CUSTOMERS), "orders": len(SPARK_ORDERS),
                "regions": len(REGIONS12)},
        "dag": {"pipes": [[k, v["label"]] for k, v in S.DAG_PIPES.items()], "info": S.DAG_PIPES,
                "executors": S.DAG_EXECUTORS, "iter": {"mr_s": 110, "spark_s": 1, "data_gb": 100, "nodes": 50,
                                                       "cache_gb": S.ITER_CACHE_GB}},
    }


@bp.get("/rpc")
@endpoint
def rpc():
    """N withdraw() calls over a lossy, duplicating network, under one retry policy, plus marshalling."""
    calls = num("calls", 10, 1, 30, integer=True)
    balance = num("balance", 2000, 100, 10000)
    amount = num("amount", 100, 10, 1000)
    latency = num("latency", 50, 1, 500)
    jitter = num("jitter", 30, 0, 300)
    loss = num("loss", 0.2, 0, 0.6)
    dup = num("dup", 0.05, 0, 0.5)
    timeout = num("timeout", 250, 20, 2000)
    retries = num("retries", 3, 0, 6, integer=True)
    seed = num("seed", 4, 0, 999, integer=True)
    policy = choice("policy", "retry", R.POLICIES)
    account = num("account", R.ACCOUNT, 0, 2 ** 31 - 1, integer=True)
    out = R.simulate(calls, balance, amount, latency, jitter, loss, dup, timeout, retries, policy, seed)
    out["compare"] = {p: {k: v for k, v in R.simulate(calls, balance, amount, latency, jitter, loss, dup, timeout, retries, p,
                                                      seed, detail=False).items()}
                      for p in R.POLICIES}
    out["sweep"] = R.sweep(calls, balance, amount, latency, jitter, dup, timeout, retries)
    out["marshal"] = R.marshal(account, amount)
    out["params"] = {"latency": latency, "jitter": jitter, "loss": loss, "dup": dup, "timeout": timeout,
                     "retries": retries, "seed": seed}
    out["round_trip_ms"] = 2 * latency + jitter + R.SERVER_MS
    return out


@bp.get("/rdd")
@endpoint
def rdd():
    """One course pipeline on the mini RDD engine: records per partition at every step."""
    pipe = choice("pipe", "avg_reduce", list(S.RDD_PIPES))
    parts = num("parts", 2, 1, 4, integer=True)
    persist = flag("persist", False)
    actions = num("actions", 1, 0, 3, integer=True)
    stats = flag("stats", False)
    out = S.rdd_lab(pipe, parts, persist, actions, stats)
    out["step"] = min(num("step", len(out["steps"]), 1, 12, integer=True), len(out["steps"]))
    return out


@bp.get("/dag")
@endpoint
def dag():
    """Stages, tasks, caching, co-partitioning, lineage recovery and the iterative-job cost model."""
    pipe = choice("pipe", "quiz", list(S.DAG_PIPES))
    parts = num("parts", 3, 1, 6, integer=True)
    out = S.dag_lab(pipe, parts, flag("cache", False), num("actions", 1, 1, 3, integer=True), flag("reuse", True),
                    flag("copart", False), num("target", 2, 1, 8, integer=True),
                    choice("how", "repartition", ["repartition", "coalesce"]),
                    num("lose", -1, -1, S.DAG_EXECUTORS - 1, integer=True))
    out["stage"] = min(num("stage", 1, 1, 8, integer=True), out["n_stages"])
    out["iter"] = S.iterative_cost(num("iterations", 10, 1, 30, integer=True), num("data_gb", 100, 10, 5000))
    return out
