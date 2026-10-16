"""Background jobs for computations too slow for one HTTP request (LSTM/GRU training).

A job runs in a worker thread and reports progress through a callback. The
client starts it with ``POST /api/jobs`` and polls ``GET /api/jobs/<id>``.
Identical requests while a job is queued or running return the same job.

Register a kind with :func:`register`::

    @register("lstm_trap", params=parse_fn)
    def run(params, progress):      # progress(fraction 0..1, message)
        ...
        return result_dict
"""
from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from flask import Blueprint, request

from .api import ApiError, clean, endpoint

KINDS: dict[str, tuple[Callable, Callable]] = {}


def register(kind: str, params: Callable[[dict], dict]):
    """params(raw_body) validates the request and returns the job's parameters."""
    def deco(fn):
        KINDS[kind] = (params, fn)
        return fn
    return deco


class JobManager:
    def __init__(self, workers: int = 1, keep: int = 50):
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="modlab-job")
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.keep = keep

    def submit(self, kind: str, params: dict) -> dict:
        key = kind + json.dumps(params, sort_keys=True)
        with self.lock:
            for j in self.jobs.values():
                if j["key"] == key and j["status"] in ("queued", "running"):
                    return j
            job = {"id": uuid.uuid4().hex[:12], "kind": kind, "params": params, "key": key,
                   "status": "queued", "progress": 0.0, "message": "Waiting for a worker",
                   "result": None, "error": None, "created": time.time(), "started": None, "finished": None}
            self.jobs[job["id"]] = job
            self._trim()
        self.pool.submit(self._run, job)
        return job

    def _trim(self):
        done = [j for j in self.jobs.values() if j["status"] in ("done", "failed")]
        for j in sorted(done, key=lambda j: j["created"])[: max(0, len(self.jobs) - self.keep)]:
            self.jobs.pop(j["id"], None)

    def _run(self, job: dict):
        _, fn = KINDS[job["kind"]]
        job["status"], job["started"] = "running", time.time()

        def progress(frac: float, message: str = ""):
            job["progress"] = max(0.0, min(1.0, float(frac)))
            if message:
                job["message"] = message
        try:
            job["result"] = clean(fn(job["params"], progress))
            job["status"], job["progress"], job["message"] = "done", 1.0, "Finished"
        except Exception as exc:  # report the failure to the client instead of hanging
            traceback.print_exc()
            job["status"], job["error"] = "failed", f"{type(exc).__name__}: {exc}"
            job["message"] = "Failed"
        finally:
            job["finished"] = time.time()

    def get(self, job_id: str) -> dict | None:
        return self.jobs.get(job_id)


def public(job: dict) -> dict:
    out = {k: job[k] for k in ("id", "kind", "params", "status", "progress", "message", "result", "error")}
    end = job["finished"] or time.time()
    out["elapsed_s"] = round(end - job["started"], 2) if job["started"] else 0.0
    return out


def blueprint(manager: JobManager) -> Blueprint:
    bp = Blueprint("jobs", __name__, url_prefix="/api/jobs")

    @bp.post("")
    @endpoint
    def start():
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or "kind" not in body:
            raise ApiError("Send a JSON body with a 'kind' field.", "missing_parameter")
        kind = body["kind"]
        if kind not in KINDS:
            raise ApiError(f"Unknown job kind {kind!r}. Known: {', '.join(sorted(KINDS))}.", "invalid_parameter")
        parse, _ = KINDS[kind]
        return {"job": public(manager.submit(kind, parse(body)))}

    @bp.get("/<job_id>")
    @endpoint
    def status(job_id: str):
        job = manager.get(job_id)
        if job is None:
            raise ApiError(f"No job with id {job_id!r}.", "not_found", 404)
        return {"job": public(job)}

    @bp.get("/kinds")
    @endpoint
    def kinds():
        return {"kinds": sorted(KINDS)}

    return bp
