"""Flask application factory.

Run locally:
    flask --app modlab.server run --port 8000          (development)
    python -m modlab.server                             (same, port from $PORT)
"""
from __future__ import annotations

import importlib
import os

from flask import Flask, jsonify, render_template
from werkzeug.exceptions import HTTPException

from .api import ApiError
from .jobs import JobManager, blueprint as jobs_blueprint

# Each module defines `bp` (a Blueprint) and may register job kinds on import.
API_MODULES = ["spread", "crossover", "shape", "markets"]


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.update(JSON_SORT_KEYS=False, MAX_CONTENT_LENGTH=1 << 20, JOB_WORKERS=int(os.environ.get("MODLAB_JOB_WORKERS", 1)))
    app.config.update(config or {})
    app.json.sort_keys = False

    for name in API_MODULES:
        mod = importlib.import_module(f"{__name__}.api_{name}")
        app.register_blueprint(mod.bp)
    app.extensions["jobs"] = JobManager(workers=app.config["JOB_WORKERS"])
    app.register_blueprint(jobs_blueprint(app.extensions["jobs"]))

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/api/routes")
    def routes():
        """Every API route with its methods and the docstring of its view."""
        out = []
        for rule in sorted(app.url_map.iter_rules(), key=lambda r: r.rule):
            if not rule.rule.startswith("/api/"):
                continue
            view = app.view_functions[rule.endpoint]
            doc = (view.__doc__ or "").strip().splitlines()
            out.append({"path": rule.rule, "methods": sorted(rule.methods - {"HEAD", "OPTIONS"}),
                        "summary": doc[0] if doc else ""})
        return jsonify({"routes": out})

    @app.errorhandler(ApiError)
    def api_error(e: ApiError):
        return jsonify({"error": {"code": e.code, "message": e.message}}), e.status

    @app.errorhandler(HTTPException)
    def http_error(e: HTTPException):
        return jsonify({"error": {"code": (e.name or "error").lower().replace(" ", "_"), "message": e.description}}), e.code

    @app.errorhandler(Exception)
    def internal(e: Exception):
        app.logger.exception("Unhandled error")
        return jsonify({"error": {"code": "internal_error", "message": "The computation failed on the server."}}), 500

    return app
