"""Flask application factory.

Run locally:
    flask --app modlab.server run --port 8000          (development)
    python -m modlab.server                             (same, port from $PORT)
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

from flask import Flask, jsonify, render_template
from werkzeug.exceptions import HTTPException

from . import catalog
from .api import ApiError
from .jobs import JobManager, blueprint as jobs_blueprint

# Each module defines `bp` (a Blueprint) and may register job kinds on import.
API_MODULES = ["spread", "crossover", "shape", "markets", "agents", "bigdata", "explain", "assistant"]


def load_env(path: Path | None = None) -> None:
    """Read KEY=VALUE lines from the project's .env into the environment (existing values win).
    Used for local secrets such as GEMINI_API_KEY; .env is ignored by git and Docker."""
    for p in [path] if path else [Path.cwd() / ".env", Path(__file__).resolve().parents[3] / ".env"]:
        if not p or not p.is_file():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.split(" #")[0].strip().strip('"').strip("'")
            if k and v and k not in os.environ:
                os.environ[k] = v
        return


def create_app(config: dict | None = None) -> Flask:
    load_env()
    app = Flask(__name__)
    app.config.update(JSON_SORT_KEYS=False, MAX_CONTENT_LENGTH=1 << 20, JOB_WORKERS=int(os.environ.get("MODLAB_JOB_WORKERS", 1)))
    app.config.update(config or {})
    app.json.sort_keys = False

    for name in API_MODULES:
        mod = importlib.import_module(f"{__name__}.api_{name}")
        app.register_blueprint(mod.bp)
    importlib.import_module(f"{__name__}.labs_catalog")          # after the section providers it summarises
    app.extensions["jobs"] = JobManager(workers=app.config["JOB_WORKERS"])
    app.register_blueprint(jobs_blueprint(app.extensions["jobs"]))

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/favicon.ico")
    def favicon():
        """Browsers ask for /favicon.ico on their own; serve the app icon there too."""
        return app.send_static_file("icons/favicon.ico")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/api/catalog")
    def get_catalog():
        """Reference data for the interface: tickers, meshes, model and agent names, dataset facts."""
        return jsonify(catalog.build())

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
