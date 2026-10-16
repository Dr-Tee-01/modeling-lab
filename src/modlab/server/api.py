"""Helpers every API blueprint uses: errors, parameter parsing, JSON output.

Conventions
-----------
* Errors are always ``{"error": {"code": str, "message": str}}`` with a 4xx/5xx status.
* Successful responses are JSON objects with a ``meta`` block:
  ``{"compute_ms": float, "cached": bool}``. ``compute_ms`` is the server time
  spent producing the result, so the frontend can show that it was computed live.
* Parameters come from the query string (GET) or a JSON body (POST) and are
  validated with :func:`num`, :func:`choice` and :func:`flag`.
"""
from __future__ import annotations

import functools
import math
import time
from typing import Any, Callable, Iterable

import numpy as np
from flask import jsonify, request


class ApiError(Exception):
    def __init__(self, message: str, code: str = "bad_request", status: int = 400):
        super().__init__(message)
        self.message, self.code, self.status = message, code, status


def _source() -> dict:
    if request.method in ("POST", "PUT", "PATCH"):
        body = request.get_json(silent=True)
        if body is not None and not isinstance(body, dict):
            raise ApiError("The request body must be a JSON object.")
        merged = dict(request.args)
        merged.update(body or {})
        return merged
    return request.args


def num(name: str, default: float | None, lo: float, hi: float, integer: bool = False) -> float | int:
    """Read a number in [lo, hi]. Missing -> default; malformed or out of range -> 400."""
    raw = _source().get(name)
    if raw is None or raw == "":
        if default is None:
            raise ApiError(f"Missing parameter '{name}'.", "missing_parameter")
        return default
    try:
        v = float(raw)
    except (TypeError, ValueError):
        raise ApiError(f"Parameter '{name}' must be a number, got {raw!r}.", "invalid_parameter")
    if not math.isfinite(v) or v < lo or v > hi:
        raise ApiError(f"Parameter '{name}' must be between {lo:g} and {hi:g}, got {raw!r}.", "out_of_range")
    if integer:
        if v != int(v):
            raise ApiError(f"Parameter '{name}' must be a whole number, got {raw!r}.", "invalid_parameter")
        return int(v)
    return v


def choice(name: str, default: str, options: Iterable[str]) -> str:
    options = list(options)
    raw = _source().get(name, default)
    if raw not in options:
        raise ApiError(f"Parameter '{name}' must be one of {', '.join(options)}; got {raw!r}.", "invalid_parameter")
    return raw


def choices(name: str, default: list[str], options: Iterable[str]) -> list[str]:
    """Comma-separated list (query) or JSON list (body), every item from options."""
    options = list(options)
    raw = _source().get(name)
    if raw is None or raw == "":
        return list(default)
    items = raw if isinstance(raw, list) else str(raw).split(",")
    items = [str(i).strip() for i in items if str(i).strip()]
    bad = [i for i in items if i not in options]
    if bad or not items:
        raise ApiError(f"Parameter '{name}' takes items from {', '.join(options)}; got {raw!r}.", "invalid_parameter")
    return list(dict.fromkeys(items))


def flag(name: str, default: bool = False) -> bool:
    raw = _source().get(name)
    if raw is None or raw == "":
        return default
    if isinstance(raw, bool):
        return raw
    if str(raw).lower() in ("1", "true", "yes", "on"):
        return True
    if str(raw).lower() in ("0", "false", "no", "off"):
        return False
    raise ApiError(f"Parameter '{name}' must be true or false, got {raw!r}.", "invalid_parameter")


def clean(o: Any, digits: int = 6) -> Any:
    """Make numpy/pandas output JSON-safe and compact (NaN/inf -> null)."""
    if isinstance(o, dict):
        return {str(k): clean(v, digits) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v, digits) for v in o]
    if isinstance(o, np.ndarray):
        if o.dtype.kind == "f":
            return _round_array(o, digits).tolist()
        if o.dtype.kind in "iub":
            return o.tolist()
        return clean(o.tolist(), digits)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return float(f"{f:.{digits}g}") if math.isfinite(f) else None
    return o


def _round_array(a: np.ndarray, digits: int) -> np.ndarray:
    """Round to `digits` significant figures, vectorised; NaN/inf become None."""
    a = np.asarray(a, dtype=float)
    with np.errstate(all="ignore"):
        mag = np.floor(np.log10(np.abs(a)))
        mag[~np.isfinite(mag)] = 0
        scale = 10.0 ** (digits - 1 - mag)
        r = np.round(a * scale) / scale
    if np.isfinite(r).all():
        return r
    out = r.astype(object)
    out[~np.isfinite(r)] = None
    return out


def endpoint(fn: Callable[..., dict]) -> Callable[..., Any]:
    """Wrap a view that returns a dict: time it, clean it, attach meta."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        cached = bool(out.pop("_cached", False)) if isinstance(out, dict) else False
        body = clean(out)
        body["meta"] = {"compute_ms": round((time.perf_counter() - t0) * 1000, 2), "cached": cached}
        return jsonify(body)
    return wrapper
