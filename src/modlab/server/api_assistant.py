"""/api/assistant: ask Gemini about the page you are on, by text or voice, and hear the answer.

The browser never talks to Gemini and never sees the key. Answers stream back as
server-sent events: ``transcript`` (for a voice question), ``delta`` (text as it is
written), then ``done`` or ``error``.
"""
from __future__ import annotations

import base64
import binascii
import json
import time

from flask import Blueprint, Response, request, stream_with_context

from ..assistant import gemini, service as A
from .api import ApiError, endpoint

bp = Blueprint("assistant", __name__, url_prefix="/api/assistant")


def _client() -> str:
    fwd = request.headers.get("X-Forwarded-For", "")
    return (fwd.split(",")[0].strip() or request.remote_addr or "?")[:64]


def _lang(raw) -> str:
    lang = str(raw or A.DEFAULT_LANG)
    if lang not in A.LANGS:
        raise ApiError(f"lang must be one of {', '.join(A.LANGS)}.", "invalid_parameter")
    return lang


def _require_on():
    if not A.enabled():
        raise ApiError("The assistant is not enabled on this server (no Gemini API key).", "assistant_off", 503)


def _limit(kind: str, per_window: int):
    msg = A.LIMITER.check(_client(), kind, per_window)
    if msg:
        raise ApiError(msg, "rate_limited", 429)


@bp.get("/status")
@endpoint
def status():
    """Whether the assistant is on, its models, languages and limits (never the key)."""
    s = A.settings()
    return {"enabled": A.enabled(), "model": s["model"], "tts_model": s["tts_model"],
            "languages": [{"id": k, "label": v} for k, v in A.LANGS.items()], "default_lang": A.DEFAULT_LANG,
            "max_record_s": A.MAX_RECORD_S, "audio_types": sorted(A.AUDIO_TYPES)}


@bp.get("/suggestions")
@endpoint
def suggestions():
    """Starter questions for a page, taken from its explainer."""
    view = str(request.args.get("view", "") or "")[:80] or None
    return {"suggestions": A.suggestions(view, _lang(request.args.get("lang")))}


def _page(raw) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    lists = lambda k, n: [str(x)[:500] for x in (raw.get(k) or [])[:n] if isinstance(x, (str, int, float))]
    params = raw.get("params") if isinstance(raw.get("params"), dict) else {}
    return {"title": str(raw.get("title", ""))[:200], "kind": str(raw.get("kind", "page"))[:20],
            "section": str(raw.get("section", ""))[:200], "kpis": lists("kpis", 8), "insights": lists("insights", 6),
            "params": {str(k)[:40]: str(v)[:60] for k, v in list(params.items())[:30]}}


def _audio(raw) -> tuple[str, str] | None:
    if not raw:
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("data"), str):
        raise ApiError("audio must be {data: base64, mime_type}.", "invalid_parameter")
    mime = str(raw.get("mime_type", "")).split(";")[0].strip().lower()
    mime = {"audio/mp4": "audio/m4a", "audio/x-m4a": "audio/m4a", "audio/x-wav": "audio/wav"}.get(mime, mime)
    if mime not in A.AUDIO_TYPES:
        raise ApiError(f"Unsupported audio type {mime!r}.", "invalid_parameter")
    try:
        size = len(base64.b64decode(raw["data"], validate=True))
    except (binascii.Error, ValueError):
        raise ApiError("audio data is not valid base64.", "invalid_parameter")
    if size > A.MAX_AUDIO_BYTES:
        raise ApiError(f"The recording is too long (keep it under {A.MAX_RECORD_S} seconds).", "too_large", 413)
    if size < 200:
        raise ApiError("The recording is empty.", "invalid_parameter")
    return raw["data"], mime


def _history(raw) -> list[dict]:
    out = []
    for t in (raw or [])[-10:]:
        if isinstance(t, dict) and t.get("role") in ("user", "model") and isinstance(t.get("text"), str) and t["text"].strip():
            out.append({"role": t["role"], "text": t["text"][:4000]})
    return out


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@bp.post("/ask")
def ask():
    """Ask about the current page. Body: {question | audio, lang, view, page, history}. Streams SSE."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ApiError("The request body must be a JSON object.")
    _require_on()
    lang = _lang(body.get("lang"))
    question = str(body.get("question") or "").strip()[:1200]
    audio = _audio(body.get("audio"))
    if not question and not audio:
        raise ApiError("Ask a question by text or voice.", "missing_parameter")
    view = str(body.get("view") or "")[:80] or None
    page, history = _page(body.get("page")), _history(body.get("history"))
    _limit("ask", 20)

    def run():
        t0 = time.perf_counter()
        q = question
        try:
            if audio:
                q = A.transcribe(audio[0], audio[1], lang)
                if not q:
                    yield _sse("error", {"code": "no_speech", "message": "I couldn't hear a question in that recording."})
                    return
                yield _sse("transcript", {"text": q})
            parts = []
            for chunk in A.answer(q, history, view, page, lang):
                parts.append(chunk)
                yield _sse("delta", {"text": chunk})
            text = "".join(parts).strip()
            if not text:
                yield _sse("error", {"code": "empty", "message": "Gemini returned an empty answer. Please ask again."})
                return
            yield _sse("done", {"text": text, "question": q, "ms": round((time.perf_counter() - t0) * 1000),
                                "model": A.settings()["model"]})
        except gemini.GeminiError as e:
            yield _sse("error", {"code": e.code, "message": e.message})
        except Exception:                                       # never leak internals into the stream
            yield _sse("error", {"code": "internal_error", "message": "The assistant failed on the server."})

    return Response(stream_with_context(run()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@bp.post("/speak")
def speak():
    """Spoken version of an answer. Body: {text, lang}. Returns audio (WAV)."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ApiError("The request body must be a JSON object.")
    _require_on()
    lang = _lang(body.get("lang"))
    text = str(body.get("text") or "").strip()
    if not text:
        raise ApiError("Nothing to read aloud.", "missing_parameter")
    if len(text) > 2500:
        raise ApiError("That answer is too long to read aloud.", "too_large", 413)
    _limit("speak", 20)
    try:
        audio, mime, cached = A.speech(text, lang)
    except gemini.GeminiError as e:
        raise ApiError(e.message, e.code, e.status)
    return Response(audio, mimetype=mime, headers={"Cache-Control": "no-store", "X-Cached": "1" if cached else "0"})
