"""A small client for the Gemini Interactions API (text, streaming text, audio in, speech out).

Uses only the standard library. The API key is read from the environment on every call
(``GEMINI_API_KEY``) and is only ever sent to Google in the ``x-goog-api-key`` header.

Wire format (Interactions API, revision 2026-05-20):
  POST https://generativelanguage.googleapis.com/v1beta/interactions[?alt=sse]
  {"model", "system_instruction", "input": [steps], "generation_config": {...}, "stream", "store"}
  A step is {"type": "user_input" | "model_output", "content": [{"type": "text", "text"} |
  {"type": "audio", "data": base64, "mime_type"}]}. Streaming sends SSE events whose JSON has
  ``event_type``: text arrives in ``step.delta`` events as ``delta = {"type": "text", "text"}``.
"""
from __future__ import annotations

import base64
import json
import os
import struct
import urllib.error
import urllib.request
from typing import Iterator

API = "https://generativelanguage.googleapis.com/v1beta/interactions"
REVISION = "2026-05-20"


class GeminiError(Exception):
    def __init__(self, message: str, status: int = 502, code: str = "gemini_error"):
        super().__init__(message)
        self.message, self.status, self.code = message, status, code


def api_key() -> str | None:
    k = os.environ.get("GEMINI_API_KEY", "").strip()
    return k or None


def _open(body: dict, stream: bool, timeout: float):
    """POST to the Interactions API. Separate so tests can replace it."""
    key = api_key()
    if not key:
        raise GeminiError("The Gemini API key is not configured on the server.", 503, "assistant_off")
    req = urllib.request.Request(
        API + ("?alt=sse" if stream else ""), data=json.dumps(body).encode(), method="POST",
        headers={"x-goog-api-key": key, "Content-Type": "application/json", "Api-Revision": REVISION,
                 "Accept": "text/event-stream" if stream else "application/json"})
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise _http_error(e) from None
    except (urllib.error.URLError, TimeoutError) as e:
        raise GeminiError(f"Could not reach Gemini ({getattr(e, 'reason', e)}).", 502, "gemini_unreachable") from None


def _http_error(e: urllib.error.HTTPError) -> GeminiError:
    msg = f"Gemini returned HTTP {e.code}."
    try:
        err = json.loads(e.read().decode() or "{}").get("error", {})
        msg = err.get("message") or msg
    except Exception:
        pass
    status = 429 if e.code == 429 else 502
    code = "gemini_rate_limited" if e.code == 429 else "gemini_auth" if e.code in (401, 403) else "gemini_error"
    if e.code in (401, 403):
        msg = "Gemini rejected the API key. Check GEMINI_API_KEY on the server."
    return GeminiError(msg, status, code)


def _events(resp) -> Iterator[dict]:
    """Parse a server-sent-event stream into JSON payloads."""
    data: list[str] = []
    for raw in resp:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if line.startswith("data:"):
            data.append(line[5:].lstrip())
        elif line == "" and data:
            payload = "\n".join(data)
            data = []
            if payload.strip() in ("", "[DONE]"):
                continue
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                continue
    if data:
        try:
            yield json.loads("\n".join(data))
        except json.JSONDecodeError:
            pass


def text_of(interaction: dict) -> str:
    """All text in the model's output steps."""
    out = []
    for step in interaction.get("steps", []) or []:
        if step.get("type", "model_output") != "model_output":
            continue
        for c in step.get("content", []) or []:
            if c.get("type") == "text" and c.get("text"):
                out.append(c["text"])
    return "".join(out)


def user(text: str | None = None, audio: tuple[str, str] | None = None) -> dict:
    content = []
    if text:
        content.append({"type": "text", "text": text})
    if audio:
        content.append({"type": "audio", "data": audio[0], "mime_type": audio[1]})
    return {"type": "user_input", "content": content}


def model_turn(text: str) -> dict:
    return {"type": "model_output", "content": [{"type": "text", "text": text}]}


def generate(model: str, steps: list[dict], system: str | None = None, max_tokens: int = 400,
             thinking: str = "minimal", timeout: float = 40) -> str:
    body = {"model": model, "input": steps, "store": False,
            "generation_config": {"max_output_tokens": max_tokens, "thinking_level": thinking}}
    if system:
        body["system_instruction"] = system
    with _open(body, False, timeout) as resp:
        out = json.loads(resp.read().decode() or "{}")
    if out.get("error"):
        raise GeminiError(out["error"].get("message", "Gemini failed."))
    return text_of(out)


def stream(model: str, steps: list[dict], system: str | None = None, max_tokens: int = 900,
           thinking: str = "minimal", timeout: float = 60) -> Iterator[str]:
    """Yield text as Gemini writes it."""
    body = {"model": model, "input": steps, "stream": True, "store": False,
            "generation_config": {"max_output_tokens": max_tokens, "thinking_level": thinking}}
    if system:
        body["system_instruction"] = system
    got = False
    with _open(body, True, timeout) as resp:
        for ev in _events(resp):
            kind = ev.get("event_type")
            if kind == "step.delta":
                d = ev.get("delta") or {}
                if d.get("type") == "text" and d.get("text"):
                    got = True
                    yield d["text"]
            elif kind == "error" or (ev.get("error") and not kind):
                raise GeminiError((ev.get("error") or {}).get("message", "Gemini failed while answering."))
            elif kind == "interaction.completed" and not got:
                text = text_of(ev.get("interaction") or {})
                if text:
                    got = True
                    yield text


def _wav(pcm: bytes, rate: int = 24000, channels: int = 1, width: int = 2) -> bytes:
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack(
        "<IHHIIHH", 16, 1, channels, rate, rate * channels * width, channels * width, width * 8) + b"data" + struct.pack("<I", len(pcm))
    return header + pcm


def speak(model: str, text: str, voice: str, style: str | None = None, timeout: float = 60) -> tuple[bytes, str]:
    """Synthesise speech. Returns (audio bytes, mime type); raw PCM is wrapped as WAV."""
    item = {"type": "text", "text": text}
    if style:
        item["annotations"] = [{"type": "speech_metadata", "style": style}]
    body = {"model": model, "input": [{"type": "user_input", "content": [item]}], "store": False,
            "response_format": {"type": "audio"}, "generation_config": {"speech_config": [{"voice": voice}]}}
    with _open(body, False, timeout) as resp:
        out = json.loads(resp.read().decode() or "{}")
    if out.get("error"):
        raise GeminiError(out["error"].get("message", "Gemini could not synthesise speech."))
    for step in out.get("steps", []) or []:
        for c in step.get("content", []) or []:
            if c.get("type") == "audio" and c.get("data"):
                raw, mime = base64.b64decode(c["data"]), (c.get("mime_type") or "audio/wav").split(";")[0]
                if mime in ("audio/l16", "audio/pcm") or (mime == "audio/wav" and not raw.startswith(b"RIFF")):
                    return _wav(raw, int(c.get("sample_rate") or 24000), int(c.get("channels") or 1)), "audio/wav"
                return raw, mime
    raise GeminiError("Gemini returned no audio.")
