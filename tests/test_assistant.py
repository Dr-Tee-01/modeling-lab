"""The Gemini assistant, tested against a fake Gemini (no network, no real key)."""
import base64
import io
import json

import pytest

from modlab.assistant import gemini, service as A


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def sse(*events):
    return "".join(f"event: x\ndata: {json.dumps(e)}\n\n" for e in events).encode()


@pytest.fixture()
def fake(monkeypatch):
    """Replace the HTTP call. Records request bodies; returns scripted replies in order."""
    calls, replies = [], []

    def _open(body, stream, timeout):
        calls.append({"body": body, "stream": stream})
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return FakeResponse(r)

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(gemini, "_open", _open)
    A.LIMITER.reset()
    A._TTS.clear()
    return calls, replies


def _events(resp):
    out = []
    for block in resp.get_data(as_text=True).strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_status_without_a_key(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    s = client.get("/api/assistant/status").get_json()
    assert s["enabled"] is False and s["default_lang"] == "hinglish"
    assert "key" not in json.dumps(s).lower().replace("api key", "")
    r = client.post("/api/assistant/ask", json={"question": "hi"})
    assert r.status_code == 503 and r.get_json()["error"]["code"] == "assistant_off"


def test_streams_a_grounded_answer(client, fake):
    calls, replies = fake
    replies.append(sse({"event_type": "interaction.created"},
                       {"event_type": "step.delta", "index": 0, "delta": {"type": "text", "text": "Dekho, "}},
                       {"event_type": "step.delta", "index": 0, "delta": {"type": "text", "text": "R₀ = β/α."}},
                       {"event_type": "interaction.completed", "interaction": {}}))
    page = {"title": "Flu in a boarding school", "kind": "model", "params": {"beta": 1.66},
            "kpis": ["Basic reproduction number 3.32"], "insights": ["The outbreak peaks on day 6.5."]}
    r = client.post("/api/assistant/ask", json={"question": "R0 kya hai?", "lang": "hinglish", "view": "spread/sir", "page": page,
                                                "history": [{"role": "user", "text": "hello"}, {"role": "model", "text": "hi"}]})
    assert r.status_code == 200 and r.mimetype == "text/event-stream"
    ev = _events(r)
    assert [e for e, _ in ev] == ["delta", "delta", "done"]
    assert ev[-1][1]["text"] == "Dekho, R₀ = β/α."
    body = calls[0]["body"]
    assert calls[0]["stream"] and body["store"] is False
    assert [s["type"] for s in body["input"]] == ["user_input", "model_output", "user_input"]
    sys = body["system_instruction"]
    assert "Roman (Latin) script" in sys and "beta = 1.66" in sys and "3.32" in sys
    assert "How an epidemic rises and falls" in sys          # the explainer is in the context


def test_voice_question_is_transcribed_then_answered(client, fake):
    calls, replies = fake
    replies.append(json.dumps({"steps": [{"type": "model_output", "content": [{"type": "text", "text": "lockdown se kya hota hai?"}]}]}).encode())
    replies.append(sse({"event_type": "step.delta", "delta": {"type": "text", "text": "Peak neeche aata hai."}}))
    audio = {"data": base64.b64encode(b"\x1aE\xdf\xa3" + b"0" * 400).decode(), "mime_type": "audio/webm;codecs=opus"}
    ev = _events(client.post("/api/assistant/ask", json={"audio": audio, "view": "spread/sir"}))
    assert ev[0] == ("transcript", {"text": "lockdown se kya hota hai?"})
    assert ev[-1][0] == "done" and ev[-1][1]["question"] == "lockdown se kya hota hai?"
    sent = calls[0]["body"]["input"][0]["content"]
    assert sent[1]["type"] == "audio" and sent[1]["mime_type"] == "audio/webm"


def test_english_mode_and_fallback_to_completed_event(client, fake):
    calls, replies = fake
    replies.append(sse({"event_type": "interaction.completed",
                        "interaction": {"steps": [{"type": "model_output", "content": [{"type": "text", "text": "In plain English."}]}]}}))
    ev = _events(client.post("/api/assistant/ask", json={"question": "Explain", "lang": "english"}))
    assert ev[-1] == ("done", ev[-1][1]) and ev[-1][1]["text"] == "In plain English."
    assert "plain English" in calls[0]["body"]["system_instruction"]


def test_gemini_errors_become_stream_errors(client, fake):
    _, replies = fake
    replies.append(gemini.GeminiError("Gemini rejected the API key.", 502, "gemini_auth"))
    ev = _events(client.post("/api/assistant/ask", json={"question": "hi"}))
    assert ev == [("error", {"code": "gemini_auth", "message": "Gemini rejected the API key."})]


def test_validation(client, fake):
    assert client.post("/api/assistant/ask", json={}).status_code == 400
    assert client.post("/api/assistant/ask", json={"question": "x", "lang": "french"}).status_code == 400
    bad = {"data": "not base64!!", "mime_type": "audio/webm"}
    assert client.post("/api/assistant/ask", json={"audio": bad}).status_code == 400
    exe = {"data": base64.b64encode(b"0" * 400).decode(), "mime_type": "application/x-msdownload"}
    assert client.post("/api/assistant/ask", json={"audio": exe}).status_code == 400
    big = {"data": base64.b64encode(b"0" * (A.MAX_AUDIO_BYTES + 1)).decode(), "mime_type": "audio/webm"}
    assert client.post("/api/assistant/ask", json={"audio": big}).status_code == 413


def test_rate_limit(client, fake):
    _, replies = fake
    for _ in range(20):
        replies.append(sse({"event_type": "step.delta", "delta": {"type": "text", "text": "ok"}}))
        assert client.post("/api/assistant/ask", json={"question": "q"}).status_code == 200
    r = client.post("/api/assistant/ask", json={"question": "q"})
    assert r.status_code == 429 and r.get_json()["error"]["code"] == "rate_limited"


def test_speech_is_wav_and_cached(client, fake):
    calls, replies = fake
    pcm = b"\x00\x01" * 2400
    replies.append(json.dumps({"steps": [{"type": "model_output", "content": [
        {"type": "audio", "mime_type": "audio/l16", "data": base64.b64encode(pcm).decode(), "sample_rate": 24000}]}]}).encode())
    r = client.post("/api/assistant/speak", json={"text": "**Dekho**, R₀ = 3.3", "lang": "hinglish"})
    assert r.status_code == 200 and r.mimetype == "audio/wav" and r.data[:4] == b"RIFF" and r.headers["X-Cached"] == "0"
    body = calls[0]["body"]
    assert body["response_format"] == {"type": "audio"} and body["generation_config"]["speech_config"][0]["voice"]
    item = body["input"][0]["content"][0]
    assert "**" not in item["text"] and "Hindi" in item["annotations"][0]["style"]
    again = client.post("/api/assistant/speak", json={"text": "**Dekho**, R₀ = 3.3", "lang": "hinglish"})
    assert again.headers["X-Cached"] == "1" and len(calls) == 1


def test_suggestions_come_from_the_explainer(client):
    s = client.get("/api/assistant/suggestions", query_string={"view": "spread/sir", "lang": "english"}).get_json()["suggestions"]
    assert len(s) >= 3 and any("R₀" in q for q in s)


def test_sse_parser_handles_split_data_lines():
    raw = b'data: {"event_type": "step.delta",\ndata:  "delta": {"type": "text", "text": "a"}}\n\n'
    assert list(gemini._events(io.BytesIO(raw)))[0]["delta"]["text"] == "a"
