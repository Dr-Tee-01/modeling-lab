"""The ModeLab AI assistant: grounded answers about the page you are on, in Hinglish or English.

Grounding: every question is answered with the view's explainer (from modlab.explain) and
what the page currently shows (its settings, result tiles and insights, which the server
computed) as context. Spoken answers use Gemini text-to-speech.
"""
from __future__ import annotations

import collections
import hashlib
import os
import re
import threading
import time

from . import gemini

LANGS = {"hinglish": "Hinglish", "english": "English"}
DEFAULT_LANG = "hinglish"
AUDIO_TYPES = {"audio/webm", "audio/ogg", "audio/wav", "audio/mp3", "audio/mpeg", "audio/m4a", "audio/aac",
               "audio/flac", "audio/opus", "audio/mp4"}
MAX_AUDIO_BYTES = 2_000_000          # about a minute of compressed speech
MAX_RECORD_S = 30


def settings() -> dict:
    return {"model": os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"),
            "tts_model": os.environ.get("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts"),
            "voice": os.environ.get("GEMINI_VOICE", "Kore"),
            "daily_limit": int(os.environ.get("GEMINI_DAILY_LIMIT", "500"))}


def enabled() -> bool:
    return gemini.api_key() is not None


# ----------------------------------------------------------------------------- prompts
_STYLE = {
    "hinglish": (
        "Reply in Hinglish: natural conversational Hindi mixed with English, written ONLY in Roman (Latin) script, "
        "the way college students in India explain things to a friend. Example tone: \"Dekho, β basically batata hai "
        "ki ek infected student roz kitne logon ko flu pass kar sakta hai.\" Keep technical terms and symbols in "
        "English (R₀, RK4, variance, gradient, eigenvalue). Never use Devanagari script."),
    "english": "Reply in clear, plain English.",
}
SPEECH_STYLE = {
    "hinglish": "Read this Hinglish (Hindi and English mixed, written in Roman script) the way a friendly teacher from "
                "India speaks: Hindi words with natural Hindi pronunciation, English terms in Indian English, warm and "
                "clear, at a relaxed pace.",
    "english": "A friendly teacher explaining clearly, warm and unhurried.",
}


def system_prompt(lang: str, context: str) -> str:
    return f"""You are the tutor built into ModeLab AI, an interactive site where a Flask server computes epidemic models,
3D shape analysis, stock-market forecasts and a Snake-playing AI live. A student is looking at one page and asks you
about it.

How to answer:
- {_STYLE[lang]}
- Ground every answer in the PAGE CONTEXT below. Use its numbers when they help, and say which setting or chart you
  mean. If the context does not cover the question, say so briefly and answer from general knowledge, marked as such.
- Teach: intuition first, then the maths if useful. Prefer a small concrete example over jargon.
- Keep it short: at most about 160 words unless the student asks for more detail. Short paragraphs. You may use
  "- " bullet lines and **bold**. No headings, tables, code blocks or LaTeX; write maths in plain Unicode (β, R₀, √n, Σ).
- Suggest one thing to try on the page (a slider or preset) when it fits.
- Markets pages are educational research, not investment advice; never give personal financial advice.
- If a question is unrelated to the site or to the maths behind it, answer in one sentence and steer back.

PAGE CONTEXT
{context}"""


_TAG = re.compile(r"<[^>]+>")


def _plain(s, limit: int = 400) -> str:
    s = _TAG.sub("", str(s or "")).strip()
    return s[:limit]


def _explainer_text(view: str | None, budget: int = 7000) -> str:
    if not view:
        return ""
    try:
        from .. import explain
    except Exception:
        return ""
    e = explain.EXPLAINERS.get(view)
    if not e:
        return ""
    parts = [f"Explainer: {e['title']}", _plain(e.get("summary"), 800)]
    for sec in e.get("sections", []):
        parts.append(f"## {sec['title']}")
        for b in sec["blocks"]:
            t = b.get("type")
            if t == "p":
                parts.append(_plain(b["text"], 700))
            elif t == "eq":
                parts.append(b["text"] + "  where " + "; ".join(f"{s} = {_plain(m, 120)}" for s, m in b.get("where", [])))
            elif t in ("steps", "list"):
                parts += [f"- {_plain(i, 300)}" for i in b["items"]]
            elif t == "callout":
                parts.append(f"Note ({b.get('title', '')}): {_plain(b['text'], 400)}")
            elif t == "figure":
                parts.append(f"[Figure: {b.get('title', b['figure'])}. {_plain(b.get('caption'), 200)}]")
    parts.append("Takeaways: " + " | ".join(_plain(t, 200) for t in e.get("takeaways", [])))
    text = "\n".join(p for p in parts if p)
    return text[:budget]


def build_context(view: str | None, page: dict) -> str:
    """The view's explainer plus what the page currently shows."""
    lines = [f"Page: {_plain(page.get('title'), 200) or 'ModeLab AI'} ({page.get('kind', 'page')}, route {view or '/'})"]
    if page.get("section"):
        lines.append(f"The student is reading the section: {_plain(page['section'], 200)}")
    params = page.get("params") or {}
    if params:
        lines.append("Current settings: " + ", ".join(f"{_plain(k, 40)} = {_plain(v, 40)}" for k, v in list(params.items())[:30]))
    tiles = page.get("kpis") or []
    if tiles:
        lines.append("Result tiles on screen (computed by the server):")
        lines += [f"- {_plain(t, 200)}" for t in tiles[:8]]
    ins = page.get("insights") or []
    if ins:
        lines.append("What-this-shows notes on screen:")
        lines += [f"- {_plain(t, 400)}" for t in ins[:6]]
    ex = _explainer_text(view)
    if ex:
        lines += ["", ex]
    return "\n".join(lines)


def suggestions(view: str | None, lang: str) -> list[str]:
    """Starter questions for a page, from its explainer."""
    out = []
    try:
        from .. import explain
        e = explain.EXPLAINERS.get(view or "")
    except Exception:
        e = None
    hing = lang == "hinglish"
    if e:
        out.append("Is page ka main idea simple words mein samjhao." if hing else "Explain the main idea of this page simply.")
        quizzes = [b["q"] for s in e["sections"] for b in s["blocks"] if b.get("type") == "quiz"]
        out += [_plain(q, 160) for q in quizzes[:2]]
        out.append("Abhi screen pe jo numbers hain unka matlab kya hai?" if hing else "What do the numbers on screen mean right now?")
    else:
        out += (["Yeh site kya karti hai?", "Kaunsa model pehle dekhun?"] if hing
                else ["What does this site do?", "Which model should I start with?"])
    return out[:4]


# ----------------------------------------------------------------------------- limits and caches
class Limiter:
    """Per-client sliding-window limits plus a daily cap for the whole server (protects the key's quota)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.hits: dict[tuple, collections.deque] = collections.defaultdict(collections.deque)
        self.day, self.used = time.strftime("%Y-%m-%d"), 0

    def check(self, client: str, kind: str, per_window: int = 20, window: float = 600) -> str | None:
        now = time.time()
        with self.lock:
            today = time.strftime("%Y-%m-%d")
            if today != self.day:
                self.day, self.used = today, 0
            if self.used >= settings()["daily_limit"]:
                return "The assistant has reached today's usage limit on this server. Please try again tomorrow."
            q = self.hits[(client, kind)]
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= per_window:
                return f"Too many requests. Please wait {int(window - (now - q[0])) + 1} seconds."
            q.append(now)
            self.used += 1
        return None

    def reset(self):
        with self.lock:
            self.hits.clear()
            self.used = 0


LIMITER = Limiter()
_TTS: "collections.OrderedDict[str, tuple[bytes, str]]" = collections.OrderedDict()
_TTS_LOCK = threading.Lock()


def speech(text: str, lang: str) -> tuple[bytes, str, bool]:
    """Spoken version of an answer (cached, so replaying it is free)."""
    s = settings()
    key = hashlib.sha256(f"{s['tts_model']}|{s['voice']}|{lang}|{text}".encode()).hexdigest()
    with _TTS_LOCK:
        if key in _TTS:
            _TTS.move_to_end(key)
            return (*_TTS[key], True)
    clean = re.sub(r"\*\*|`|^- ", "", text, flags=re.M)
    audio = gemini.speak(s["tts_model"], clean, s["voice"], SPEECH_STYLE[lang])
    with _TTS_LOCK:
        _TTS[key] = audio
        while len(_TTS) > 64:
            _TTS.popitem(last=False)
    return (*audio, False)


def transcribe(audio_b64: str, mime: str, lang: str) -> str:
    prompt = ("Transcribe this spoken question exactly. If it is Hindi or Hinglish, write it in Roman (Latin) script, "
              "not Devanagari. Output only the transcript." if lang == "hinglish" else
              "Transcribe this spoken question exactly, translating it to English if it is not English. Output only the transcript.")
    return gemini.generate(settings()["model"], [gemini.user(prompt, (audio_b64, mime))], max_tokens=300).strip().strip('"')


def answer(question: str, history: list[dict], view: str | None, page: dict, lang: str):
    steps = []
    for turn in history[-8:]:
        steps.append(gemini.user(turn["text"]) if turn["role"] == "user" else gemini.model_turn(turn["text"]))
    steps.append(gemini.user(question))
    return gemini.stream(settings()["model"], steps, system_prompt(lang, build_context(view, page)))
