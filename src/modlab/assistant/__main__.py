"""Check the Gemini setup with two tiny real calls:  python -m modlab.assistant

Reads GEMINI_API_KEY from the environment or the project's .env. Prints a one-line Hinglish
answer (streamed) and writes a short spoken sample to assistant-check.wav. Uses about two
small requests of your quota.
"""
import sys
import time

from modlab.server import load_env

load_env()

from . import gemini, service as A  # noqa: E402

if not A.enabled():
    sys.exit("GEMINI_API_KEY is not set. Put it in the project's .env file (see .env.example).")
s = A.settings()
print(f"model {s['model']} · speech {s['tts_model']} · voice {s['voice']}")
try:
    t0 = time.perf_counter()
    print("answer: ", end="", flush=True)
    text = ""
    for chunk in A.answer("R₀ kya hota hai? Ek line mein batao.", [], "spread/sir", {"title": "SIR check"}, "hinglish"):
        text += chunk
        print(chunk, end="", flush=True)
    print(f"\n  ({time.perf_counter() - t0:.1f} s, streamed)")
    audio, mime, _ = A.speech(text or "Namaste! ModeLab AI ka tutor ready hai.", "hinglish")
    with open("assistant-check.wav", "wb") as f:
        f.write(audio)
    print(f"speech: {len(audio) / 1024:.0f} KB {mime} written to assistant-check.wav. Play it to check the Hinglish voice.")
except gemini.GeminiError as e:
    sys.exit(f"Gemini error ({e.code}): {e.message}")
