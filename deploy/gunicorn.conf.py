"""gunicorn settings for the container. One worker process (in-memory job queue and
caches), several threads for concurrent requests, and a timeout long enough for the
slowest synchronous endpoint (pooled walk-forward returns, about 10 s)."""
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '7860')}"
workers = 1
threads = int(os.environ.get("GUNICORN_THREADS", 8))
worker_class = "gthread"
timeout = 120
graceful_timeout = 30
accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info")
