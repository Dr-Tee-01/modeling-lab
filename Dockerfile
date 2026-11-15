# ModeLab AI: Flask + gunicorn, computing every result on request.
# Runs as uid 1000 on port 7860 (what Hugging Face Spaces expects); any host can override PORT.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860 \
    MODLAB_MARKETS_WORKERS=2 \
    OMP_NUM_THREADS=1 \
    MPLCONFIGDIR=/tmp

RUN useradd --create-home --uid 1000 app
WORKDIR /app

# Dependencies first so code changes don't reinstall them.
COPY requirements.txt .
RUN pip install -r requirements.txt \
 && pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

COPY --chown=app:app pyproject.toml README.md ./
COPY --chown=app:app src ./src
COPY --chown=app:app data ./data
COPY --chown=app:app deploy/gunicorn.conf.py ./deploy/
RUN pip install --no-deps -e .

USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health', timeout=4)"

# One process: the job queue and caches live in memory. Threads serve concurrent requests.
CMD ["gunicorn", "-c", "deploy/gunicorn.conf.py", "modlab.server:create_app()"]
