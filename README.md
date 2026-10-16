# Modeling Lab

Epidemics, 3D shapes and stock markets, computed live on one numerical toolkit.

A Flask server solves every model when the page asks for it. Moving a slider sends a
request; the server runs an RK4 integration, a Monte Carlo simulation, an
eigen-decomposition, a least-squares fit or a walk-forward model fit, and returns
the arrays with the time it took. Nothing is precomputed. Expensive pure results
(an eigenbasis, a fitted model) are kept in an in-memory LRU cache keyed by their
inputs, and training too slow for one request (LSTM/GRU) runs as a background job
the page polls.

| Pillar | What it is | Computed live |
|---|---|---|
| **Spread** | Six MC312 models: drug dosing, Bass adoption, SIR on the 1978 boarding-school flu, rumours, radioactive decay, random walks | RK4 solves, least-squares fit of β and α, Monte Carlo runs |
| **Shape** | The ShapeLab geometry library: Fourier descriptors, Procrustes shape models, curvature, Laplace–Beltrami spectra, skinning | FFTs, Procrustes + PCA, sparse eigensolves, LBS/DQS deformation |
| **Markets** | Walk-forward forecasting of returns and volatility for six stocks, 1993–2026 | Yearly refits of ridge, gradient boosting, ARIMA, GARCH, HAR; LSTM/GRU as jobs |
| **Crossover** | The market data seen through the other two toolkits | Variance-ratio tests, VIX "dose" fits, PCA of 68 crash shapes |

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"          # add ".[rnn]" for the LSTM/GRU jobs (PyTorch)
.venv/bin/python -m modlab.server          # http://127.0.0.1:8000
```

Development server with reload: `.venv/bin/flask --app modlab.server run --debug --port 8000`.

Production (one process, many threads, because the job queue and caches live in memory):

```bash
pip install gunicorn
gunicorn -w 1 --threads 8 -b 0.0.0.0:8000 "modlab.server:create_app()"
```

With Docker (the image the live site runs):

```bash
docker build -t modeling-lab .
docker run -p 7860:7860 modeling-lab          # http://localhost:7860
```

## Deploy

The live app runs on a Hugging Face Space (Docker SDK, 2 vCPU / 16 GB). Every push to
`main` runs [CI](.github/workflows/ci.yml): the test suite, a Docker build, a smoke test
of the running container (including a real LSTM training job), and then
`deploy/push_space.py`, which uploads the app to the Space. The deploy step needs a
repository secret `HF_TOKEN` (a Hugging Face token with write access) and a repository
variable `HF_SPACE` (for example `yourname/modeling-lab`); without them it is skipped.

## Tests

```bash
.venv/bin/python -m pytest
```

The tests check the numerics against theory, not just status codes: RK4 converges at
fourth order, the Bass solver matches its closed form, the rumour peak matches the
conserved-quantity formula, random-walk moments match n(2p − 1) and 4np(1 − p), the
SIR fit recovers β ≈ 1.67 and α ≈ 0.44 on the 1978 data, Gauss–Bonnet holds on the
meshes, dual-quaternion skinning keeps a twisted cylinder round while linear blend
skinning collapses it, and the vectorised GARCH matches the reference loop. They
also cover validation errors, caching and concurrent first requests.

## API

All endpoints return JSON. Successful responses carry
`meta: {compute_ms, cached}`; errors are always
`{"error": {"code", "message"}}` with a 4xx/5xx status. `GET /api/routes` lists
every route, and the API tab in the app calls them.

| Area | Endpoints |
|---|---|
| Spread | `/api/spread/{dosing, bass, sir, rumour, decay, coins, walk, walk2d}`, `POST /api/spread/sir/fit` |
| Shape | `/api/shape/{outlines, fourier, jaw-model, jaw-sample, meshes, mesh/<name>, spectral, segment, curvature, surface-curvature, skin}` |
| Markets | `/api/markets/{tickers, prices, volatility, returns, stylized}`, jobs `rnn_returns`, `lstm_trap` |
| Crossover | `/api/crossover/{random-walk, walks, vix-events, vix-model, crash-shapes, crash-sample}` |
| Jobs | `POST /api/jobs {"kind": ..., ...}`, `GET /api/jobs/<id>` |

Example:

```bash
curl "http://127.0.0.1:8000/api/spread/sir?beta=1.67&alpha=0.44&lockdown=true&lock_start=3"
curl -X POST http://127.0.0.1:8000/api/jobs -H "Content-Type: application/json" \
     -d '{"kind": "lstm_trap", "ticker": "SPY", "epochs": 5}'
```

## Layout

```
src/modlab/
  core/        rk4, Monte Carlo helpers, PCA, least squares   (shared by every pillar)
  spread/      the six MC312 models
  crossover/   the crossover study, parameterised
  markets/     live walk-forward forecasting on top of stocklab
  shape_service.py   memoised calls into shapelab
  server/      Flask app: api_*.py blueprints, jobs, cache, templates/, static/
src/shapelab/  geometry library (IT575 coursework, extended)
src/stocklab/  forecasting study code (features, models, evaluation)
data/          prices (Yahoo Finance, frozen 13 Oct 2026), meshes, outlines, landmarks
tests/         pytest
deploy/        gunicorn config, push_space.py (Hugging Face upload)
Dockerfile     the production image (gunicorn, CPU-only PyTorch, uid 1000, port 7860)
.github/       CI: tests, Docker build + smoke test, deploy to the Space
web/, crossover/crossover.py   the earlier static demo and its build scripts
```

## Limits

- One process: the job queue and caches are in memory. Several workers would need
  Redis (or similar) for both.
- Market data is a frozen snapshot; there is no live price feed.
- The VIX "dose" model is first-order clearance toward a fixed median, mathematically
  a thresholded AR(1) process. The pharmacology is an analogy that fits, not a mechanism.
- Crash shapes include only drawdowns that recovered, and all six stocks are large
  companies that exist today.

*Educational research project. Not investment advice.*

Credits: ShapeLab began as IT575 Computational Shape Modeling coursework (Autumn
2024) with Varun Vyas. Jawline landmarks from the Helen dataset. Flu data: BMJ, 1978.
