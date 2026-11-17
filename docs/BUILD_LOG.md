# ModeLab AI: build log

A running record of how this product is being built. Every phase lists what was done and
every decision in the form **chosen / alternatives / why**. Newest phase at the bottom.
Entries written after the fact (phases 0–8) were reconstructed from the working sessions;
from phase 9 onward they are written as the work happens.

**Who decides what.** "You" is the product owner. Decisions marked *(owner)* were made by
you when asked; the rest were engineering calls made during the build, with the reasoning
recorded so you can overturn any of them.

---

## Phase 0: One project from three (2026-10-16)

**Goal:** combine three earlier projects (stock-price prediction, the MC312 mathematical
modelling labs, and the IT575 computational shape modelling library) into one product.

| Decision | Chosen | Alternatives considered | Why |
|---|---|---|---|
| How to combine *(owner)* | **Platform + crossover**: one shared numerical core, three sections, plus a fourth section that applies the epidemic and shape tools to market data | Three tabs side by side; a single merged narrative | Tabs would just be three projects in one window. A narrative would force connections. A shared core plus a real crossover study shows the same maths working across domains, which is the strongest portfolio story. |
| What to build first *(owner)* | **Demo first**: a static, precomputed web page | Start with the full backend | It made the story visible in one session and fixed the content before investing in infrastructure. |

**Built:** a static demo published as a claude.ai artifact.

---

## Phase 1: A live backend (2026-10-16)

**Goal (yours):** "a flask/django server … where things are not pre-calculated".

| Decision | Chosen | Alternatives | Why |
|---|---|---|---|
| Web framework | **Flask** (app factory + one blueprint per section) | Django; FastAPI | Django's ORM, admin and auth would sit unused; there is no database. FastAPI's async adds little here, because the work is CPU-bound NumPy and SciPy. Flask is the smallest thing that fits and is easy to read. |
| Compute model | **Compute on every request**, memoising pure results in an in-memory LRU cache keyed by inputs | Precompute grids of results; a Redis cache | Precomputing would break your "nothing pre-calculated" rule. Redis adds an extra service for a single-process app. |
| Duplicate requests | **In-flight de-duplication** in the cache (the second caller waits for the first) | Let both compute | Dragging a slider fires many identical requests, and de-duplication stops expensive eigen-decompositions from running twice. |
| Slow work (LSTM/GRU training) | **Background job manager** in a thread, polled by the page (`POST /api/jobs`, `GET /api/jobs/<id>`) | Celery + Redis; block the request | Celery needs a broker and a worker process. A blocked request times out and freezes the page. A thread pool keeps it to one deployable process. |
| API shape | **JSON everywhere**: errors always `{"error": {"code", "message"}}`, results carry `meta.compute_ms` and `meta.cached` | Ad hoc errors | A uniform shape lets one front-end engine handle every view. Showing the compute time proves results are live. |
| Numerics | One **shared core**: RK4, Monte Carlo helpers, PCA, least squares. GARCH and VIX clearance written as IIR filters (`scipy.signal.lfilter`) | Separate code per section; Python loops | Sharing the core is the point of the product. `lfilter` makes the recursive models about 100× faster than a Python loop, and tests check it against the reference loop. |

---

## Phase 2: Docker and deployment (2026-10-16)

| Decision | Chosen | Alternatives | Why |
|---|---|---|---|
| Host *(owner)* | **Hugging Face Spaces** (Docker SDK) | Render, Fly.io, Railway | Free CPU tier with enough RAM for PyTorch, and it is the natural home for an ML portfolio. |
| Repository *(owner)* | **Public GitHub repo** `Dr-Tee-01/modeling-lab` | Private repo | Recruiters can read the code. |
| Server process | **gunicorn, 1 worker × 8 threads** | Several workers | The job queue and caches live in memory, so several workers would each keep their own copy and jobs would get lost between them. Threads share memory, and NumPy releases the GIL. |
| Image | **CPU-only PyTorch**, non-root user (uid 1000), port 7860 | Full CUDA PyTorch | CUDA wheels are several gigabytes and Spaces' free tier has no GPU. HF Spaces expects uid 1000 and port 7860. |
| Pipeline | **GitHub Actions**: tests → build image → smoke-test the running container (including a real LSTM job) → view harness → deploy via the `huggingface_hub` API | Manual deploys | Every push to `main` is checked end to end before it reaches the Space. |
| Secrets | **You set `HF_TOKEN` and `HF_SPACE` yourself** | I handle the token | Credentials never pass through the assistant. |

---

## Phase 3: A consistent, modern interface (2026-11-14)

**Goal (yours):** "make the UI very consistent and modern, interactive and illustrative".

| Decision | Chosen | Alternatives | Why |
|---|---|---|---|
| Front-end stack | **Vanilla ES modules**, a hand-written SVG plotting kit (`plot.js`), three.js only for 3D | React/Vue with Chart.js, Plotly or D3 | There is no build step, nothing to install, and it loads fast. The charts needed are simple, and owning them gives full control of the look. three.js is the standard for mesh viewing. |
| Consistency mechanism | **One workbench engine** (`ui/workbench.js`) that renders every view from a declarative **spec** | Hand-written pages per view | Every one of the 21 views has the same layout: header, live illustration, result tiles, controls, charts, then "What this shows" and "The maths". A new view gets this layout for free. |
| Routing and sharing | **Hash routes holding the full state** (`#/spread/sir?beta=1.66…`), so "Copy link" reproduces a setup | Client-side router library; no shareable state | No server routing is needed, and links are exact. |
| Navigation | Sidebar plus a **⌘K command palette** | Top tabs only | 21 views are too many for tabs, and a palette is fast for power users. |
| Illustrations | A tiny **diagram kit** (`illustrate.js`: nodes, edges, labels) where arrow widths can carry a rate | Static images | Diagrams redraw as sliders move, so they explain the mechanism while you play. |
| Testing the UI | A **Node harness** (`tools/check_views.mjs`) that runs every spec's requests, tiles and insights against a live server, plus browser sweeps | Browser tests only | It is fast and runs in CI without a browser. |
| Delivery | I built the engine and the Spread views as the reference, then **parallel agents ported Shape, Markets and Crossover**, and I reviewed each in the browser | One person doing everything in sequence | It was faster, and the reference format kept the agents consistent. |

---

## Phase 4: The Snake project as a fifth section (2026-11-14)

**Goal (yours):** integrate `Snake-Game-main.zip` (pygame + PyTorch DQN).

| Decision | Chosen | Alternatives | Why |
|---|---|---|---|
| Running the game | **Headless port** on the server (`agents/snake.py`), with the same board, rules, rewards and 11-feature state as the original | Ship pygame; rewrite in JS | pygame needs a window, and a JS rewrite would separate the AI from the game. A headless port lets training, replays and comparisons run on the server. |
| Fidelity details | Kept the original's quirks: **the tail cell is deadly** (collision is checked before the tail moves), and the step limit is 100 × length | Make the rules "cleaner" | A faithful port lets results be compared with the original project. |
| Agents | **Random, greedy, BFS (with capped flood fill), DQN** | DQN only | Baselines make the DQN's performance meaningful. BFS shows what planning buys. |
| DQN training | Same algorithm and hyperparameters, with **replay targets batched** into one pass | The original per-sample loop | Same maths, about 10× faster, so 200 games take around 15 s. |
| Long work | Training and comparisons run as **background jobs** | Synchronous requests | A request would time out on a full comparison. |
| Untrusted download | The zip was **read only, never executed** | Run its code | Downloaded archives are treated as untrusted. |

---

## Phase 5: Darker, bluer theme (2026-11-15)

| Decision | Chosen | Alternatives | Why |
|---|---|---|---|
| Palette | **Deep navy** surfaces (`#050A16` background) with blue/violet accents; light theme made bluer too | Only tweak the accent | You asked for both darker and bluer. Changing the surface tokens shifts the whole interface at once. |
| Default theme | **Dark by default**, with the choice remembered | Follow the operating system | The darker look is the product's identity. Light mode remains one click away. |

---

## Phase 6: Every visible number comes from the backend (2026-11-15)

**Goal (yours):** "ensure every data visible comes from the backend and nothing is stored in the front end".

| Decision | Chosen | Alternatives | Why |
|---|---|---|---|
| Reference data (tickers, meshes, model and agent names, dataset facts) | **`GET /api/catalog`**, built from per-section providers registered with `@section`, and loaded once at start-up by `catalog.js` | Embed it in the HTML template; one endpoint per list | One request at start-up, and each section owns its own facts. Embedding in the template would put data in the front-end bundle. |
| Derived numbers (tiles, "best model", averages, shares) | Each response carries a **`summary`** computed on the server | Keep JS reductions | Moves computation, not just storage, to the backend, so the front end only formats and draws. |
| Snake "Play" | **Server-side game sessions** (`POST /api/agents/snake/play`, then `/step` per move) | Game loop in the browser | The game logic was the largest piece of front-end computation. The cost is one small request per move, which is fine at 4–20 moves per second. |
| Best score | **Server record** | `localStorage` | Browser storage is front-end state. |
| What stays in the browser | Formatting, layout, colours, UI wording, control definitions, preset values | — | That is presentation, not data. |
| Delivery | **Three agents** (Spread, Shape, Markets + Crossover) worked in parallel under a written rule; I did Snake, the catalog infrastructure and a final audit by grep | Sequential work | Parallel work on separate files had no conflicts. Each agent added tests for its new fields. |

---

## Phase 7: Gradient, glass and motion (2026-11-15)

**Goal (yours):** "more modern, gradient-like, interactive, illustrative, sophisticated".

| Decision | Chosen | Alternatives | Why |
|---|---|---|---|
| Implementation | **A CSS "visual layer"** on top of the existing tokens, plus small engine hooks | Tailwind; an animation library such as GSAP or Framer | No new dependencies or build step. Every view upgraded at once because they share one engine. |
| Background | Slow **aurora** of radial gradients plus a dot grid, animated only with `transform` | Video or canvas background | Animating `transform` is cheap and keeps the GPU cost low. |
| Cards | **Glass** (backdrop blur) with a gradient hairline and a **pointer spotlight** | Flat cards | Adds depth and makes the page respond to the pointer. One listener throttled with `requestAnimationFrame` serves the whole app. |
| Result tiles | **Count-up animation** that keeps the server's number formatting, plus a flash on change | Instant swap | You can see which numbers changed after moving a slider. |
| Charts | Lines **draw in** on first render, glow in their series colour, and get a gradient fill when there are only one or two main lines | Fill everything | Fills on busy charts become muddy, so they are capped at two lines. |
| Diagrams | Gradient nodes, a glow filter, and **particles flowing along arrows** (SVG `animateMotion`), faster and denser on thicker edges | JS particle system | Native SVG animation needs no per-frame JS and maps the flow rate directly to motion. |
| Accessibility | Every animation is turned off under **`prefers-reduced-motion`** | — | Motion should never be forced on anyone. |

---

## Phase 8: Illustrated explanation pages (2026-11-15)

**Goal (yours):** a "See explanation" button on every model that opens a detailed, illustrated explanation.

| Decision | Chosen | Alternatives | Why |
|---|---|---|---|
| Where content lives | **Server-side Python modules** (`src/modlab/explain/<section>.py`) served by `/api/explain/...` | Markdown files rendered in the browser; a CMS | This keeps your backend-only rule. Prose can compute its own numbers (for example, the fitted β is inserted from the fit, not typed), so text cannot drift from the models. |
| Content format | **Typed blocks**: `p`, `eq`, `steps`, `cards`, `compare`, `callout`, `quiz`, `try`, `figure` | Free-form HTML | One renderer styles every page the same way, and a test can validate structure. |
| Figures | Each figure is **computed by the server** from the real model code and returned as data (`lines`, `bars`, `scatter`, `paths`, `flow`, `grid`, `stats`). Sliders refetch `/api/explain/figure/<id>` | Static images; recompute in JS | Figures are real output and stay correct when models change. Recomputing in JS would break the backend-only rule. |
| Notes under figures | **Server-written sentences** computed from the figure's numbers | Hand-written captions only | The text always matches the chart, at every slider position. |
| Safety of server HTML | The renderer keeps **only a whitelist of inline tags** and strips attributes | Trust `innerHTML` | Defence in depth, even though the content is first-party. |
| Reading experience | Contents list that tracks your scroll, reading-progress bar, reveal-on-scroll, quizzes with explanations, "try it" links that open the live model with matching settings | Plain long page | Long technical pages need navigation and active recall to be usable. |
| Quality gate | **One generic test** (`tests/test_explain.py`) for all pages: structure is valid, every figure computes, and every slider works at both ends and every choice option | Per-page tests | Consistent checks however many pages there are. |
| Delivery | I built the system and the SIR reference page; **five agents wrote the other 20 pages in parallel**, one per section | Write all 21 myself | Each section needs domain depth, and parallel writers on separate files do not conflict. |
| Format gaps the agents reported | Added **choice controls** (segmented), **reference lines** (`hlines`), **date axes** (`xdate`), **hidden series** (`legend: false`, `hover: false`) and **explicit legends** | Leave the workarounds | Several agents independently hit the same gaps, a sign the format was missing them. |

---

## Phase 9: Gemini assistant with Hinglish voice (in progress, 2026-11-15)

**Goal (yours):** "Gemini API integration to ask live questions and Hinglish audio".

| Decision | Chosen | Alternatives | Why |
|---|---|---|---|
| Voice mode *(owner)* | **Push-to-talk plus spoken answers**: type or hold the mic; answers appear as text and can be read aloud | Spoken answers only; full live voice conversation (Gemini Live API) | It works on any host including HF Spaces, needs no WebSocket relay, and costs far less than a continuous live session. |
| Answer language *(owner)* | **A toggle, Hinglish by default** (Roman script), with English as the alternative; audio follows the toggle | Always Hinglish; English text with Hinglish audio | You get both, and the default suits the audience. |
| Where the Gemini call happens | **On the Flask server**; the browser talks only to `/api/assistant/...` | Call Gemini from the browser | The key would be exposed to every visitor. It also keeps your backend-only rule. |
| API | **Gemini Interactions API** (`/v1beta/interactions`, revision 2026-05-20), checked against Google's current docs before writing code | Older `generateContent` endpoint | It is what Google's docs currently document for text, streaming, audio input and speech output. |
| Client library | **Python standard library (`urllib`)**, about 150 lines | `google-genai` SDK; `requests` | No new dependency in the Docker image. The REST surface needed is small (one endpoint), and a thin client is easy to test with a fake transport. |
| Models | Defaults **`gemini-3.8-flash`** (answers and transcription) and **`gemini-3.8-flash-tts`** (speech), overridable with `GEMINI_MODEL` / `GEMINI_TTS_MODEL` | Gemini 4 models; older 2.5 models | The 3.8 flash models are listed as current and fast. Gemini 4 status was unlabelled in the docs, and 2.5 Flash is deprecated. Environment overrides let you switch without a code change. |
| Grounding | Every question carries **the page's explainer text plus what is on screen** (settings, tiles, insights) | General chat with no context | Answers are about *this* model and *these* numbers, not generic. |
| Hinglish | Prompted as **conversational Hindi + English in Roman script**, with technical terms kept in English; speech uses a style instruction for natural Hindi pronunciation | Translate to Devanagari; a separate translation model | Google lists Hindi but not Hinglish as a TTS language, so behaviour is steered by prompt and style. **Untested until a key is added**; if pronunciation is poor, the next step is to transliterate to Devanagari before speaking. |
| Voice questions | **Two calls**: transcribe the recording, then answer the transcript | One call that hears audio and answers | Showing you exactly what was heard builds trust and makes mistakes visible. It costs one extra short call. |
| Streaming | **Server-sent events** from Flask to the page | WebSockets; wait for the full answer | Text appears as it is written. SSE is one-directional, which is all this needs, and it works through gunicorn threads and HF's proxy. |
| Protecting your quota | **Per-visitor rate limits** and a **daily cap** (`GEMINI_DAILY_LIMIT`, default 500 calls); spoken answers are **cached** so replays are free | No limits | On a public Space, anyone could otherwise spend your key's quota. |
| Key handling | You put the key in a **`.env` file** (ignored by git and Docker) locally, and in a **Space secret** when deployed | Commit it; paste it into chat | The key never enters the repository, the image or this conversation. |
| Chat history | Kept **in memory in the open panel only**, not saved | Save it in browser storage | Consistent with storing nothing in the front end. |
| Page context | The shell gathers the page's **live settings, result tiles, notes and current explainer section**, and the server adds the full explainer text | Send only the question | Answers can cite "your β = 1.66 gives R₀ = 3.32". Everything sent was rendered from server data in the first place. |
| Starter questions | Served by **`/api/assistant/suggestions`**, built from each explainer's quiz questions | Hard-coded in the panel | Keeps them specific to each page, and they come from the server. |
| Answer formatting | A tiny safe formatter (**bold**, `code`, "- " bullets, paragraphs) built from DOM nodes | A Markdown library with `innerHTML` | Model output is untrusted text, so it never becomes HTML. No dependency. |
| Push-to-talk UX | **Hold the mic to talk, or click to start and stop**, with a live level meter and a 30 s cap; voice questions **auto-play** the spoken answer | Click-only recording | Holding feels natural on desktop, and tapping works on phones. Auto-play makes the voice path hands-free. |
| Testing without a key | A **fake Gemini transport** in pytest (10 tests: streaming, transcription, fallbacks, errors, validation, rate limits, WAV wrapping, caching) and a faked network in the browser to check the panel | Wait for a real key | Everything except Google's actual responses is verified now. |

**Status (2026-11-15):** built and tested against a fake Gemini; the panel works on every page, and with no key it explains how to switch on. **Not yet verified with the real API:** that needs your key in `.env`. Two things to check then: the exact streaming event names, and Hinglish pronunciation. Microphone capture also needs a real browser with mic permission, which the built-in test browser doesn't have.

---

## Phase 10: Build log and a new name (2026-11-15)

**Goals (yours):** keep a build log of every decision; rename the product to include modelling, AI and lab.

| Decision | Chosen | Alternatives | Why |
|---|---|---|---|
| Build log | **`docs/BUILD_LOG.md` in the repo**, one phase per request, decisions as *chosen / alternatives / why*, owner decisions marked | A chat summary; a separate document outside the repo | It travels with the code, recruiters can read it on GitHub, and it is updated as work happens. |
| Product name *(owner)* | **ModeLab AI** | AI Modeling Lab, ModeLAIb, Modeling × AI Lab, LiveModel AI Lab, ModAI Lab, Model Intelligence Lab. "AIM Lab" was ruled out because it clashes with the existing game Aim Lab | "Model" and "Lab" share the L, which makes it short and brandable, and "AI" is explicit. |
| Wordmark | **"ModeLab" in gradient text plus a small gradient "AI" badge** | Plain text "ModeLab AI" | Reads as a logo, and the badge carries the AI part without lengthening the name. |
| What was renamed | Every user-facing name (page titles, sidebar and top bar, meta description, home page, the old static demo in `web/`, assistant prompts, README, Dockerfile, the Space card title in `deploy/push_space.py`), the **distribution name** (`modelab-ai` in `pyproject.toml`) and the **Docker image tag** (`modelab-ai` in README and CI) | — | "Rename everything" covers anything a person can see or type. |
| Kept as is | The **Python package `modlab`** and the **GitHub repo path** `Dr-Tee-01/modeling-lab` | Rename `modlab` to `modelab`; rename the repo now | The package name is internal, already reads as "mod-lab", and renaming it would touch every import, test and deploy script for no visible gain. Renaming the GitHub repo or the Hugging Face Space changes public URLs, so that waits for your explicit go-ahead. |

---

## Phase 11: Four labs (2026-11-15)

**Goal (yours):** regroup the models under named labs such as "Reinforcement Learning Lab", "Modeling Lab" and "Time Series Lab".

| Decision | Chosen | Alternatives | Why |
|---|---|---|---|
| The labs *(owner)* | **Modeling Lab** (6 MC312 models), **Geometry Lab** (5 shape models), **Time Series Lab** (volatility, returns, neural nets + the 3 market "crossover" studies), **Reinforcement Learning Lab** (4 Snake pages) | Shapes merged into Modeling Lab; a separate Crossover lab | You chose a separate Geometry Lab and folding Crossover into Time Series, because all three crossover studies analyse market time series. The sidebar balances at 6 / 5 / 6 / 4. |
| Where the grouping lives | **On the server** (`src/modlab/labs.py`), published in `/api/catalog` with each lab's name, order, colour and a description computed from live data | Hard-code the groups in the front end | Keeps the backend-only rule, and one list drives the sidebar, home page, breadcrumbs and explainer prev/next links. |
| Routes and keys | **Unchanged** (`#/spread/sir`, `#/crossover/dose`, explainer keys like `markets/neural`), plus a new lab shortcut such as `#/timeseries` | Re-key everything as `#/timeseries/dose` | Old links keep working, and the API, explainers and tests need no churn. The lab is where a model is shown, not what it is called internally. |
| Explainer navigation | Previous/next now **follows the lab**, so Neural nets → Is it a random walk? | Within the old section only | Reading order matches the sidebar. |
| Short names | A `short` name per lab ("RL Lab") for tight spaces such as the home diagram | Shrink the font | "Reinforcement Learning Lab" did not fit its box. |
| Visible text | Explainer kickers, breadcrumbs, the home page and README now say the lab names | — | No page mentions the old Spread/Shape/Markets/Crossover/Agents labels. |
| Guard | `tests/test_labs.py`: every model in exactly one lab, every lab model has an explainer, catalog order, cross-lab explainer links | — | Adding a model without placing it in a lab now fails the tests. |

**Also finished in this phase (earlier requests):**
- Grid figures gained axis titles and tick labels (`xlabel`, `ylabel`, `xticks`, `yticks`), as the explainer helpers asked. The drug-dosing map and the monkey-saddle curvature map use them, and their captions and notes no longer depend on colour names.
- The Shape explainers' last pass added mesh and outline pickers and colour legends, and cut start-up cost from about 0.8 s to 0.25 s.
- Explanation sections now also reveal on scroll if the browser delays its visibility check (seen in background tabs).
- **Checks:** 190 tests pass, all 21 views and their explainers pass the page check, and the browser shows no errors.

---

## Phase 12: App icon (2026-11-15)

**Goal (yours):** give the app an icon; the Chrome tab showed none.

| Decision | Chosen | Alternatives | Why |
|---|---|---|---|
| The mark | A **rounded tile in the brand gradient** (blue → violet → cyan) with a **white S-shaped growth curve ending in a data point** and a small **four-point spark** for AI | A letter "M"; the old three-colour glyph | The curve is the shape the models actually produce (epidemic, adoption, learning curves), and the spark marks the AI. Few, bold shapes stay legible at 16 px. |
| Formats | **SVG** (sharp at any size, used by Chrome), **PNG 16/32** (fallbacks), **apple-touch-icon 180**, **192/512 PNG + web manifest** (installable app, Android), **favicon.ico** with embedded PNGs, also served at `/favicon.ico` | SVG only | Browsers ask for different files: Safari and iOS need PNG, some clients request `/favicon.ico` directly, and the manifest lets the site install as an app. |
| How they are made | **`tools/make_icons.py`**: one geometry, rendered as SVG and rasterised with NumPy (supersampled shapes), with a hand-written PNG/ICO encoder | Pillow; ImageMagick; a design tool | No new dependency, and every size is generated from the same source, so changing the mark means one script run. |
| Brand consistency | The **sidebar and mobile top bar** now use the same mark next to "ModeLab **AI**" | Keep the old glyph | The tab, the app header and an installed app icon all match. |
| Guard | A test checks `/favicon.ico` is a valid ICO and that every linked icon file is served | — | Catches a broken link or missing file. |

---

## Phase 13: Big Data Lab from the IE494 course slides (2026-11-17)

**Goal (yours):** add sections from the Big Data Processing course archive (18 slide decks), as the standard hands-on labs for those lessons, simulating or implementing whatever can be.

| Decision | Chosen | Alternatives | Why |
|---|---|---|---|
| Handling the archive | Unzipped into an **empty scratch folder** and treated as untrusted: **read only, never executed**; no PDF tool was installed for it | Install a PDF library; open files in place | Downloaded archives are untrusted, and the built-in PDF reader was enough. |
| Reading 18 decks | **Three parallel readers**, grouped by topic, each returning algorithms, exact worked datasets, simulation ideas and **slide errors** | Read everything in the main session | Keeps the main context free for building, and checking the decks against each other surfaced many slide errors (e.g. swapped iris labels, an average combiner that isn't valid, wrong stage claims). |
| Where it goes | A fifth lab, **Big Data Lab**, with **12 views in lesson order**: HDFS, MapReduce engine, MapReduce joins, RPC, RDD playground, DAG & stages, Spark join strategies, Spark SQL plans, Tables & file layout, OLAP operations, Cube SQL operators, Cube lattice | Fold topics into existing labs | It is a separate course with its own sequence; one lab keeps the lessons together. |
| Lab style *(owner)* | Each view mirrors the **standard lab exercise** for its lesson: a task, controls, the intermediate data at every stage (per node, partition or reducer), data movement in records and bytes, and the expected output to check | Concept demos only | You asked for the labs typically paired with these lessons. |
| Data | **The slides' own datasets, copied exactly** into `src/modlab/bigdata/datasets.py` and checked against slide totals (30 employees over 3 nodes, WordCount splits, customer/orders joins, Han & Kamber sales cube, 30-row fact table totalling 3350, enrolment table totalling 863) | Synthetic data | Students can match every number to the lecture; synthetic data is used only where the slides give none, and is labelled. |
| Engine | **Plain Python/pandas re-implementations** of HDFS, MapReduce, Spark RDDs, stages, join planning, Catalyst-style rules and cube operators, labelled as simulations | Run real Hadoop/Spark | Real clusters can't run per request on a single Space and hide the intermediate steps that a lab needs to show. |
| Slide errors | Implement the **correct** behaviour and list the slide's error on the explanation page | Reproduce the slides verbatim | A teaching tool must not teach mistakes, but students should see where their slides differ. |
| Beyond the slides | Topics the decks don't cover but the labs need (Spark planner rules, AQE/skew, the 10 MB broadcast threshold, HRU view selection, GROUPING SETS) are included and **labelled "beyond the slides"** | Leave them out | They complete the lab; the label keeps it honest. |
| Parallel build | **Four builders**, each owning separate backend, route, view, stylesheet, explainer and test files; shared wiring auto-loads them (`api_bigdata.py` imports `bigdata_routes_*`, the explainer package imports `bigdata_*`, `views/bigdata.js` gathers topic files) | One builder in sequence | No two builders touch the same file, so work runs in parallel without conflicts. |

**Implementation choices made while building (by area):**

| Area | Decision | Why |
|---|---|---|
| HDFS | Hadoop's **rack-aware placement** (first replica local, second and third on one other rack) with a "next nodes" policy to compare; the scale-up of joins is a **cost model**, not a real run at scale | Shows why racks matter when a whole rack fails; a real 10 GB join can't run per request. |
| MapReduce | A real engine that keeps every stage's data; the **range partitioner** uses sampled split points for numbers and letter ranges for words (matching the slide's A–I/J–Q/R–Z); the hash partitioner uses Java's `hashCode` | Students can match each reducer's input to the slides. |
| MR joins | The customers map task runs third so the customer record arrives mid-list, **exactly as slide p12 shows**; inner/left/right/full joins selectable | Reproduces the slide's arrival order and exposes its accidental full-outer join. |
| RPC | Seeded event simulation; three retry policies on the **same seed** | Differences come only from the policy, not the dice. |
| RDD / DAG | Stage cutting mirrors Spark's DAGScheduler; **shuffle files are kept by default** (as in Spark), with a switch; repartition deals records round-robin from each partition's index (Spark starts randomly); Spark's first iteration assumed equal to one MapReduce pass | Faithful where it matters, deterministic where Spark is random. Assumptions are stated on the page. |
| Spark SQL | Join selection follows **Spark 3's planner** checks in order; partition/bucket ids use a **Murmur3 matching Spark's own `hash()`** (checked against Spark's documented values) | The partition and bucket numbers shown are the ones real Spark would compute. |
| Tables | A **seeded synthetic employee table** shaped like the course's (the slides show only 3 rows), labelled as synthetic | The deck has too few rows to show pruning or skipping. |
| OLAP / cube | Slice and dice members are chosen by **clicking table headers** (hidden state); results are **checked against the slides' own numbers** on the page; the 4th and 5th lattice dimensions are a **seeded synthetic extension** | Clicking beats 30 switches; the self-check builds trust. Limitation: "Copy link" doesn't capture clicked selections yet. |

**Result:** 12 views and 12 explainers (6–8 sections, 5–8 server-computed figures each), all reproducing the slides' worked numbers. About 30 slide errors are corrected and listed on the explanation pages (e.g. L1's read-time formula, the MAX slide that sums, the full outer join presented as an inner join, coalesce called "wide", the "100× faster" claim). **Checks:** 329 tests pass; all 33 views and explainers pass the page check; a browser sweep of all 24 Big Data pages showed every chart and figure drawn and no console errors.

**Requested by the builders, not yet done:** a `table` figure kind for explanation pages, a `select` (dropdown) control for long option lists, dependent slider ranges, and drawing checks in `tools/check_views.mjs`.

---

## How the work is verified (all phases)

- **pytest**: numerics checked against theory (RK4 order, closed forms, conservation laws), API contracts, and every explainer figure and slider.
- **View harness** (`node tools/check_views.mjs …`): runs every view's requests, tiles, insights, presets and explainer against a live server.
- **Browser sweeps** in the built-in browser: every route opened, console checked for errors, page text checked for `undefined` or `NaN`, plus screenshots of key pages.
- Nothing is pushed or deployed without your explicit go-ahead.
