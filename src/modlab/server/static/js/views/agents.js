// Pillar 5 · Agents: the Snake-Game project. The server runs the environment, the agents
// and the DQN training; the browser draws the board and runs the trained network for hints.
import { call, fmt, fmtMs, int, pct } from '../api.js';
import { E, extent, frame, hover, nearest, zip } from '../plot.js';
import { h } from '../ui/dom.js';
import { S, edge, node, text, widthFor } from '../ui/illustrate.js';

const W = 32, H = 24;
const DIRS = [[1, 0], [0, 1], [-1, 0], [0, -1]];               // right, down, left, up
const ACTIONS = ['straight', 'right', 'left'];
const FEATURES = ['danger straight', 'danger right', 'danger left', 'moving left', 'moving right', 'moving up', 'moving down', 'food left', 'food right', 'food up', 'food down'];
const AGENT = {
  random: { label: 'Random walker', color: 'neutral', what: 'Picks straight, right or left at random: Lab 10\'s random walk with a body.' },
  greedy: { label: 'Greedy', color: 's3', what: 'Steps toward the food, skipping moves that would kill it on the next step.' },
  bfs: { label: 'BFS path-finder', color: 's1', what: 'Searches the shortest free path to the food; if none is safe, takes the move with the most room.' },
  dqn: { label: 'Deep Q-network', color: 's2', what: 'Your trained 11 → 256 → 3 network: it scores each move from 11 yes/no features and takes the best.' },
};
const CAUSE = { wall: 'hit a wall', self: 'hit itself', timeout: 'ran out of moves', limit: 'reached the step cap', won: 'filled the board' };
const tip = (title, rows) => `<b>${title}</b>` + rows.map(([k, v]) => `<br>${k}: <span class="n">${v}</span>`).join('');
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/* ---------------------------------------------------------------- shared game logic (same rules as the server) */
function newGame(rng = Math.random) {
  const hx = W >> 1, hy = H >> 1;
  const g = { d: 0, snake: [[hx, hy], [hx - 1, hy], [hx - 2, hy]], score: 0, steps: 0, over: null };
  g.body = new Set(g.snake.map(k));
  placeFood(g, rng);
  return g;
}
const k = p => p[0] + ',' + p[1];
function placeFood(g, rng = Math.random) { do { g.food = [Math.floor(rng() * W), Math.floor(rng() * H)]; } while (g.body.has(k(g.food))); }
const turn = (d, a) => (d + [0, 1, -1][a] + 4) % 4;
const ahead = (g, a) => { const [dx, dy] = DIRS[turn(g.d, a)]; return [g.snake[0][0] + dx, g.snake[0][1] + dy]; };
const hits = (g, p) => p[0] < 0 || p[0] >= W || p[1] < 0 || p[1] >= H || g.body.has(k(p));
function features(g) {
  const [hx, hy] = g.snake[0], [fx, fy] = g.food, d = g.d;
  return [hits(g, ahead(g, 0)), hits(g, ahead(g, 1)), hits(g, ahead(g, 2)), d === 2, d === 0, d === 3, d === 1, fx < hx, fx > hx, fy < hy, fy > hy].map(Number);
}
function step(g, a, rng = Math.random) {
  g.steps++; g.d = turn(g.d, a);
  const nh = ahead(g, 0);
  if (nh[0] < 0 || nh[0] >= W || nh[1] < 0 || nh[1] >= H) return (g.over = 'wall');
  if (g.body.has(k(nh))) return (g.over = 'self');
  g.snake.unshift(nh); g.body.add(k(nh));
  if (g.steps > 100 * g.snake.length) return (g.over = 'timeout');
  if (nh[0] === g.food[0] && nh[1] === g.food[1]) { g.score++; placeFood(g, rng); return 'ate'; }
  g.body.delete(k(g.snake.pop()));
  return null;
}
function qvalues(w, f) {
  const hdn = w.w1.map((row, i) => Math.max(0, row.reduce((s, v, j) => s + v * f[j], w.b1[i])));
  return w.w2.map((row, i) => row.reduce((s, v, j) => s + v * hdn[j], w.b2[i]));
}

/* ---------------------------------------------------------------- board drawing */
function drawBoard(cv, snake, food, { hintDir = null, dead = false, overlay = '' } = {}) {
  const box = cv.parentElement.getBoundingClientRect();
  const cell = Math.max(6, Math.floor(Math.min(box.width / W, 560 / H)));
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  if (cv.width !== W * cell * dpr) { cv.width = W * cell * dpr; cv.height = H * cell * dpr; cv.style.width = W * cell + 'px'; cv.style.height = H * cell + 'px'; }
  const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.fillStyle = css('--sunk'); c.fillRect(0, 0, W * cell, H * cell);
  c.fillStyle = css('--line'); for (let x = 0; x < W; x++) for (let y = 0; y < H; y++) if ((x + y) % 2) c.fillRect(x * cell, y * cell, cell, cell);
  if (food) { c.fillStyle = css('--s2'); c.beginPath(); c.arc((food[0] + 0.5) * cell, (food[1] + 0.5) * cell, cell * 0.38, 0, 7); c.fill(); }
  const body = dead ? css('--bad') : css('--s1');
  snake.forEach((p, i) => {
    c.fillStyle = body; c.globalAlpha = i === 0 ? 1 : 0.55 + 0.45 * (1 - i / snake.length);
    const r = cell * 0.22; const x = p[0] * cell + 1, y = p[1] * cell + 1, s = cell - 2;
    c.beginPath(); c.roundRect ? c.roundRect(x, y, s, s, r) : c.rect(x, y, s, s); c.fill();
  });
  c.globalAlpha = 1;
  if (hintDir != null && snake.length) {
    const [hx, hy] = snake[0], [dx, dy] = DIRS[hintDir];
    c.strokeStyle = css('--s3'); c.lineWidth = 3; c.lineCap = 'round';
    const x0 = (hx + 0.5) * cell, y0 = (hy + 0.5) * cell, x1 = x0 + dx * cell * 1.6, y1 = y0 + dy * cell * 1.6;
    c.beginPath(); c.moveTo(x0, y0); c.lineTo(x1, y1); c.stroke();
    c.beginPath(); c.arc(x1, y1, 4, 0, 7); c.fillStyle = css('--s3'); c.fill();
  }
  if (overlay) {
    c.fillStyle = 'rgba(10,14,20,.55)'; c.fillRect(0, 0, W * cell, H * cell);
    c.fillStyle = '#fff'; c.font = `600 ${Math.max(14, cell)}px ${css('--f-body') || 'sans-serif'}`; c.textAlign = 'center';
    overlay.split('\n').forEach((line, i) => c.fillText(line, W * cell / 2, H * cell / 2 + i * cell * 1.3));
  }
}

/* The 11-feature panel ("what the network sees") with optional Q-values. */
function featurePanel(el, f, q, action) {
  el.replaceChildren(...[
    h('p.eyebrow', 'What the agent sees'),
    h('div', { style: { display: 'flex', flexWrap: 'wrap', gap: '6px' } }, ...FEATURES.map((name, i) => h('span.chip' + (f[i] ? (i < 3 ? '.bad' : '.good') : ''), { style: { opacity: f[i] ? 1 : 0.55 } }, name))),
    q ? h('p.eyebrow', { style: { marginTop: '6px' } }, 'Q-values (higher is better)') : null,
    q ? h('div', { style: { display: 'grid', gap: '6px' } }, ...ACTIONS.map((a, i) => {
      const lo = Math.min(...q), hi = Math.max(...q), wpc = hi > lo ? 15 + 85 * (q[i] - lo) / (hi - lo) : 100;
      return h('div', { style: { display: 'grid', gridTemplateColumns: '64px 1fr 56px', gap: '8px', alignItems: 'center', fontSize: '13px' } },
        h('span', a), h('span', { style: { height: '8px', borderRadius: '4px', width: wpc + '%', background: `var(--${i === q.indexOf(hi) ? 's2' : 'line-2'})` } }), h('span.mono', { style: { textAlign: 'right' } }, fmt(q[i], 2)));
    })) : null,
    action != null ? h('p', { style: { fontSize: '14px' } }, 'Move taken: ', h('b', ACTIONS[action])) : null].filter(Boolean));
}

/* ================================================================ Play */
const live = { score: 0, best: 0, length: 3, agree: 0, moves: 0 };
try { live.best = +localStorage.getItem('ml-snake-best') || 0; } catch (e) {}

const play = {
  id: 'play', group: 'agents', nav: 'Play Snake', tag: 'You vs AI', title: 'Play Snake',
  question: 'Arrow keys or WASD to steer, space to pause. Turn on the hint to see what the trained network would do.',
  params: [
    { k: 'speed', label: 'Speed', min: 4, max: 20, step: 1, value: 10, unit: 'moves/s', group: 'Game' },
    { k: 'hint', type: 'switch', label: 'Show the DQN\'s suggestion', value: true, group: 'Game', help: 'Runs the latest network trained on the Train page, in your browser, on the same 11 features.' },
  ],
  request: async (s, sig) => {
    const m = await call('/api/agents/snake/models', {}, { signal: sig });
    let w = null;
    if (m.models.length) w = await call('/api/agents/snake/weights', {}, { signal: sig });
    return { models: m.models, weights: w, meta: m.meta };
  },
  kpis: d => [
    { label: 'Score', value: live.score, note: `length ${live.length}` },
    { label: 'Your best', value: live.best, note: 'kept in this browser' },
    { label: 'Agreed with the DQN', value: live.moves ? pct(live.agree / live.moves, 0) : '—', note: d.weights ? `${live.moves} moves compared` : 'train a DQN to compare', tone: d.weights ? null : 'warn' },
    { label: 'Network', value: d.models[0]?.id || 'none', note: d.models[0] ? `record ${d.models[0].record} in training` : 'train one on the Train page' },
  ],
  illustration: {
    title: 'How the network decides', height: 200,
    draw(svg) {
      const inp = node(svg, { x: 10, y: 50, w: 150, h: 100, title: '11 features', sub: 'danger · heading · food' });
      const hid = node(svg, { x: 225, y: 60, w: 150, h: 80, title: 'Hidden layer', sub: '256 ReLU units', hot: true });
      const out = node(svg, { x: 440, y: 50, w: 150, h: 100, title: '3 Q-values', sub: 'straight · right · left' });
      edge(svg, [inp.r + 4, inp.cy], [hid.x - 6, hid.cy], { label: 'weights', width: 5, flow: true });
      edge(svg, [hid.r + 4, hid.cy], [out.x - 6, out.cy], { label: 'weights', width: 5, flow: true });
      text(svg, 300, 182, 'The move with the highest Q-value is the suggestion.', 'n-sub');
    },
  },
  charts: [{
    id: 'game', kind: 'html', title: 'The board', sub: 'Same rules as the original game: walls and your own body (including the tail) are fatal.',
    draw(host, d, s, ctx) {
      if (!ctx.extra.game) startGame(host, ctx);
      ctx.extra.weights = d.weights;
      ctx.extra.paint?.();
    },
  }],
  insights: d => [
    'The network only ever sees <b>11 yes/no features</b>, not the board. That is why it can trap itself: nothing tells it where its body will be.',
    d.weights ? 'The green arrow is the network\'s choice for your current position, computed in your browser from weights trained on the server.' : 'Train a network on the <b>Train the DQN</b> page and come back to get its suggestions.',
  ],
  maths: { eq: 'Q(s, a) = W₂ · ReLU(W₁ s + b₁) + b₂\nsuggestion = argmax over a of Q(s, a)', notes: ['s is the 11-feature state: danger straight/right/left, heading (4), and food left/right/up/down.'] },
  destroy(ctx) { ctx.extra.stop?.(); },
};

function startGame(host, ctx) {
  const cv = h('canvas', { role: 'img', 'aria-label': 'Snake board' });
  const panel = h('div', { style: { display: 'grid', gap: '10px', alignContent: 'start' } });
  const pad = h('div', { style: { display: 'grid', gridTemplateColumns: 'repeat(3, 44px)', gap: '6px', justifyContent: 'center' } });
  const status = h('p.c-note');
  const btn = (label, dir) => h('button.btn.sm', { type: 'button', 'aria-label': label, onclick: () => press(dir) }, label);
  pad.append(h('span'), btn('↑', 3), h('span'), btn('←', 2), btn('↓', 1), btn('→', 0));
  const restart = h('button.btn.primary.sm', { type: 'button', onclick: () => reset(true) }, 'New game');
  host.replaceChildren(h('div.split-panel', h('div', { style: { minWidth: 0 } }, cv), h('div', { style: { display: 'grid', gap: '12px', alignContent: 'start' } }, panel, pad, restart, status)));
  let g = newGame(), want = null, running = false, timer = null, last = 0, raf = 0;
  const reset = startNow => { g = newGame(); want = null; live.score = 0; live.length = 3; running = !!startNow; paint(); };
  function hint() {
    const w = ctx.extra.weights; if (!w || !ctx.state.hint) return null;
    const q = qvalues(w, features(g)); return { q, a: q.indexOf(Math.max(...q)) };
  }
  function paint() {
    const hn = hint();
    drawBoard(cv, g.snake, g.food, { hintDir: hn && !g.over ? turn(g.d, hn.a) : null, dead: !!g.over, overlay: g.over ? `Game over: ${CAUSE[g.over]}\nScore ${g.score} · press New game` : !running ? 'Press an arrow key to start' : '' });
    featurePanel(panel, features(g), hn?.q, null);
    status.textContent = g.over ? '' : running ? 'Space pauses.' : 'Arrow keys or WASD to move.';
  }
  function press(dir) {
    if (g.over) return;
    if ((dir + 2) % 4 === g.d) return;                 // ignore reversing into yourself
    want = dir; if (!running) { running = true; tick.last = 0; }
  }
  function tick(t) {
    raf = requestAnimationFrame(tick);
    if (!running || g.over) return;
    if (t - (tick.last || 0) < 1000 / ctx.state.speed) return;
    tick.last = t;
    let a = 0;
    if (want != null) a = want === g.d ? 0 : want === (g.d + 1) % 4 ? 1 : want === (g.d + 3) % 4 ? 2 : 0;
    want = null;
    const hn = hint();
    if (hn) { live.moves++; if (hn.a === a) live.agree++; }
    const res = step(g, a);
    live.score = g.score; live.length = g.snake.length;
    if (res === 'ate' || g.over) {
      if (g.score > live.best) { live.best = g.score; try { localStorage.setItem('ml-snake-best', String(live.best)); } catch (e) {} }
      ctx.redraw();
    }
    paint();
  }
  const keys = { ArrowRight: 0, d: 0, D: 0, ArrowDown: 1, s: 1, S: 1, ArrowLeft: 2, a: 2, A: 2, ArrowUp: 3, w: 3, W: 3 };
  const onKey = e => {
    if (/input|textarea|select/i.test(document.activeElement?.tagName)) return;
    if (e.key in keys) { e.preventDefault(); press(keys[e.key]); }
    else if (e.key === ' ') { e.preventDefault(); if (!g.over) { running = !running; paint(); } }
  };
  let touch = null;
  cv.addEventListener('touchstart', e => { touch = [e.touches[0].clientX, e.touches[0].clientY]; }, { passive: true });
  cv.addEventListener('touchend', e => { if (!touch) return; const dx = e.changedTouches[0].clientX - touch[0], dy = e.changedTouches[0].clientY - touch[1]; if (Math.max(Math.abs(dx), Math.abs(dy)) > 20) press(Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 0 : 2) : (dy > 0 ? 1 : 3)); touch = null; });
  document.addEventListener('keydown', onKey);
  const ro = new ResizeObserver(paint); ro.observe(host);
  raf = requestAnimationFrame(tick);
  ctx.extra.game = true; ctx.extra.paint = paint;
  ctx.extra.stop = () => { cancelAnimationFrame(raf); document.removeEventListener('keydown', onKey); ro.disconnect(); };
  paint();
}

/* ================================================================ Watch */
const watch = {
  id: 'watch', group: 'agents', nav: 'Watch an agent', tag: 'Replay', title: 'Watch an agent play',
  question: 'The server plays a full game with the agent you pick. Scrub through it and see what the agent saw at every move.',
  params: [
    { k: 'agent', type: 'seg', label: 'Agent', value: 'bfs', options: [['random', 'Random'], ['greedy', 'Greedy'], ['bfs', 'BFS'], ['dqn', 'DQN']], group: 'Game',
      help: 'DQN needs a network trained on the Train the DQN page.' },
    { k: 'seed', label: 'Board seed', min: 1, max: 500, step: 1, value: 7, group: 'Game', help: 'Same seed, same food positions: compare agents on an identical game.' },
    { k: 'speed', label: 'Replay speed', min: 5, max: 120, step: 5, value: 40, unit: 'moves/s', group: 'Replay' },
  ],
  presets: [
    { label: 'BFS on board 7', values: { agent: 'bfs', seed: 7 } },
    { label: 'Greedy on board 7', values: { agent: 'greedy', seed: 7 } },
    { label: 'Random walker', values: { agent: 'random', seed: 7 } },
    { label: 'Your DQN', values: { agent: 'dqn', seed: 7 } },
  ],
  request: (s, sig) => call('/api/agents/snake/episode', { agent: s.agent, seed: s.seed }, { signal: sig }).catch(e => {
    if (e.code === 'no_model') { e.message = 'There is no trained DQN on the server yet. Train one on "Train the DQN" (about 10 seconds), then come back.'; }
    throw e;
  }),
  kpis: d => [
    { label: 'Score', value: d.score, note: `final length ${d.length}`, tone: d.score >= 40 ? 'good' : d.score >= 10 ? null : 'warn' },
    { label: 'Moves', value: int(d.steps), note: d.score ? `${fmt(d.steps / d.score, 0)} moves per food` : 'never ate' },
    { label: 'How it ended', value: CAUSE[d.cause] || d.cause, tone: d.cause === 'limit' || d.cause === 'won' ? 'good' : 'bad' },
    { label: 'Moves by type', value: ACTIONS.map((a, i) => pct(d.actions.filter(x => x === i).length / d.actions.length, 0)).join(' / '), note: 'straight / right / left' },
  ],
  illustration: {
    title: 'One decision', height: 200,
    draw(svg, d, s) {
      const ag = AGENT[s.agent];
      const inp = node(svg, { x: 10, y: 60, w: 140, h: 80, title: 'Board', sub: s.agent === 'bfs' ? 'full grid' : '11 features' });
      const mid = node(svg, { x: 220, y: 50, w: 160, h: 100, title: ag.label, sub: { random: 'coin flip', greedy: 'closer to food?', bfs: 'shortest path', dqn: 'argmax Q' }[s.agent], hot: true });
      edge(svg, [inp.r + 4, inp.cy], [mid.x - 6, mid.cy], { width: 4, flow: true });
      const share = d ? ACTIONS.map((_, i) => d.actions.filter(x => x === i).length / d.actions.length) : [1 / 3, 1 / 3, 1 / 3];
      ACTIONS.forEach((a, i) => {
        const o = node(svg, { x: 470, y: 18 + i * 60, w: 120, h: 44, title: a, sub: d ? pct(share[i], 0) + ' of moves' : '' });
        edge(svg, [mid.r + 4, mid.cy], [o.x - 6, o.cy], { width: widthFor(share[i], 0.8) });
      });
    },
  },
  charts: [
    { id: 'replay', kind: 'html', title: 'Replay', sub: (d, s) => AGENT[s.agent].what,
      draw(host, d, s, ctx) { replayer(host, d, ctx); } },
    { id: 'score', title: 'Score through the game', sub: 'Each step up is a piece of food.', legend: (d, s) => [[AGENT[s.agent].color, 'line', AGENT[s.agent].label]],
      draw(host, d, s) {
        let sc = 0; const ys = d.ate.map(a => (sc += a ? 1 : 0)), xs = ys.map((_, i) => i + 1);
        const P = frame(host, { h: 200, x: [0, Math.max(10, xs.length)], y: [0, Math.max(5, sc) * 1.1], xl: 'Move', yl: 'Score' });
        P.line(zip(xs, ys), AGENT[s.agent].color === 'neutral' ? 'sn' : AGENT[s.agent].color);
        hover(host, P, px => { const i = Math.max(0, Math.min(xs.length - 1, Math.round(P.xInv(px)) - 1)); return { cross: P.X(xs[i]), html: tip(`Move ${xs[i]}`, [['Score', ys[i]], ['Move', ACTIONS[d.actions[i]]]]) }; });
      } },
  ],
  insights: (d, s) => [
    `${AGENT[s.agent].label}: <b>${d.score}</b> food in ${int(d.steps)} moves, then it ${CAUSE[d.cause] || d.cause}.`,
    { random: 'A random walker almost never finds food on a 32 × 24 board: it hits a wall within about a hundred moves.', greedy: 'Greedy grows quickly but has no plan: as the body gets long it walks into dead ends it cannot see.', bfs: 'BFS sees the whole board and checks it keeps room to move, so it survives far longer than the others.', dqn: 'The DQN plays from 11 features only. Watch the danger chips light up just before it turns.' }[s.agent],
    'Try the same seed with another agent: the food appears in the same places until their paths diverge.',
  ],
  maths: { eq: 'state  s = 11 features (danger ×3, heading ×4, food ×4)\naction a ∈ {straight, right, left}\nreward  +10 food, −10 death, 0 otherwise', notes: ['Games end on a wall, on the snake\'s own body (tail included) or after 100 × length moves, as in the original game. Replays are capped at 3,000 moves.'] },
  destroy(ctx) { ctx.extra.stopReplay?.(); },
};

function replayer(host, d, ctx) {
  ctx.extra.stopReplay?.();
  const cv = h('canvas', { role: 'img', 'aria-label': 'Replay of the game' });
  const panel = h('div', { style: { display: 'grid', gap: '10px', alignContent: 'start' } });
  const slider = h('input', { type: 'range', min: 0, max: d.heads.length, step: 1, value: 0, 'aria-label': 'Move' });
  const playBtn = h('button.btn.sm', { type: 'button' }, 'Pause');
  const label = h('span.mono', { style: { fontSize: '13px' } });
  host.replaceChildren(h('div.split-panel',
    h('div', { style: { minWidth: 0, display: 'grid', gap: '10px' } }, cv, h('div', { style: { display: 'flex', gap: '10px', alignItems: 'center' } }, playBtn, slider, label)), panel));
  // Precompute every frame's snake and food from the compact replay.
  const frames = []; let snake = d.start.map(p => p.slice()), fi = 0;
  frames.push({ snake: snake.slice(), food: d.foods[0] });
  d.heads.forEach((hd, i) => {
    snake = [hd, ...snake]; if (d.ate[i]) fi++; else snake.pop();
    frames.push({ snake: snake.slice(), food: d.foods[Math.min(fi, d.foods.length - 1)] });
  });
  let t = 0, playing = true, raf = 0, last = 0;
  const show = i => {
    t = Math.max(0, Math.min(frames.length - 1, i)); slider.value = t;
    const f = frames[t], end = t === frames.length - 1;
    drawBoard(cv, f.snake, f.food, { dead: end && ['wall', 'self', 'timeout'].includes(d.cause), overlay: end ? `${CAUSE[d.cause] || d.cause} · score ${d.score}` : '' });
    const si = Math.min(t, d.states.length - 1);
    featurePanel(panel, d.states[si].split('').map(Number), d.q?.[si], d.actions[si]);
    label.textContent = `move ${t} / ${frames.length - 1}`;
  };
  const loop = ts => {
    raf = requestAnimationFrame(loop);
    if (!playing || ts - last < 1000 / ctx.state.speed) return;
    last = ts;
    if (t >= frames.length - 1) { playing = false; playBtn.textContent = 'Replay'; return; }
    show(t + 1);
  };
  playBtn.addEventListener('click', () => { if (t >= frames.length - 1) show(0); playing = !playing; playBtn.textContent = playing ? 'Pause' : 'Play'; });
  slider.addEventListener('input', () => { playing = false; playBtn.textContent = 'Play'; show(+slider.value); });
  const ro = new ResizeObserver(() => show(t)); ro.observe(host);
  raf = requestAnimationFrame(loop);
  ctx.extra.stopReplay = () => { cancelAnimationFrame(raf); ro.disconnect(); };
  show(0);
}

/* ================================================================ Train */
const train = {
  id: 'train', group: 'agents', nav: 'Train the DQN', tag: 'Deep Q-learning', title: 'Train the Deep Q-network',
  question: 'The original project\'s agent, trained on the server while you watch: it starts random and learns from its own rewards.',
  params: [
    { k: 'games', label: 'Games to play', min: 20, max: 1000, step: 10, value: 200, group: 'Training', help: 'About 20 games a second on the server; longer runs score higher.' },
    { k: 'explore_games', label: 'Exploration games', min: 0, max: 300, step: 10, value: 80, group: 'Training', help: 'ε = (this − games played) / 200: the chance of a random move. 80 is the original setting.' },
    { k: 'gamma', label: 'Discount γ', min: 0.5, max: 0.99, step: 0.01, value: 0.9, group: 'Learning', help: 'How much future food counts compared with food now.' },
    { k: 'lr', label: 'Learning rate', min: 0.0001, max: 0.01, step: 0.0001, value: 0.001, group: 'Learning' },
    { k: 'hidden', label: 'Hidden units', min: 16, max: 512, step: 16, value: 256, group: 'Network' },
    { k: 'seed', label: 'Random seed', min: 1, max: 999, step: 1, value: 7, group: 'Network' },
  ],
  presets: [
    { label: 'Original settings', values: { games: 200, explore_games: 80, gamma: 0.9, lr: 0.001, hidden: 256 } },
    { label: 'Quick run (50 games)', values: { games: 50, explore_games: 40, gamma: 0.9, lr: 0.001, hidden: 256 } },
    { label: 'No exploration', values: { games: 200, explore_games: 0, gamma: 0.9, lr: 0.001, hidden: 256 } },
    { label: 'Short-sighted (γ = 0.5)', values: { games: 200, explore_games: 80, gamma: 0.5, lr: 0.001, hidden: 256 } },
    { label: 'Tiny network (16)', values: { games: 200, explore_games: 80, gamma: 0.9, lr: 0.001, hidden: 16 } },
  ],
  request: (s, sig) => call('/api/agents/snake/models', {}, { signal: sig }),
  actions: [{ label: 'Train on the server', icon: 'play', primary: true, async run(ctx, ui) {
    const s = ctx.state;
    const r = await ctx.job({ kind: 'snake_dqn', games: s.games, explore_games: s.explore_games, gamma: s.gamma, lr: s.lr, hidden: s.hidden, seed: s.seed });
    ctx.toast(`Trained ${r.model.id}: record ${r.model.record}, last-50 mean ${fmt(r.model.mean_last50, 1)}.`);
    await ctx.now();
  } }],
  kpis: d => {
    const m = d.models[0];
    if (!m) return [{ label: 'Trained networks', value: 0, note: 'press Train on the server to start', tone: 'warn' }];
    return [
      { label: 'Best score in training', value: m.record, note: m.id },
      { label: 'Mean of the last 50 games', value: fmt(m.mean_last50, 1), tone: m.mean_last50 > 15 ? 'good' : m.mean_last50 > 3 ? null : 'warn' },
      { label: 'Games played', value: int(m.games), note: `ε from ${fmt(m.explore_games / 200, 2)} to 0` },
      { label: 'Training time', value: fmtMs(m.seconds * 1000), note: `${m.hidden} hidden units · γ ${m.gamma}` },
    ];
  },
  illustration: {
    title: 'The learning loop', height: 220,
    caption: 'Every move is trained on immediately; after each game, a random batch of up to 1,000 remembered moves is replayed.',
    draw(svg, d, s) {
      const net = node(svg, { x: 225, y: 12, w: 150, h: 70, title: 'Q-network', sub: `11 → ${s.hidden} → 3`, hot: true });
      const env = node(svg, { x: 15, y: 112, w: 150, h: 66, title: 'Snake game', sub: '+10 food · −10 death' });
      const mem = node(svg, { x: 435, y: 112, w: 150, h: 66, title: 'Replay memory', sub: '≤ 100,000 moves' });
      edge(svg, [env.cx - 20, env.y - 4], [net.x - 6, net.cy - 10], { label: 'state, reward', labelAt: [70, 56], width: 3, flow: true });
      edge(svg, [net.x - 6, net.cy + 14], [env.cx + 24, env.y - 4], { label: 'move', labelAt: [190, 104], width: 3 });
      edge(svg, [env.r + 4, env.cy + 14], [mem.x - 6, mem.cy + 14], { below: 'remember every move', width: 2 });
      edge(svg, [mem.cx, mem.y - 4], [net.r + 4, net.cy], { label: 'replay a batch', labelAt: [520, 56], width: 3, cls: 'st2' });
      text(svg, 300, 214, `ε random for the first ${s.explore_games} games · target = reward + ${s.gamma} × max Q(next)`, 'n-sub');
    },
  },
  charts: [
    { id: 'curve', title: 'Learning curve', show: d => !!d.latest, sub: d => d.latest ? `Model ${d.latest.model}: score of each training game and the running mean, like the original project's plot.` : '',
      legend: [['s2', 'dot', 'Score per game'], ['s1', 'line', 'Running mean'], ['s3', 'line', 'Mean of last 20']],
      draw(host, d) {
        const c = d.latest, n = c.scores.length, xs = c.scores.map((_, i) => i + 1);
        const roll = c.scores.map((_, i) => { const a = c.scores.slice(Math.max(0, i - 19), i + 1); return a.reduce((x, y) => x + y, 0) / a.length; });
        const P = frame(host, { h: 300, x: [0, n + 1], y: [0, Math.max(...c.scores, 5) * 1.08], xl: 'Training game', yl: 'Score' });
        c.scores.forEach((v, i) => P.dot(xs[i], v, 'f2', n > 300 ? 2 : 3));
        P.line(zip(xs, roll), 's3'); P.line(zip(xs, c.mean), 's1');
        hover(host, P, px => { const i = Math.max(0, Math.min(n - 1, Math.round(P.xInv(px)) - 1)); return { cross: P.X(xs[i]), html: tip(`Game ${xs[i]}`, [['Score', c.scores[i]], ['Running mean', fmt(c.mean[i], 2)], ['Last 20', fmt(roll[i], 1)], ['ε', fmt(c.epsilon[i], 2)]]) }; });
      } },
    { id: 'eps', title: 'Exploration and game length', show: d => !!d.latest, sub: 'ε falls to zero; games get longer as the snake learns to survive.', legend: [['s4', 'line', 'Exploration ε'], ['neutral', 'line', 'Moves per game (scaled)']],
      draw(host, d) {
        const c = d.latest, xs = c.epsilon.map((_, i) => i + 1), mx = Math.max(...c.steps, 1);
        const P = frame(host, { h: 220, x: [0, xs.length + 1], y: [0, Math.max(0.42, ...c.epsilon) * 1.05], xl: 'Training game', yl: 'ε' });
        P.line(zip(xs, c.steps.map(v => v / mx * 0.4)), 'sn thin'); P.line(zip(xs, c.epsilon), 's4');
      } },
    { id: 'empty', kind: 'html', title: 'No network yet', show: d => !d.latest,
      draw(host) { host.classList.add('empty'); host.textContent = 'Press "Train on the server". 200 games take about 10 seconds; the learning curve appears here.'; } },
  ],
  chartLayout: 'one',
  insights: d => {
    const c = d.latest;
    if (!c) return ['The network starts with random weights and plays randomly. Within about 100 games it learns to head for food and avoid walls.'];
    const first = c.scores.slice(0, 20), last = c.scores.slice(-20), avg = a => a.reduce((x, y) => x + y, 0) / a.length;
    return [
      `The first 20 games averaged <b>${fmt(avg(first), 1)}</b>; the last 20 averaged <b>${fmt(avg(last), 1)}</b>.`,
      'Scores stay noisy because food appears in random places and one bad turn ends the game. The running mean is the signal.',
      'Compare it with the scripted agents on <b>Compare agents</b>: BFS sees the whole board and usually wins.',
    ];
  },
  maths: { eq: 'ε = max(0, explore − game) / 200\nQ_target = r + γ · max_a′ Q(s′, a′)   (r only, if the game ended)\nloss = (Q(s, a) − Q_target)²   ·   Adam', notes: ['Same algorithm and hyperparameters as the original project; the replay targets are computed in one batched pass instead of a Python loop.'] },
};

/* ================================================================ Compare */
const compare = {
  id: 'compare', group: 'agents', nav: 'Compare agents', tag: 'Monte Carlo', title: 'Compare the agents',
  question: 'Every agent plays the same set of boards. Which strategy scores most, and how do they die?',
  params: [
    { k: 'random', type: 'switch', label: AGENT.random.label, value: true, group: 'Agents' },
    { k: 'greedy', type: 'switch', label: AGENT.greedy.label, value: true, group: 'Agents' },
    { k: 'bfs', type: 'switch', label: AGENT.bfs.label, value: true, group: 'Agents', help: 'The slowest: about half a second per game.' },
    { k: 'dqn', type: 'switch', label: AGENT.dqn.label, value: false, group: 'Agents', help: 'Uses your latest network from Train the DQN.' },
    { k: 'episodes', label: 'Games per agent', min: 5, max: 100, step: 5, value: 20, group: 'Games' },
    { k: 'seed', label: 'First board seed', min: 1, max: 999, step: 1, value: 1, group: 'Games' },
  ],
  emptyText: 'Pick the agents and press "Run the comparison". Every agent plays the same boards.',
  actions: [{ label: 'Run the comparison', icon: 'play', primary: true, async run(ctx) {
    const s = ctx.state, agents = Object.keys(AGENT).filter(a => s[a]);
    if (!agents.length) throw new Error('Pick at least one agent.');
    const t0 = performance.now();
    const r = await ctx.job({ kind: 'snake_compare', agents, episodes: s.episodes, seed: s.seed });
    ctx.setData({ ...r, meta: { compute_ms: performance.now() - t0 } });
  } }],
  kpis: d => {
    const rows = Object.entries(d.agents).sort((a, b) => b[1].mean - a[1].mean), [best, bs] = rows[0];
    return [
      { label: 'Best agent', value: AGENT[best].label, note: `mean ${fmt(bs.mean, 1)} ± ${fmt(bs.ci95, 1)}`, tone: 'good' },
      { label: 'Highest single game', value: Math.max(...rows.map(r => r[1].max)), note: AGENT[rows.reduce((a, b) => (b[1].max > a[1].max ? b : a))[0]].label },
      d.agents.random ? { label: 'Random walker', value: fmt(d.agents.random.mean, 1), note: 'mean score: the baseline' } : null,
      d.agents.dqn && d.agents.bfs ? { label: 'DQN vs BFS', value: pct(d.agents.dqn.mean / d.agents.bfs.mean, 0), note: 'of BFS\'s mean score' } : { label: 'Games played', value: int(rows.length * d.episodes), note: `${d.episodes} per agent` },
    ];
  },
  illustration: {
    title: 'A fair comparison', height: 190,
    caption: 'Board k always uses seed k, so every agent faces the same food positions.',
    draw(svg, d, s) {
      const ag = Object.keys(AGENT).filter(a => s[a]);
      const seeds = node(svg, { x: 10, y: 60, w: 150, h: 70, title: `${s.episodes} boards`, sub: `seeds ${s.seed}–${s.seed + s.episodes - 1}` });
      ag.forEach((a, i) => {
        const y = 10 + i * (170 / Math.max(1, ag.length));
        const n = node(svg, { x: 300, y, w: 150, h: 34, title: AGENT[a].label, sub: '' });
        edge(svg, [seeds.r + 4, seeds.cy], [n.x - 6, n.cy], { width: 2 });
        if (d?.agents[a]) text(svg, 470, n.cy + 5, `mean ${fmt(d.agents[a].mean, 1)}`, 'edge-lbl', 'start');
      });
    },
  },
  chartLayout: 'two',
  charts: [
    { id: 'dist', title: 'Score of every game', sub: 'Each dot is one game; the bar is the mean with its 95% confidence interval.',
      legend: d => Object.keys(d.agents).map(a => [AGENT[a].color, 'dot', AGENT[a].label]),
      draw(host, d) {
        const names = Object.keys(d.agents), mx = Math.max(5, ...names.flatMap(a => d.agents[a].scores));
        const P = frame(host, { h: 300, x: [-0.6, names.length - 0.4], y: [0, mx * 1.08], xt: names.map((_, i) => i), xf: i => AGENT[names[i]]?.label.split(' ')[0] || '', yl: 'Score' });
        names.forEach((a, i) => {
          const r = d.agents[a], col = AGENT[a].color === 'neutral' ? 'fn' : 'f' + AGENT[a].color.slice(1);
          r.scores.forEach((v, j) => P.dot(i + ((j * 37) % 17 - 8) / 30, v, col, 3));
          const x0 = P.X(i - 0.3), x1 = P.X(i + 0.3);
          E('line', { x1: x0, x2: x1, y1: P.Y(r.mean), y2: P.Y(r.mean), stroke: 'var(--ink)', 'stroke-width': 2.5 }, P.body);
          E('rect', { x: x0, width: x1 - x0, y: P.Y(r.mean + r.ci95), height: Math.max(1, P.Y(r.mean - r.ci95) - P.Y(r.mean + r.ci95)), fill: 'var(--ink)', opacity: 0.08 }, P.body);
        });
        hover(host, P, px => { const i = Math.max(0, Math.min(names.length - 1, Math.round(P.xInv(px)))), r = d.agents[names[i]]; return { html: tip(AGENT[names[i]].label, [['Mean', `${fmt(r.mean, 1)} ± ${fmt(r.ci95, 1)}`], ['Median', fmt(r.median, 0)], ['Best', r.max], ['Moves per food', fmt(r.steps_per_food, 0)]]) }; });
      } },
    { id: 'causes', title: 'How the games ended', sub: 'Share of games by cause.',
      legend: [['bad', 'box', 'Hit a wall'], ['s2', 'box', 'Hit itself'], ['s4', 'box', 'Ran out of moves'], ['s3', 'box', 'Step cap or full board']],
      draw(host, d) {
        const names = Object.keys(d.agents), rowH = 40, P = frame(host, { h: names.length * rowH + 46, x: [0, 1], y: [0, names.length], yt: [], xf: v => Math.round(v * 100) + '%', xn: 4, m: { l: 110, r: 16, t: 8, b: 30 } });
        const order = [['wall', 'var(--bad)'], ['self', 'var(--s2)'], ['timeout', 'var(--s4)'], ['limit', 'var(--s3)'], ['won', 'var(--s3)']];
        names.forEach((a, i) => {
          const r = d.agents[a], n = r.scores.length, y = P.Y(names.length - i - 0.5);
          let acc = 0;
          order.forEach(([c, col]) => { const v = r.causes[c] / n; if (!v) return; E('rect', { x: P.X(acc) + 1, y: y - 11, width: Math.max(1, P.X(acc + v) - P.X(acc) - 2), height: 22, rx: 3, fill: col }, P.body); acc += v; });
          P.text(102, y + 4, AGENT[a].label.split(' ')[0], { 'text-anchor': 'end' });
        });
        hover(host, P, (px, py) => { const i = names.length - 1 - Math.floor((P.h - P.M.b - py) / rowH); if (i < 0 || i >= names.length) return null; const r = d.agents[names[i]], n = r.scores.length; return { html: tip(AGENT[names[i]].label, Object.entries(CAUSE).filter(([c]) => r.causes[c]).map(([c, t]) => [t, `${r.causes[c]} of ${n}`])) }; });
      } },
  ],
  insights: d => {
    const a = d.agents, out = [];
    if (a.random) out.push(`The random walker averages <b>${fmt(a.random.mean, 1)}</b>: on a 32 × 24 board a random walk hits a wall long before it finds food, as Lab 10's √n spread predicts.`);
    if (a.greedy && a.bfs) out.push(`BFS beats greedy (<b>${fmt(a.bfs.mean, 1)}</b> vs ${fmt(a.greedy.mean, 1)}) because it plans a path and checks it keeps room; greedy dies in dead ends it walks into.`);
    if (a.dqn) out.push(`Your DQN averages <b>${fmt(a.dqn.mean, 1)}</b>. It only sees 11 features, so it cannot plan around its own body the way BFS does.`);
    return out;
  },
  maths: { eq: 'mean ± 1.96 · sd / √n   (95% confidence interval)', notes: ['Games are capped at 3,000 moves. A game that reaches the cap is still alive; its score is a lower bound.'] },
};

export default [play, watch, train, compare];
