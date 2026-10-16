// Crossover tab: market data through the spread and shape toolkits, every number from the server.
import { $, $$, bind, call, fdate, fmt, int, livePanel, mon, pct, readouts, segmented, verdicts } from './api.js';
import { E, extent, frame, hover, nearest, responsive, zip } from './plot.js';

const tip = (title, rows) => `<b>${title}</b>` + rows.map(([k, v]) => `<br>${k}: <span class="n">${v}</span>`).join('');
const yearOf = s => +s.slice(0, 4) + (+s.slice(5, 7) - 1) / 12 + (+s.slice(8, 10)) / 365;

/* ------------------------------------------------------------ A: random walk */
function randomWalk(root) {
  const s = { ticker: 'SPY', sims: 2000, seed: 312, n: 20, walkSeed: 1, tails: 'raw' };
  const sec = $('[data-study="rw"]', root);
  const pPaths = $('[data-panel="rw-paths"]', sec), pStudy = $('[data-panel="rw-study"]', sec);
  const paths = livePanel(pPaths, { request: sig => call('/api/crossover/walks', { ticker: s.ticker, n: s.n, seed: s.walkSeed }, { signal: sig }), render: renderPaths });
  const study = livePanel(pStudy, { request: sig => call('/api/crossover/random-walk', { ticker: s.ticker, sims: s.sims, seed: s.seed }, { signal: sig }), render: renderStudy, delay: 250 });
  segmented($('[data-seg="ticker"]', sec), v => { s.ticker = v; $$('[data-out="ticker"]', root).forEach(e => e.textContent = v); paths.refresh(); study.refresh(); });
  $('[data-act="walks"]', pPaths).addEventListener('click', () => { s.walkSeed = Math.floor(Math.random() * 1e9); paths.now(); });
  bind(pStudy, s, 'sims', () => study.refresh(), v => int(v) + ' histories');
  $('[data-act="reseed"]', pStudy).addEventListener('click', () => { s.seed = Math.floor(Math.random() * 1e9); study.now(); });
  segmented($('[data-seg="tails"]', pStudy), v => { s.tails = v; study.redraw(); });

  const hp = $('.plot-host', pPaths);
  const drawPaths = responsive(hp, () => {
    const d = paths.data; if (!d) return;
    const yrs = d.dates.map(yearOf), [lo, hi] = extent(d.real, ...d.walks);
    const yt = [0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500].map(Math.log).filter(v => v >= lo - 0.3 && v <= hi + 0.3);
    const P = frame(hp, { h: 340, x: [yrs[0], yrs[yrs.length - 1]], y: [Math.min(lo, Math.log(0.5)), hi * 1.03], yt, yf: v => +Math.exp(v).toPrecision(2) + '×', xf: v => Math.round(v) });
    d.walks.forEach(w => P.line(zip(yrs, w), 'sn thin').setAttribute('opacity', '.45'));
    P.line(zip(yrs, d.real), 's1'); P.hline(0);
    hover(hp, P, px => { const i = nearest(yrs, P.xInv(px)); return { cross: P.X(yrs[i]), pts: [[P.X(yrs[i]), P.Y(d.real[i]), 'f1']], html: tip(mon(d.dates[i]), [[`${s.ticker}: $1 →`, '$' + fmt(Math.exp(d.real[i]), 2)]]) }; });
  });
  function renderPaths(d) {
    readouts($('.readout', pPaths), [['Weekly drift', pct(d.weekly_mu, 2)], ['Weekly σ', pct(d.weekly_sigma, 2)], [`${s.ticker} ends at`, fmt(d.real_multiple, 1) + '×'],
      ['Walks end between', `${fmt(d.walk_multiples[0], 1)}× and ${fmt(d.walk_multiples[d.walk_multiples.length - 1], 1)}×`]]);
    drawPaths();
  }

  const [hT, hV, hA, hM] = $$('.plot-host', pStudy);
  const normCdf = z => { const t = 1 / (1 + 0.2316419 * Math.abs(z)), q = 0.3989423 * Math.exp(-z * z / 2) * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274)))); return z > 0 ? 1 - q : q; };
  const drawStudy = () => {
    const d = study.data; if (!d) return;
    const H = d.hist, raw = s.tails === 'raw', counts = raw ? H.raw : H.scaled, n = raw ? H.n_raw : H.n_scaled, ed = H.edges;
    let P = frame(hT, { h: 280, x: [-8, 8], y: [0.3, 3000], ylog: true, yt: [1, 10, 100, 1000], yf: v => v >= 1000 ? v / 1000 + 'k' : v, xt: [-8, -6, -4, -2, 0, 2, 4, 6, 8], xf: v => v + 'σ', xl: 'Daily move in standard deviations', yl: 'Days (log scale)' });
    P.bars(ed, counts, 'f1', 0.3);
    const exp = []; for (let z = -8; z <= 8.001; z += 0.05) exp.push([z, Math.max(n * (normCdf(z + 0.125) - normCdf(z - 0.125)), 0.3)]);
    P.line(exp, 's2'); P.vline(-4); P.vline(4);
    hover(hT, P, px => { const z = P.xInv(px), i = Math.max(0, Math.min(ed.length - 2, Math.floor((z + 8) / 0.25))), e = n * (normCdf(ed[i + 1]) - normCdf(ed[i])); return { html: tip(`${fmt(ed[i])}σ to ${fmt(ed[i + 1])}σ`, [['Days', counts[i]], ['Normal expects', e < 0.01 ? e.toExponential(1) : fmt(e, 2)]]) }; });
    const T = d.tails;
    readouts($('[data-out="tails"]', pStudy), [['Days beyond 4σ', raw ? T.beyond4 : T.beyond4_scaled], ['Normal predicts', fmt(raw ? T.expected4 : T.expected4_scaled, 2)], ['Kurtosis', fmt(raw ? T.kurtosis : T.kurtosis_scaled, 1) + ' (normal 3)'], ['Worst day', `${fdate(T.worst_day)}, ${pct(T.worst_return)}`]]);

    const V = d.variance_ratio;
    P = frame(hV, { h: 250, x: [1, 63], xlog: true, xt: [1, 2, 5, 10, 21, 63], y: [Math.min(0.5, ...V.map(v => v.vr)) - 0.05, Math.max(1.3, ...V.map(v => v.hi)) + 0.05], xl: 'Horizon k, trading days (log)', yl: 'Variance ratio' });
    P.area(V.map(v => [v.k, v.hi]), V.map(v => [v.k, v.lo]), 'band-null'); P.hline(1);
    P.line(V.map(v => [v.k, v.vr]), 's1'); V.forEach(v => P.dot(v.k, v.vr, 'f1', 4));
    const rob = Object.fromEntries(d.robust.map(r => [r.k, r]));
    hover(hV, P, px => { const xs = V.map(v => P.X(v.k)), i = nearest(xs, px), v = V[i], r = rob[v.k]; return { cross: xs[i], pts: [[xs[i], P.Y(v.vr), 'f1']], html: tip(`k = ${v.k} days`, [['Ratio', fmt(v.vr, 3)], ['Shuffled 95%', `${fmt(v.lo)}–${fmt(v.hi)}`]].concat(r ? [['Robust z', fmt(r.z)]] : [])) }; });
    $('[data-out="vr"]', pStudy).textContent = `Lo–MacKinlay robust z: ${d.robust.map(r => `${fmt(r.z, 1)} at ${r.k}d`).join(', ')}. After volatility scaling: ${fmt(rob[10].z_scaled, 1)} at 10d. By decade at 10d: ${d.decades.map(x => `${fmt(x.z10, 1)} (${x.span.replace('-', '–')})`).join(', ')}.`;

    const A = d.acf_abs, R = d.acf_r, L = A.map((_, i) => i + 1), b = d.acf_band;
    P = frame(hA, { h: 250, x: [1, L.length], y: [Math.min(-0.1, ...R) - 0.02, Math.max(0.35, ...A) + 0.02], xl: 'Lag, trading days', yl: 'Autocorrelation' });
    P.area([[1, b], [L.length, b]], [[1, -b], [L.length, -b]], 'band-null'); P.hline(0);
    P.line(zip(L, R), 's2 thin'); P.line(zip(L, A), 's1');
    hover(hA, P, px => { const i = nearest(L, P.xInv(px)); return { cross: P.X(L[i]), pts: [[P.X(L[i]), P.Y(A[i]), 'f1'], [P.X(L[i]), P.Y(R[i]), 'f2']], html: tip(`Lag ${L[i]}`, [['|r|', fmt(A[i], 3)], ['r', fmt(R[i], 3)]]) }; });

    const M = d.drawdown, bins = M.bins, hs = M.hist_normal, hb = M.hist_bootstrap;
    P = frame(hM, { h: 240, x: [0, 1], y: [0, Math.max(...hs, ...hb) * 1.12], xf: v => Math.round(v * 100) + '%', xl: 'Maximum drawdown over the whole period', yl: 'Simulations' });
    P.bars(bins, hs, 'fn', 0);
    P.line(bins.slice(0, -1).map((x, i) => [x + 0.01, hb[i]]), 's3');
    P.vline(M.real, 'ln s1'); P.text(P.X(M.real) + 6, P.M.t + 12, `${s.ticker} real: ${pct(M.real, 0)}`);
    hover(hM, P, px => { const i = Math.max(0, Math.min(hs.length - 1, Math.floor(P.xInv(px) / 0.02))); return { html: tip(`${Math.round(bins[i] * 100)}–${Math.round(bins[i + 1] * 100)}%`, [['Normal walks', hs[i]], ['Resampled real days', hb[i]]]) }; });
    readouts($('[data-out="mdd"]', pStudy), [['Real max drawdown', pct(M.real, 0)], ['Median simulated', pct(M.median_normal, 0)], ['Normal walks with a deeper crash', pct(M.share_worse_normal, 1)], ['Resampled walks with a deeper crash', pct(M.share_worse_bootstrap, 1)]]);
  };
  [hT, hV, hA, hM].forEach(h => responsive(h, drawStudy));
  function renderStudy(d) {
    const kh = d.kurtosis_by_horizon, T = d.tails;
    verdicts($('.verdict', sec), [
      ['Independent steps?', `<b>No.</b> Big days cluster: |r| autocorrelation is ${fmt(d.acf_abs[0], 2)} at lag 1 and still ${fmt(d.acf_abs[99], 2)} at lag 100, where a random walk gives 0.`],
      ['Normal steps?', `<b>No.</b> ${T.beyond4} days beyond 4σ where ${fmt(T.expected4, 1)} are expected. Kurtosis falls from ${fmt(kh[0].kurtosis, 1)} daily to ${fmt(kh[kh.length - 1].kurtosis, 1)} quarterly, on the way to the CLT's 3.`],
      ['Wrong about crashes?', `<b>${d.drawdown.share_worse_normal > 0.05 ? 'Not as much as expected.' : 'Yes.'}</b> ${pct(d.drawdown.share_worse_normal, 0)} of ${int(d.drawdown.sims)} simulated histories have a deeper drawdown than the real one.`],
    ]);
    drawStudy();
  }
  return [paths, study];
}

/* ------------------------------------------------------------ B: dosing */
function dosing(root) {
  const sec = $('[data-study="dose"]', root);
  const e = { ratio: 1.6, gap: 63, horizon: 60 };
  const m = { compartments: '1', half_life: 12.5, threshold: 3.4, window: 'all', a: 0.77, hl_fast: 5.9, hl_slow: 44 };
  const pE = $('[data-panel="dose-events"]', sec), pM = $('[data-panel="dose-model"]', sec);
  const events = livePanel(pE, { request: sig => call('/api/crossover/vix-events', e, { signal: sig }), render: renderEvents });
  const model = livePanel(pM, { request: sig => call('/api/crossover/vix-model', m, { signal: sig }), render: renderModel, delay: 80 });
  bind(pE, e, 'ratio', () => events.refresh(), v => '×' + fmt(v, 2)); bind(pE, e, 'gap', () => events.refresh(), v => v + ' days');
  const hl = bind(pM, m, 'half_life', () => model.refresh(), v => fmt(v, 1) + ' days');
  bind(pM, m, 'threshold', () => model.refresh(), v => '+' + fmt(v, 1) + ' pts');
  segmented($('[data-seg="comp"]', pM), v => { m.compartments = v; $('[data-k="half_life"]', pM).disabled = v === '2'; model.refresh(); });
  segmented($('[data-seg="window"]', pM), v => { m.window = v; model.refresh(); });
  $('[data-act="fit"]', pM).addEventListener('click', async () => {
    const btn = $('[data-act="fit"]', pM); btn.disabled = true;
    try {
      const f = await call('/api/crossover/vix-model', Object.assign({}, m, { fit: true, points: 100 }));
      if (m.compartments === '1') hl.set(+f.half_life.toFixed(1)); else Object.assign(m, { a: f.a, hl_fast: f.hl_fast, hl_slow: f.hl_slow });
      await model.now();
    } finally { btn.disabled = false; }
  });

  const hC = $('.plot-host', pE);
  const drawEvents = responsive(hC, () => {
    const d = events.data; if (!d) return;
    const h = d.h, lo = Math.min(-0.2, ...d.q25);
    const P = frame(hC, { h: 280, x: [0, h[h.length - 1]], y: [lo, 1.05], yt: [0, 0.25, 0.5, 0.75, 1], yf: v => Math.round(v * 100) + '%', xl: 'Trading days after the peak', yl: 'Share of the spike left' });
    P.area(zip(h, d.q75), zip(h, d.q25), 'band'); P.hline(0); P.hline(0.5);
    P.line(zip(h, d.one.curve), 's2 dashed'); P.line(zip(h, d.two.curve), 's3'); P.line(zip(h, d.median), 's1');
    hover(hC, P, px => { const i = nearest(h, P.xInv(px)); return { cross: P.X(i), pts: [[P.X(i), P.Y(d.median[i]), 'f1']], html: tip(`Day ${i}`, [['Median spike', pct(d.median[i], 0)], ['Middle 50%', `${pct(d.q25[i], 0)}–${pct(d.q75[i], 0)}`], ['One compartment', pct(d.one.curve[i], 0)], ['Two compartments', pct(d.two.curve[i], 0)]]) }; });
  });
  function renderEvents(d) {
    $('[data-out="n"]', pE).textContent = d.events.length;
    readouts($('.readout', pE), [['One compartment', `half-life ${fmt(d.one.half_life, 1)} d, R² ${fmt(d.one.r2)}`], ['Two compartments', `${pct(d.two.a, 0)} at ${fmt(d.two.hl_fast, 1)} d + rest at ${d.two.hl_slow == null ? '∞' : fmt(d.two.hl_slow, 0) + ' d'}, R² ${fmt(d.two.r2)}`], ['Whole-series AR(1)', `half-life ${fmt(d.ar1.half_life, 0)} d`]]);
    $('tbody', pE).innerHTML = d.events.map(x => `<tr><td>${fdate(x.date)}</td><td class="num">${fmt(x.peak, 1)}</td><td class="num">${fmt(x.base, 1)}</td><td class="num">${x.days_to_half ?? '> ' + d.params.horizon}</td></tr>`).join('');
    drawEvents(); verdict();
  }

  const hD = $('.plot-host', pM);
  const drawModel = responsive(hD, () => {
    const d = model.data; if (!d) return;
    const yrs = d.dates.map(yearOf), [, hi] = extent(d.vix, d.model);
    const P = frame(hD, { h: 320, x: [yrs[0], yrs[yrs.length - 1]], y: [0, hi * 1.05], yl: 'VIX', xf: v => m.window === 'all' ? Math.round(v) : fmt(v, 1) });
    P.line(zip(yrs, d.vix), 's1 thin'); P.line(zip(yrs, d.model), 's2 thin'); P.hline(d.baseline);
    hover(hD, P, px => { const i = nearest(yrs, P.xInv(px)); return { cross: P.X(yrs[i]), pts: [[P.X(yrs[i]), P.Y(d.vix[i]), 'f1'], [P.X(yrs[i]), P.Y(d.model[i]), 'f2']], html: tip(fdate(d.dates[i]), [['VIX', fmt(d.vix[i], 1)], ['Model', fmt(d.model[i], 1)]]) }; });
  });
  function renderModel(d) {
    const w = d.windows;
    readouts($('.readout', pM), [['Doses', `${int(d.n_doses)} of ${int(d.n_days)} days`], ['R², all days', fmt(d.r2)], ['R² 2008 / 2020 / 2025', `${fmt(w['2008'])} / ${fmt(w['2020'])} / ${fmt(w['2025'])}`],
      d.compartments === 1 ? ['Half-life', fmt(d.half_life, 1) + ' d'] : ['Compartments', `${pct(d.a, 0)} at ${fmt(d.hl_fast, 1)} d, rest at ${fmt(d.hl_slow, 0)} d`]]);
    drawModel(); verdict();
  }
  function verdict() {
    const d = events.data, mm = model.data; if (!d || !mm) return;
    verdicts($('.verdict', sec), [
      ['How fast is fear cleared?', `<b>Fast, then slowly.</b> The median spike loses half its excess in about ${fmt(d.one.half_life, 0)} trading days; a whole-series AR(1) says ${fmt(d.ar1.half_life, 0)}, because it averages the two phases.`],
      ['One compartment or two?', `<b>${d.two.r2 > d.one.r2 + 0.1 ? 'Two.' : 'Hard to tell.'}</b> R² ${fmt(d.one.r2)} with one, ${fmt(d.two.r2)} with two: a fast phase (${fmt(d.two.hl_fast, 1)} d) and a slow tail, like a drug's distribution and elimination phases.`],
      ['Is the VIX just doses and decay?', `<b>Partly.</b> With these settings, doses on ${pct(mm.n_doses / mm.n_days, 1)} of days plus clearance explain ${pct(mm.r2, 0)} of VIX variation (2008: ${pct(mm.windows['2008'], 0)}).`],
    ]);
  }
  return [events, model];
}

/* ------------------------------------------------------------ C: crash shapes */
function shapes(root) {
  const sec = $('[data-study="shape"]', root);
  const p = { spy_depth: 0.10, stock_depth: 0.20, landmarks: 48 };
  const b = { b1: 0, b2: 0, b3: 0 };
  let sel = null;
  const pS = $('[data-panel="crash-model"]', sec), pP = $('[data-panel="crash-space"]', sec);
  const space = livePanel(pP, { request: sig => call('/api/crossover/crash-shapes', p, { signal: sig }), render: renderSpace, delay: 200 });
  const sample = livePanel(pS, { request: sig => call('/api/crossover/crash-sample', Object.assign({}, p, b), { signal: sig }), render: () => drawShape(), delay: 30 });
  const ch = () => { sel = null; space.refresh(); sample.refresh(); };
  bind(pP, p, 'spy_depth', ch, v => pct(v, 0)); bind(pP, p, 'stock_depth', ch, v => pct(v, 0)); bind(pP, p, 'landmarks', ch, v => v);
  const sl = ['b1', 'b2', 'b3'].map(k => bind(pS, b, k, () => sample.refresh(), v => (v > 0 ? '+' : '') + fmt(v, 1) + ' sd'));

  const hS = $('.plot-host', pS), hP = $('.plot-host', pP);
  const drawShape = responsive(hS, () => {
    const d = space.data, c = sample.data; if (!d || !c) return;
    const P = frame(hS, { h: 300, x: [0, 1], y: [-1.4, 0.3], yt: [-1, -0.5, 0], yf: v => Math.round(v * 100) + '%', xt: [0, 0.25, 0.5, 0.75, 1], xf: v => Math.round(v * 100) + '%', xl: 'Time from peak (0%) to recovery (100%)', yl: 'Drop, share of the crash depth' });
    P.hline(0); P.line(zip(d.s, d.mean), 'sn dashed');
    if (sel != null) P.line(zip(d.s, d.episodes[sel].curve), 's1');
    P.line(zip(c.s, c.curve), 's2');
  });
  const drawSpace = responsive(hP, () => {
    const d = space.data; if (!d) return;
    const xs = d.episodes.map(e => e.scores[0]), ys = d.episodes.map(e => e.scores[1]);
    const [x0, x1] = extent(xs, [-3, 3]), [y0, y1] = extent(ys, [-2.5, 2.5]);
    const P = frame(hP, { h: 340, x: [x0 - 0.3, x1 + 0.3], y: [y0 - 0.3, y1 + 0.3], xl: 'Mode 1: U-shaped ← → V-shaped', yl: 'Mode 2: early ↓ / late ↑ bottom' });
    P.vline(0); P.hline(0);
    const dots = d.episodes.map((e, i) => ({ i, e, x: P.X(e.scores[0]), y: P.Y(e.scores[1]) }));
    dots.slice().sort((a, c) => (a.e.ticker === 'SPY') - (c.e.ticker === 'SPY')).forEach(t => E('circle', { cx: t.x, cy: t.y, r: t.i === sel ? 7 : 5, class: (t.e.ticker === 'SPY' ? 'f1' : 'f3') + ' ring' + (t.i === sel ? ' sel' : '') }, P.body));
    const lab = { '2007-10-09': ['SPY 2008', 9, -8], '2020-02-19': ['SPY 2020', 9, 16], '2000-03-24': ['SPY dot-com', -9, -8, 'end'] };
    dots.forEach(t => { const L = t.e.ticker === 'SPY' && lab[t.e.peak]; if (L) P.text(t.x + L[1], t.y + L[2], L[0], L[3] ? { 'text-anchor': L[3] } : {}); });
    const near = (px, py) => { let best = null, bd = 400; dots.forEach(t => { const dd = (t.x - px) ** 2 + (t.y - py) ** 2; if (dd < bd) { bd = dd; best = t; } }); return best; };
    hover(hP, P, (px, py) => { const t = near(px, py); if (!t) return null; const e = t.e; return { pts: [[t.x, t.y, e.ticker === 'SPY' ? 'f1' : 'f3']], html: tip(`${e.ticker} ${e.peak.slice(0, 4)}–${e.recovered.slice(0, 4)}`, [['Depth', pct(e.depth, 0)], ['Days', int(e.days)], ['Modes', `${fmt(e.scores[0], 1)}, ${fmt(e.scores[1], 1)}`]]) + '<br>Click to load' }; },
      { onClick: (px, py) => { const t = near(px, py); if (!t) return; sel = t.i; t.e.scores.slice(0, 3).forEach((v, k) => sl[k].set(Math.max(-3, Math.min(3, +v.toFixed(1))))); describe(); sample.now(); drawSpace(); } });
  });
  function describe() {
    const e = space.data.episodes[sel];
    $('[data-out="sel"]', pS).innerHTML = `<b style="color:var(--ink)">${e.ticker}</b>: peak ${fdate(e.peak)}, bottom ${fdate(e.trough)}, recovered ${fdate(e.recovered)}. Depth ${pct(e.depth, 0)}, ${int(e.days)} trading days. Orange is this crash rebuilt from three modes.`;
  }
  function renderSpace(d) {
    $('[data-out="sel"]', pS).textContent = 'No crash selected. Click a dot on the right.';
    readouts($('.readout', pP), [['Crashes', d.episodes.length], ...d.variance.slice(0, 3).map((v, i) => [`Mode ${i + 1}`, pct(v, 0)]), ['First three', pct(d.cumulative[2], 0)]]);
    verdicts($('.verdict', sec), [
      ['Is there a typical crash?', `<b>A loose one.</b> Three modes explain ${pct(d.cumulative[2], 0)} of the variation, against 90% for ShapeLab's jawlines.`],
      ['What sets the shape?', `<b>Where the bottom falls.</b> Mode 2 tracks the timing of the trough (correlation ${fmt(d.correlations.mode2_trough_at)}). Mode 1 separates quick V's from long U's.`],
      ['Are deep crashes different?', `<b>${Math.abs(d.correlations.mode1_depth) > 0.2 ? 'Somewhat.' : 'Not clearly.'}</b> Mode 1 vs depth: ${fmt(d.correlations.mode1_depth)}${d.correlations.mode1_depth < 0 ? '; deeper crashes lean U-shaped' : ''}.`],
    ]);
    drawSpace(); drawShape();
  }
  return [space, sample];
}

export function initCrossover(root) {
  const all = [...randomWalk(root), ...dosing(root), ...shapes(root)];
  all.forEach(l => l.now());
  return all;
}
