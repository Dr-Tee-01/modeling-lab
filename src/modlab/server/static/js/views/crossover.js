// Pillar 4 · Crossover: market data through the spread and shape toolkits, as workbench specs.
import { call, fdate, fmt, fmtMs, int, mon, pct } from '../api.js';
import { extent, frame, hover, nearest, zip } from '../plot.js';
import { seg } from '../ui/dom.js';
import { S, edge, node, spark, text, widthFor } from '../ui/illustrate.js';

const tip = (title, rows) => `<b>${title}</b>` + rows.map(([k, v]) => `<br>${k}: <span class="n">${v}</span>`).join('');
const pick = (s, keys) => Object.fromEntries(keys.map(k => [k, s[k]]));
const merge = (a, b) => ({ compute_ms: (a.meta?.compute_ms || 0) + (b.meta?.compute_ms || 0), cached: !!(a.meta?.cached && b.meta?.cached) });
const yearOf = s => +s.slice(0, 4) + (+s.slice(5, 7) - 1) / 12 + (+s.slice(8, 10)) / 365;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const signed = (v, d = 1) => (v > 0 ? '+' : '') + fmt(v, d);
const every = (arr, n) => { const k = Math.max(1, Math.ceil(arr.length / n)); return arr.filter((_, i) => i % k === 0 || i === arr.length - 1); };
const normCdf = z => { const t = 1 / (1 + 0.2316419 * Math.abs(z)), q = 0.3989423 * Math.exp(-z * z / 2) * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274)))); return z > 0 ? 1 - q : q; };

/* Log-scale ticks for growth of $1: 1-2-5 steps, thinned to powers of ten on long ranges. */
function growthTicks(lo, hi) {
  const all = [];
  for (let e = -2; e <= 8; e++) for (const m of [1, 2, 5]) all.push(m * 10 ** e);
  let t = all.filter(v => Math.log(v) >= lo - 0.2 && Math.log(v) <= hi + 0.2);
  if (t.length > 7) t = t.filter(v => Math.abs(Math.log10(v) - Math.round(Math.log10(v))) < 1e-9);
  if (t.length > 7) t = t.filter((_, i) => i % 2 === 0);
  return t.map(Math.log);
}
const mult = v => { const x = Math.exp(v); return x >= 1e6 ? fmt(x / 1e6, 0) + 'M×' : x >= 1e4 ? fmt(x / 1e3, 0) + 'k×' : +x.toPrecision(2) + '×'; };

/* ================================================================ A: random walk */
const TICKERS = ['SPY', 'AAPL', 'MSFT', 'NVDA', 'JPM', 'XOM'];

/* The four tests, scored the same way for the tiles, the illustration and the insights. */
function tests(d) {
  const T = d.tails, r10 = d.robust.find(r => r.k === 10), A = d.acf_abs, M = d.drawdown;
  return [
    { name: 'Normal steps', fail: T.beyond4 > 3 * Math.max(1, T.expected4), sub: `${T.beyond4} days past 4σ vs ${fmt(T.expected4, 1)}` },
    { name: 'No mean reversion', fail: Math.abs(r10.z) > 1.96, sub: `VR(10) ${fmt(r10.vr, 2)}, robust z ${fmt(r10.z, 1)}` },
    { name: 'No memory', fail: A[0] > 2 * d.acf_band, sub: `|r| memory ${fmt(A[0], 2)} at lag 1` },
    { name: 'Crash size', fail: M.share_worse_normal < 0.05, sub: `${pct(M.share_worse_normal, 0)} of walks fall further` },
  ];
}

const randomWalk = {
  id: 'random-walk', group: 'crossover', nav: 'Is it a random walk?', tag: 'Labs 9–10', title: 'Is a stock the random walk from Lab 10?',
  question: "Give Lab 10's walker the stock's own drift and volatility. By eye the paths look alike. Which of four tests tell them apart?",
  params: [
    { k: 'ticker', type: 'seg', label: 'Stock', value: 'SPY', options: TICKERS.map(t => [t, t]), group: 'Data',
      help: 'Daily total-return prices from 1993 to 2026. SPY is the S&P 500 index fund.' },
    { k: 'n', label: 'Random walks shown', min: 5, max: 60, step: 1, value: 20, unit: 'paths', group: 'Data',
      help: "Each walk takes independent normal weekly steps with the stock's mean and standard deviation." },
    { k: 'sims', label: 'Monte Carlo histories', min: 200, max: 5000, step: 100, value: 2000, group: 'Tests',
      help: 'Simulated daily histories of the same length, for the drawdown test. More histories give a smoother distribution.' },
  ],
  hidden: { seed: 312, walk_seed: 1, tails: 'raw' },
  reset(ctx) { Object.assign(ctx.state, { seed: 312, walk_seed: 1, tails: 'raw' }); ctx.extra.tailsSeg?.set('raw'); },
  debounce: 250,
  presets: [
    { label: 'SPY', values: { ticker: 'SPY' } },
    { label: 'NVDA (volatile)', values: { ticker: 'NVDA' } },
    { label: 'JPM (bank)', values: { ticker: 'JPM' } },
    { label: 'XOM (energy)', values: { ticker: 'XOM' } },
  ],
  request: async (s, sig) => {
    const [a, b] = await Promise.all([
      call('/api/crossover/random-walk', { ticker: s.ticker, sims: s.sims, seed: s.seed }, { signal: sig }),
      call('/api/crossover/walks', { ticker: s.ticker, n: s.n, seed: s.walk_seed }, { signal: sig })]);
    return Object.assign(a, { paths: b, meta: merge(a, b) });
  },
  actions: [
    { label: 'Draw new walks', icon: 'dice', run(ctx) { ctx.state.walk_seed = Math.floor(Math.random() * 1e9); return ctx.now(); } },
    { label: 'New Monte Carlo seed', icon: 'dice', run(ctx) { ctx.state.seed = Math.floor(Math.random() * 1e9); return ctx.now(); } },
  ],
  kpis: d => {
    const t = tests(d), T = d.tails, r10 = d.robust.find(r => r.k === 10), A = d.acf_abs, M = d.drawdown;
    const tone = x => x.fail ? 'bad' : 'good';
    return [
      { label: 'Days beyond 4σ', value: int(T.beyond4), unit: 'days', note: `a normal walk expects ${fmt(T.expected4, 2)}`, tone: tone(t[0]) },
      { label: 'Variance ratio at 10 days', value: fmt(r10.vr, 2), note: `robust z ${fmt(r10.z, 1)}; a walk gives 1`, tone: tone(t[1]) },
      { label: 'Memory in |r| at lag 1', value: fmt(A[0], 2), note: `${fmt(A[99], 2)} at lag 100; band ±${fmt(d.acf_band, 2)}`, tone: tone(t[2]) },
      { label: 'Maximum drawdown', value: pct(M.real, 0), note: `${pct(M.share_worse_normal, 1)} of normal walks fall further`, tone: tone(t[3]) },
    ];
  },
  illustration: {
    title: 'A coin-flip walker against the real path', height: 220,
    caption: 'Grey: walks with the same weekly drift and spread. Each check turns green when the real data passes the test and red when it rejects the walk.',
    draw(svg, d, s) {
      S('rect', { x: 8, y: 30, width: 288, height: 150, rx: 10, class: 'node' }, svg);
      text(svg, 152, 20, `${s.ticker} vs ${s.n} random walks`, 'n-title');
      const box = [20, 42, 264, 126];
      if (d) {
        const P = d.paths, real = every(P.real, 160), walks = P.walks.slice(0, 6).map(w => every(w, 160));
        const xs = real.map((_, i) => i), yr = extent(real, ...walks);
        walks.forEach(w => spark(svg, xs, w, box, 'ln-m', yr));
        const { X, Y } = spark(svg, xs, real, box, 'st1', yr);
        text(svg, X(xs.length - 1) - 2, Y(real[real.length - 1]) - 8, s.ticker, 'edge-lbl', 'end');
        text(svg, 152, 198, `$1 → ${mult(Math.log(P.real_multiple))} real · walks ${mult(Math.log(P.walk_multiples[0]))} to ${mult(Math.log(P.walk_multiples[P.walk_multiples.length - 1]))}`, 'n-sub');
      } else {
        let x = 0, r = s.walk_seed * 9301 + 49297; const w = [0];
        for (let i = 0; i < 120; i++) { r = (r * 9301 + 49297) % 233280; x += r / 233280 < 0.5 ? -1 : 1; w.push(x); }
        spark(svg, w.map((_, i) => i), w, box, 'ln-m');
        text(svg, 152, 198, 'each week: a coin flip of ±σ around the drift', 'n-sub');
      }
      edge(svg, [300, 105], [328, 105], { width: 2 });
      const t = d ? tests(d) : [
        { name: 'Normal steps', sub: 'days beyond 4σ vs normal' }, { name: 'No mean reversion', sub: 'variance ratio = 1' },
        { name: 'No memory', sub: 'autocorrelation = 0' }, { name: 'Crash size', sub: 'drawdown vs simulations' }];
      t.forEach((c, i) => {
        const n = node(svg, { x: 334, y: 8 + i * 52, w: 258, h: 46, title: c.name, sub: c.sub, hot: d ? c.fail : false });
        const mark = !d ? '·' : c.fail ? '✗' : '✓';
        S('text', { x: n.x + 16, y: n.cy + 7, 'text-anchor': 'middle', class: 'n-val', style: `fill:var(--${!d ? 'mute' : c.fail ? 'bad' : 'good'})` }, svg).textContent = mark;
      });
    },
  },
  chartLayout: 'two',
  charts: [
    { id: 'paths', title: 'Growth of $1 next to random walks', height: 300,
      sub: (d, s) => `Weekly steps with the same drift (${pct(d.paths.weekly_mu, 2)}) and volatility (${pct(d.paths.weekly_sigma, 2)}). Log scale.`,
      legend: (d, s) => [['s1', 'line', `${s.ticker}, total return`], ['neutral', 'line', 'Random walks']],
      draw(host, d, s) {
        const P0 = d.paths, yrs = P0.dates.map(yearOf), [lo, hi] = extent(P0.real, ...P0.walks);
        const P = frame(host, { h: 300, x: [yrs[0], yrs[yrs.length - 1]], y: [Math.min(lo, Math.log(0.5)), hi + 0.1 * (hi - lo)], yt: growthTicks(Math.min(lo, Math.log(0.5)), hi), yf: mult, xf: v => Math.round(v), yl: 'Growth of $1' });
        P0.walks.forEach(w => P.line(zip(yrs, w), 'sn thin').setAttribute('opacity', '.45'));
        P.hline(0); P.line(zip(yrs, P0.real), 's1');
        hover(host, P, px => {
          const i = nearest(yrs, P.xInv(px)), col = P0.walks.map(w => w[i]);
          return { cross: P.X(yrs[i]), pts: [[P.X(yrs[i]), P.Y(P0.real[i]), 'f1']], html: tip(mon(P0.dates[i]), [[s.ticker, '$' + fmt(Math.exp(P0.real[i]), 2)], ['Walks', `$${fmt(Math.exp(Math.min(...col)), 2)} to $${fmt(Math.exp(Math.max(...col)), 2)}`]]) };
        });
      } },
    { id: 'tails', title: 'Fat tails', height: 300,
      sub: (d, s) => s.tails === 'raw' ? 'Daily moves in standard deviations, log count. The line is what a normal walk expects.' : 'Each day divided by the volatility known the evening before. The tails shrink but stay fat.',
      legend: [['s1', 'box', 'Real days'], ['s2', 'line', 'Normal random walk']],
      tools(el, ctx) { ctx.extra.tailsSeg = seg([['raw', 'Raw'], ['scaled', 'Volatility-scaled']], ctx.state.tails, v => { ctx.state.tails = v; ctx.redraw(); }, { label: 'Returns' }); el.append(ctx.extra.tailsSeg); },
      caption: (d, s) => {
        const T = d.tails, raw = s.tails === 'raw';
        return `${raw ? T.beyond4 : T.beyond4_scaled} days beyond 4σ where a normal walk expects ${fmt(raw ? T.expected4 : T.expected4_scaled, 2)}. Kurtosis ${fmt(raw ? T.kurtosis : T.kurtosis_scaled, 1)} (normal 3). Worst day ${fdate(T.worst_day)}, ${pct(T.worst_return)}.`;
      },
      draw(host, d, s) {
        const H = d.hist, raw = s.tails === 'raw', counts = raw ? H.raw : H.scaled, n = raw ? H.n_raw : H.n_scaled, ed = H.edges, bw = ed[1] - ed[0];
        const P = frame(host, { h: 300, x: [-8, 8], y: [0.3, 3000], ylog: true, yt: [1, 10, 100, 1000], yf: v => v >= 1000 ? v / 1000 + 'k' : v, xt: [-8, -6, -4, -2, 0, 2, 4, 6, 8], xf: v => v + 'σ', xl: 'Daily move in standard deviations', yl: 'Days (log scale)' });
        P.bars(ed, counts, 'f1', 0.3);
        const exp = []; for (let z = -8; z <= 8.001; z += 0.05) exp.push([z, Math.max(n * (normCdf(z + bw / 2) - normCdf(z - bw / 2)), 0.3)]);
        P.line(exp, 's2'); P.vline(-4); P.vline(4);
        hover(host, P, px => {
          const i = clamp(Math.floor((P.xInv(px) - ed[0]) / bw), 0, ed.length - 2), e = n * (normCdf(ed[i + 1]) - normCdf(ed[i]));
          return { html: tip(`${fmt(ed[i])}σ to ${fmt(ed[i + 1])}σ`, [['Days', counts[i]], ['Normal expects', e < 0.01 ? e.toExponential(1) : fmt(e, 2)]]) };
        });
      } },
    { id: 'vr', title: 'Variance ratio', height: 260,
      sub: 'Variance of k-day returns over k times the daily variance. A random walk gives 1. Below 1, moves partly reverse.',
      legend: [['s1', 'line', 'Real'], ['neutral', 'box', 'Shuffled days, 95%']],
      caption: d => {
        const r10 = d.robust.find(r => r.k === 10);
        return `Lo–MacKinlay robust z: ${d.robust.map(r => `${fmt(r.z, 1)} at ${r.k} d`).join(', ')}. After volatility scaling: ${fmt(r10.z_scaled, 1)} at 10 d. By decade at 10 d: ${d.decades.map(x => `${fmt(x.z10, 1)} (${x.span.replace('-', '–')})`).join(', ')}.`;
      },
      draw(host, d) {
        const V = d.variance_ratio, rob = Object.fromEntries(d.robust.map(r => [r.k, r]));
        const P = frame(host, { h: 260, x: [1, 63], xlog: true, xt: [1, 2, 5, 10, 21, 63], y: [Math.min(0.5, ...V.map(v => v.vr)) - 0.05, Math.max(1.3, ...V.map(v => v.hi)) + 0.05], xl: 'Horizon k, trading days (log)', yl: 'Variance ratio' });
        P.area(V.map(v => [v.k, v.hi]), V.map(v => [v.k, v.lo]), 'band-null'); P.hline(1);
        P.line(V.map(v => [v.k, v.vr]), 's1'); V.forEach(v => P.dot(v.k, v.vr, 'f1', 4));
        hover(host, P, px => {
          const xs = V.map(v => P.X(v.k)), i = nearest(xs, px), v = V[i], r = rob[v.k];
          return { cross: xs[i], pts: [[xs[i], P.Y(v.vr), 'f1']], html: tip(`k = ${v.k} days`, [['Ratio', fmt(v.vr, 3)], ['Shuffled 95%', `${fmt(v.lo)} to ${fmt(v.hi)}`]].concat(r ? [['Robust z', fmt(r.z, 2)], ['Scaled z', fmt(r.z_scaled, 2)]] : [])) };
        });
      } },
    { id: 'acf', title: 'Memory in returns and their size', height: 260,
      sub: 'Autocorrelation by lag. Inside the grey band there is no detectable memory.',
      legend: [['s1', 'line', 'Size |r|'], ['s2', 'line', 'Return r'], ['neutral', 'box', 'No-memory band']],
      draw(host, d) {
        const A = d.acf_abs, R = d.acf_r, L = A.map((_, i) => i + 1), b = d.acf_band;
        const P = frame(host, { h: 260, x: [1, L.length], y: [Math.min(-0.1, ...R) - 0.02, Math.max(0.35, ...A) + 0.02], xl: 'Lag, trading days', yl: 'Autocorrelation' });
        P.area([[1, b], [L.length, b]], [[1, -b], [L.length, -b]], 'band-null'); P.hline(0);
        P.line(zip(L, R), 's2 thin'); P.line(zip(L, A), 's1');
        hover(host, P, px => { const i = nearest(L, P.xInv(px)); return { cross: P.X(L[i]), pts: [[P.X(L[i]), P.Y(A[i]), 'f1'], [P.X(L[i]), P.Y(R[i]), 'f2']], html: tip(`Lag ${L[i]} days`, [['|r|', fmt(A[i], 3)], ['r', fmt(R[i], 3)], ['Band', '±' + fmt(b, 3)]]) }; });
      } },
    { id: 'mdd', title: 'Worst crash in a simulated history', height: 260,
      sub: d => `Maximum drawdown in ${int(d.drawdown.sims)} simulated histories as long as the real one (${d.start.slice(0, 4)}–${d.end.slice(0, 4)}).`,
      legend: (d, s) => [['neutral', 'box', 'Normal walks'], ['s3', 'line', 'Resampled real days'], ['s1', 'line', `${s.ticker} real`]],
      caption: d => { const M = d.drawdown; return `Median simulated drawdown ${pct(M.median_normal, 0)} (normal) and ${pct(M.median_bootstrap, 0)} (resampled). ${pct(M.share_worse_normal, 1)} and ${pct(M.share_worse_bootstrap, 1)} go deeper than the real ${pct(M.real, 0)}.`; },
      draw(host, d, s) {
        const M = d.drawdown, bins = M.bins, hs = M.hist_normal, hb = M.hist_bootstrap, bw = bins[1] - bins[0];
        const P = frame(host, { h: 260, x: [0, 1], y: [0, Math.max(...hs, ...hb) * 1.12], xf: v => Math.round(v * 100) + '%', xl: 'Maximum drawdown over the whole period', yl: 'Simulations' });
        P.bars(bins, hs, 'fn', 0);
        P.line(bins.slice(0, -1).map((x, i) => [x + bw / 2, hb[i]]), 's3');
        P.vline(M.real, 'ln s1');
        P.text(P.X(M.real) + (M.real > 0.75 ? -6 : 6), P.M.t + 12, `${s.ticker} real: ${pct(M.real, 0)}`, M.real > 0.75 ? { 'text-anchor': 'end' } : {});
        hover(host, P, px => { const i = clamp(Math.floor(P.xInv(px) / bw), 0, hs.length - 1); return { html: tip(`${Math.round(bins[i] * 100)}–${Math.round(bins[i + 1] * 100)}%`, [['Normal walks', hs[i]], ['Resampled real days', hb[i]]]) }; });
      } },
    { id: 'kurt', title: 'Tails thin out with the horizon', height: 260,
      sub: 'Kurtosis of k-day returns. Sums of many days drift toward the normal value of 3, as the central limit theorem says.',
      legend: [['s1', 'line', 'Real'], ['neutral', 'dash', 'Normal, 3']],
      draw(host, d) {
        const K = d.kurtosis_by_horizon;
        const P = frame(host, { h: 260, x: [1, 63], xlog: true, xt: K.map(k => k.k), y: [0, Math.max(...K.map(k => k.kurtosis)) * 1.12], xl: 'Horizon k, trading days (log)', yl: 'Kurtosis' });
        P.hline(3); P.line(K.map(k => [k.k, k.kurtosis]), 's1'); K.forEach(k => P.dot(k.k, k.kurtosis, 'f1', 4));
        hover(host, P, px => {
          const xs = K.map(k => P.X(k.k)), i = nearest(xs, px), k = K[i];
          return { cross: xs[i], pts: [[xs[i], P.Y(k.kurtosis), 'f1']], html: tip(`${k.k}-day returns`, [['Kurtosis', fmt(k.kurtosis, 2)], ['Samples', int(k.count)]]) };
        });
      } },
  ],
  insights: (d, s) => {
    const T = d.tails, K = d.kurtosis_by_horizon, A = d.acf_abs, M = d.drawdown, z = k => d.robust.find(r => r.k === k).z, dec = d.decades;
    const weaker = Math.abs(dec[dec.length - 1].z10) < Math.abs(dec[0].z10);
    return [
      `<b>The steps are not normal.</b> ${s.ticker} has ${T.beyond4} days beyond 4σ where a normal walk expects ${fmt(T.expected4, 2)}. Kurtosis falls from ${fmt(K[0].kurtosis, 1)} for daily returns to ${fmt(K[K.length - 1].kurtosis, 1)} for quarterly ones, on its way to the normal 3.`,
      `<b>The steps are not independent.</b> Big days cluster: |r| autocorrelation is ${fmt(A[0], 2)} at lag 1 and still ${fmt(A[99], 2)} at lag 100. Variance ratios below 1 give a robust z of ${fmt(z(5), 1)} at 5 days and ${fmt(z(10), 1)} at 10 days${weaker ? `, weaker since ${dec[dec.length - 1].span.slice(0, 4)} (${fmt(dec[dec.length - 1].z10, 1)})` : ''}.`,
      M.share_worse_normal >= 0.05
        ? `<b>Crash size alone does not reject the walk.</b> ${pct(M.share_worse_normal, 0)} of ${int(M.sims)} normal walks have a deeper drawdown than the real ${pct(M.real, 0)}.`
        : `<b>The walk understates crashes.</b> Only ${pct(M.share_worse_normal, 1)} of ${int(M.sims)} normal walks fall as far as the real ${pct(M.real, 0)}.`,
    ];
  },
  maths: { eq: 'random walk     log Pₜ = log Pₜ₋₁ + μ + σ εₜ,   εₜ ~ N(0, 1) independent\nvariance ratio  VR(k) = Var(rₜ + … + rₜ₋ₖ₊₁) / (k Var(rₜ)),   1 for a walk\nmemory          ρ(ℓ) = corr(rₜ, rₜ₋ℓ),   band ±1.96 / √n\ndrawdown        MDD = maxₜ (1 − Pₜ / maxₛ≤ₜ Pₛ)',
    notes: ['The shuffled band comes from 200 random orderings of the real days: same distribution, no order. The Lo–MacKinlay z is robust to volatility that changes over time.',
      'Normal histories draw each day from N(μ, σ²). Resampled histories draw real days with replacement, so they keep the fat tails but not the clustering.'] },
};

/* ================================================================ B: fear as a dose */
const doseQuery = s => Object.assign({ compartments: s.compartments, threshold: s.threshold, window: s.window },
  s.compartments === '1' ? { half_life: s.half_life } : { a: s.a, hl_fast: s.hl_fast, hl_slow: s.hl_slow });
const fitted = f => f.compartments === 1 ? { half_life: clamp(+f.half_life.toFixed(1), 1, 60) }
  : { a: clamp(+f.a.toFixed(2), 0.05, 0.95), hl_fast: clamp(+f.hl_fast.toFixed(1), 0.5, 20), hl_slow: clamp(Math.round(f.hl_slow), 5, 200) };
const hlSlow = T => T.hl_slow == null ? '∞' : fmt(T.hl_slow, 0);

const dose = {
  id: 'dose', group: 'crossover', nav: 'Fear as a drug dose', tag: 'Lab 2', title: 'Does fear clear like a drug dose?',
  question: 'Treat each VIX spike as a dose of fear. How fast is it cleared, and does one compartment describe it?',
  params: [
    { k: 'ratio', label: 'Spike threshold', min: 1.3, max: 2.5, step: 0.05, value: 1.6, unit: '×', group: 'What counts as a spike',
      help: 'A spike is a VIX peak at least this many times its prior 3-month median.' },
    { k: 'gap', label: 'Minimum gap between spikes', min: 21, max: 126, step: 1, value: 63, unit: 'days', group: 'What counts as a spike',
      help: 'Peaks closer than this are merged into one spike, keeping the higher one.' },
    { k: 'compartments', type: 'seg', label: 'Clearance model', value: '1', options: [['1', 'One compartment'], ['2', 'Two compartments']], group: 'Rebuild the VIX',
      help: 'One compartment: the excess decays at one rate. Two: each dose splits between a fast and a slow tank.' },
    { k: 'half_life', label: 'Half-life', min: 1, max: 60, step: 0.1, value: 12.5, unit: 'days', group: 'Rebuild the VIX', show: s => s.compartments === '1',
      help: 'Trading days for the excess over the median to halve.' },
    { k: 'a', label: 'Share in the fast tank', min: 0.05, max: 0.95, step: 0.01, value: 0.77, scale: 100, unit: '%', group: 'Rebuild the VIX', show: s => s.compartments === '2',
      help: 'Fraction of each dose that clears quickly. The rest clears slowly.' },
    { k: 'hl_fast', label: 'Fast half-life', min: 0.5, max: 20, step: 0.1, value: 5.9, unit: 'days', group: 'Rebuild the VIX', show: s => s.compartments === '2' },
    { k: 'hl_slow', label: 'Slow half-life', min: 5, max: 200, step: 1, value: 44, unit: 'days', group: 'Rebuild the VIX', show: s => s.compartments === '2' },
    { k: 'threshold', label: 'Dose threshold', min: 1, max: 8, step: 0.1, value: 3.4, unit: 'pts', group: 'Rebuild the VIX',
      help: 'Only daily VIX jumps larger than this count as doses.' },
    { k: 'window', type: 'seg', label: 'Show', value: 'all', options: [['all', '1993–2026'], ['2008', '2008'], ['2020', '2020'], ['2025', '2025']], group: 'Rebuild the VIX',
      help: 'Zoom the chart on one crisis. The fit always uses every day.' },
  ],
  hidden: { fit: 0 },
  presets: [
    { label: 'Default spikes (×1.6)', values: { ratio: 1.6, gap: 63 } },
    { label: 'Only big spikes (×2)', values: { ratio: 2, gap: 63 } },
    { label: 'One compartment, fitted', note: 'Fits the half-life by least squares on the server', values: { compartments: '1', fit: 1 } },
    { label: 'Two compartments', values: { compartments: '2', a: 0.77, hl_fast: 5.9, hl_slow: 44 } },
  ],
  request: async (s, sig, ctx) => {
    const [e, m] = await Promise.all([
      call('/api/crossover/vix-events', { ratio: s.ratio, gap: s.gap, horizon: 60 }, { signal: sig }),
      call('/api/crossover/vix-model', Object.assign(doseQuery(s), s.fit ? { fit: true } : {}), { signal: sig })]);
    if (s.fit && !sig.aborted) ctx.set({ ...fitted(m), fit: 0 }, { run: false });
    return { events: e, model: m, meta: merge(e, m) };
  },
  actions: [{ label: 'Fit clearance', icon: 'fit', primary: true, async run(ctx, ui) {
    ui.note('Fitting on the server…');
    const f = await call('/api/crossover/vix-model', Object.assign(doseQuery(ctx.state), { fit: true, points: 100 }));
    ctx.set(fitted(f), { immediate: true });
    ui.note(f.compartments === 1
      ? `Least squares: half-life ${fmt(f.half_life, 2)} days, R² ${fmt(f.r2, 3)} (${fmtMs(f.meta.compute_ms)}).`
      : `Least squares: ${pct(f.a, 0)} at ${fmt(f.hl_fast, 1)} d, the rest at ${fmt(f.hl_slow, 0)} d, R² ${fmt(f.r2, 3)} (${fmtMs(f.meta.compute_ms)}).`);
  } }],
  kpis: d => {
    const E = d.events, M = d.model, w = M.windows;
    return [
      { label: 'Median spike half-life', value: fmt(E.one.half_life, 1), unit: 'days', note: `one compartment, R² ${fmt(E.one.r2)}`, tone: E.one.r2 > 0.9 ? 'good' : 'warn' },
      { label: 'Two compartments', value: `${pct(E.two.a, 0)} at ${fmt(E.two.hl_fast, 1)}`, unit: 'days', note: `the rest at ${hlSlow(E.two)} d, R² ${fmt(E.two.r2)}`, tone: E.two.r2 > 0.9 ? 'good' : 'warn' },
      { label: 'Whole-series AR(1) half-life', value: fmt(E.ar1.half_life, 0), unit: 'days', note: 'averages the fast and slow phases' },
      { label: 'VIX rebuilt from doses', value: fmt(M.r2, 2), unit: 'R²', note: `2008 ${fmt(w['2008'])} · 2020 ${fmt(w['2020'])} · 2025 ${fmt(w['2025'])}`, tone: M.r2 > 0.7 ? 'good' : M.r2 > 0.4 ? 'warn' : 'bad' },
    ];
  },
  illustration: {
    title: 'Where a dose of fear goes', height: 236,
    caption: 'Compartment values are least-squares fits to the median spike. Arrow widths follow the shares and clearance rates.',
    draw(svg, d, s) {
      const E = d?.events, two = s.compartments === '2';
      const spike = node(svg, { x: 6, y: 72, w: 120, h: 72, title: 'VIX spike', value: `×${fmt(s.ratio, 2)}`, sub: E ? `${E.events.length} spikes` : 'prior median', hot: true });
      const base = node(svg, { x: 478, y: 72, w: 116, h: 72, title: 'Baseline', sub: '3-month median' });
      if (!two) {
        const O = E?.one;
        const c = node(svg, { x: 226, y: 62, w: 150, h: 92, title: 'One compartment', value: O ? `t½ ${fmt(O.half_life, 1)} d` : '…', sub: O ? `R² ${fmt(O.r2)}` : '', fill: 0.5, fillCls: 'fill2' });
        edge(svg, [spike.r + 4, spike.cy], [c.x - 6, c.cy], { label: 'dose', width: 5, flow: true });
        edge(svg, [c.r + 4, c.cy], [base.x - 6, base.cy], { label: O ? `k = ${fmt(O.k, 3)}/d` : 'k', width: widthFor(O ? O.k : 0.1, 0.4), flow: true });
      } else {
        const T = E?.two, a = T ? T.a : s.a;
        const fast = node(svg, { x: 226, y: 8, w: 150, h: 80, title: 'Fast', value: T ? `t½ ${fmt(T.hl_fast, 1)} d` : '…', sub: `${pct(a, 0)} of the dose`, fill: a, fillCls: 'fill3' });
        const slow = node(svg, { x: 226, y: 130, w: 150, h: 80, title: 'Slow', value: T ? `t½ ${hlSlow(T)} d` : '…', sub: `${pct(1 - a, 0)} of the dose`, fill: 1 - a, fillCls: 'fill3' });
        edge(svg, [spike.r + 4, spike.cy - 10], [fast.x - 6, fast.cy], { label: pct(a, 0), labelAt: [160, 62], width: widthFor(a, 1), flow: true });
        edge(svg, [spike.r + 4, spike.cy + 10], [slow.x - 6, slow.cy], { label: pct(1 - a, 0), labelAt: [160, 164], width: widthFor(1 - a, 1), flow: true });
        edge(svg, [fast.r + 4, fast.cy], [base.x - 6, base.cy - 10], { label: T ? `${fmt(T.k_fast, 2)}/d` : 'k_f', labelAt: [438, 62], width: widthFor(T ? T.k_fast : 0.3, 0.4), flow: true });
        edge(svg, [slow.r + 4, slow.cy], [base.x - 6, base.cy + 10], { label: T ? `${fmt(T.k_slow, 3)}/d` : 'k_s', labelAt: [438, 164], width: widthFor(T ? T.k_slow : 0.02, 0.4), flow: true });
      }
      const m = two ? `${pct(s.a, 0)} at ${fmt(s.hl_fast, 1)} d + ${fmt(s.hl_slow, 0)} d` : `t½ ${fmt(s.half_life, 1)} d`;
      text(svg, 300, 228, `VIX model: doses above +${fmt(s.threshold, 1)} pts, ${m}${d ? `, R² ${fmt(d.model.r2)}` : ''}`, 'n-sub');
    },
  },
  charts: [
    { id: 'clear', title: 'Clearance of the median spike', height: 290,
      sub: (d, s) => `${d.events.events.length} spikes at least ×${fmt(s.ratio, 2)} the prior 3-month median, ${s.gap}+ days apart. Each is scaled so the peak is 100% and the prior median 0%.`,
      legend: [['s1', 'line', 'Median spike'], ['s1', 'box', 'Middle 50%'], ['s2', 'dash', 'One compartment'], ['s3', 'line', 'Two compartments']],
      draw(host, d) {
        const E = d.events, h = E.h, lo = Math.min(-0.2, ...E.q25);
        const P = frame(host, { h: 290, x: [0, h[h.length - 1]], y: [lo, 1.05], yt: [0, 0.25, 0.5, 0.75, 1], yf: v => Math.round(v * 100) + '%', xl: 'Trading days after the peak', yl: 'Share of the spike left' });
        P.area(zip(h, E.q75), zip(h, E.q25), 'band'); P.hline(0); P.hline(0.5);
        P.line(zip(h, E.one.curve), 's2 dashed'); P.line(zip(h, E.two.curve), 's3'); P.line(zip(h, E.median), 's1');
        P.text(P.w - P.M.r - 4, P.Y(0.5) - 5, 'half cleared', { 'text-anchor': 'end' });
        hover(host, P, px => {
          const i = nearest(h, P.xInv(px));
          return { cross: P.X(h[i]), pts: [[P.X(h[i]), P.Y(E.median[i]), 'f1']], html: tip(`Day ${h[i]}`, [['Median spike', pct(E.median[i], 0)], ['Middle 50%', `${pct(E.q25[i], 0)} to ${pct(E.q75[i], 0)}`], ['One compartment', pct(E.one.curve[i], 0)], ['Two compartments', pct(E.two.curve[i], 0)]]) };
        });
      } },
    { id: 'vix', title: 'Rebuild the VIX from doses and clearance', height: 320,
      sub: (d, s) => `Only daily jumps above +${fmt(s.threshold, 1)} points count as doses (${int(d.model.n_doses)} of ${int(d.model.n_days)} days). Between doses the excess over the median ${fmt(d.model.baseline, 1)} decays${s.compartments === '1' ? ` with a ${fmt(s.half_life, 1)}-day half-life` : ' in two tanks'}.`,
      legend: [['s1', 'line', 'VIX'], ['s2', 'line', 'Doses + clearance'], ['neutral', 'dash', 'Median']],
      caption: d => `R² on all days ${fmt(d.model.r2)}. 2008: ${fmt(d.model.windows['2008'])}, 2020: ${fmt(d.model.windows['2020'])}, 2025: ${fmt(d.model.windows['2025'])}.${d.model.fitted ? ' Clearance fitted by least squares.' : ' Press Fit clearance to let the server choose the clearance.'}`,
      draw(host, d, s) {
        const M = d.model, yrs = M.dates.map(yearOf), [, hi] = extent(M.vix, M.model);
        const P = frame(host, { h: 320, x: [yrs[0], yrs[yrs.length - 1]], y: [0, hi * 1.05], yl: 'VIX', xf: v => s.window === 'all' ? Math.round(v) : fmt(v, 1) });
        P.hline(M.baseline); P.line(zip(yrs, M.vix), 's1 thin'); P.line(zip(yrs, M.model), 's2 thin');
        hover(host, P, px => { const i = nearest(yrs, P.xInv(px)); return { cross: P.X(yrs[i]), pts: [[P.X(yrs[i]), P.Y(M.vix[i]), 'f1'], [P.X(yrs[i]), P.Y(M.model[i]), 'f2']], html: tip(fdate(M.dates[i]), [['VIX', fmt(M.vix[i], 1)], ['Model', fmt(M.model[i], 1)]]) }; });
      } },
    { id: 'events', kind: 'html', title: 'Every spike',
      sub: (d, s) => `Days until each spike had lost half its excess over the prior median. More than 60 means it never did in the window.`,
      draw(host, d) {
        const E = d.events, open = host.querySelector('details')?.open;
        host.style.minHeight = '0';
        const rows = E.events.map(x => `<tr><td>${fdate(x.date)}</td><td class="num">${fmt(x.peak, 1)}</td><td class="num">${fmt(x.base, 1)}</td><td class="num">×${fmt(x.peak / x.base, 2)}</td><td class="num">${x.days_to_half ?? '> ' + E.params.horizon}</td></tr>`).join('');
        host.innerHTML = `<details class="more"${open ? ' open' : ''}><summary>Show all ${E.events.length} spikes</summary><div class="tbl-wrap"><table><thead><tr><th>Peak</th><th class="num">VIX</th><th class="num">Prior median</th><th class="num">Ratio</th><th class="num">Days to half</th></tr></thead><tbody>${rows}</tbody></table></div></details>`;
      } },
  ],
  insights: d => {
    const E = d.events, M = d.model, better = E.two.r2 > E.one.r2 + 0.1;
    return [
      `The median spike loses half its excess in about <b>${fmt(E.one.half_life, 1)} trading days</b>. An AR(1) fitted to the whole series says ${fmt(E.ar1.half_life, 0)} days, because it averages a fast phase and a slow one.`,
      `<b>${better ? 'Two compartments fit much better.' : 'One compartment is nearly as good.'}</b> R² is ${fmt(E.one.r2)} with one and ${fmt(E.two.r2)} with two: ${pct(E.two.a, 0)} of the excess clears with a ${fmt(E.two.hl_fast, 1)}-day half-life and the rest over ${hlSlow(E.two)} days, like a drug's distribution and elimination phases.`,
      `Doses on <b>${pct(M.n_doses / M.n_days, 1)}</b> of days plus clearance explain ${pct(M.r2, 0)} of the VIX's variation (2008: ${pct(M.windows['2008'], 0)}, 2025: ${pct(M.windows['2025'], 0)}). The rest is fear that builds slowly, without a single jump.`,
    ];
  },
  maths: { eq: 'one compartment   E(t) = e^(−k t),   t½ = ln 2 / k\ntwo compartments  E(t) = a e^(−k_f t) + (1 − a) e^(−k_s t)\nVIX model         xₜ = m + eₜ,   eₜ = e^(−k) eₜ₋₁ + dₜ\n                  dₜ = Δxₜ if Δxₜ > threshold, else 0',
    notes: ["E(t) is the share of a spike's excess over its prior 3-month median left t trading days after the peak. Both curves are least-squares fits to the median.",
      'In the VIX model m is the median of the whole series. With two compartments each dose splits a : 1 − a between a fast and a slow tank, each with its own e^(−k).'] },
};

/* ================================================================ C: crash shapes */
const DEF = ['spy_depth', 'stock_depth', 'landmarks'];
const spyMain = d => d.episodes.reduce((best, e, i) => e.ticker === 'SPY' && (best < 0 || e.depth > d.episodes[best].depth) ? i : best, -1);
const describe = e => `<b>${e.ticker}</b>: peak ${fdate(e.peak)}, bottom ${fdate(e.trough)}, recovered ${fdate(e.recovered)}. Depth ${pct(e.depth, 0)}, ${int(e.days)} trading days. Orange is this crash rebuilt from three modes.`;

const crashShapes = {
  id: 'crash-shapes', group: 'crossover', nav: 'Crash shapes', tag: 'ShapeLab', title: 'Do market crashes have a shape?',
  question: 'Stretch every completed crash to the same length and depth, then run the PCA that ShapeLab uses for jawlines. Is there a typical crash?',
  params: [
    { k: 'spy_depth', label: 'SPY depth at least', min: 0.05, max: 0.4, step: 0.01, value: 0.1, scale: 100, unit: '%', group: 'What counts as a crash',
      help: 'A completed SPY drawdown, from a peak back to that peak, at least this deep.' },
    { k: 'stock_depth', label: 'Single-stock depth at least', min: 0.1, max: 0.6, step: 0.01, value: 0.2, scale: 100, unit: '%', group: 'What counts as a crash',
      help: 'Single stocks swing more, so they need a deeper fall to count.' },
    { k: 'landmarks', label: 'Landmarks', min: 16, max: 128, step: 4, value: 48, unit: 'points', group: 'What counts as a crash',
      help: 'Points sampled along each normalised crash before the PCA.' },
    { k: 'b1', label: 'Mode 1 · U ↔ V', min: -3, max: 3, step: 0.1, value: 0, unit: 'sd', group: 'Shape model', help: 'Negative: a long U with a flat bottom. Positive: a sharp V.' },
    { k: 'b2', label: 'Mode 2 · early ↔ late bottom', min: -3, max: 3, step: 0.1, value: 0, unit: 'sd', group: 'Shape model', help: 'Moves the trough earlier or later in the crash.' },
    { k: 'b3', label: 'Mode 3 · double dip ↔ single bottom', min: -3, max: 3, step: 0.1, value: 0, unit: 'sd', group: 'Shape model', help: 'A second dip against a single bottom.' },
  ],
  debounce: 60,
  reset(ctx) { ctx.extra.sel = null; },
  presets: [
    { label: 'Default thresholds', values: { spy_depth: 0.1, stock_depth: 0.2, landmarks: 48 } },
    { label: 'Only deep crashes', values: { spy_depth: 0.2, stock_depth: 0.35, landmarks: 48 } },
    { label: 'Fewer landmarks (16)', values: { spy_depth: 0.1, stock_depth: 0.2, landmarks: 16 } },
  ],
  request: async (s, sig, ctx) => {
    const def = pick(s, DEF), key = JSON.stringify(def);
    if (ctx.extra.key !== key) { ctx.extra.key = key; ctx.extra.sel = null; }
    const [a, b] = await Promise.all([
      call('/api/crossover/crash-shapes', def, { signal: sig }),
      call('/api/crossover/crash-sample', pick(s, [...DEF, 'b1', 'b2', 'b3']), { signal: sig })]);
    return Object.assign(a, { sample: b, meta: merge(a, b) });
  },
  kpis: d => {
    const n = d.episodes.length, spy = d.episodes.filter(e => e.ticker === 'SPY').length, C = d.correlations, v = d.variance;
    return [
      { label: 'Completed crashes', value: int(n), note: `${spy} SPY · ${n - spy} single stocks` },
      { label: 'Three modes explain', value: pct(d.cumulative[2], 0), note: `mode 1 ${pct(v[0], 0)} · mode 2 ${pct(v[1], 0)} · mode 3 ${pct(v[2], 0)}`, tone: d.cumulative[2] > 0.8 ? 'good' : 'warn' },
      { label: 'Mode 2 vs trough timing', value: fmt(C.mode2_trough_at, 2), unit: 'r', note: 'where the bottom falls', tone: C.mode2_trough_at > 0.5 ? 'good' : null },
      { label: 'Mode 1 vs depth', value: fmt(C.mode1_depth, 2), unit: 'r', note: C.mode1_depth < -0.2 ? 'deeper crashes lean U-shaped' : C.mode1_depth > 0.2 ? 'deeper crashes lean V-shaped' : 'depth barely sets the shape' },
    ];
  },
  illustration: {
    title: 'From a drawdown to a shape', height: 216,
    caption: 'Time is stretched to 0–1 and the fall scaled to 1, then sampled at evenly spaced landmarks. Right: the mean ± 2 sd along each mode.',
    draw(svg, d, s, ctx) {
      const W = 160, Y0 = 30, H = 140, xs = [8, 220, 432];
      const sel = ctx?.extra?.sel, ep = d ? d.episodes[sel ?? spyMain(d)] : null;
      const L = Math.round(s.landmarks), sv = d ? d.s : Array.from({ length: L }, (_, i) => i / (L - 1));
      const curve = ep ? ep.curve : sv.map(t => -Math.pow(Math.sin(Math.PI * Math.pow(t, 0.75)), 1.4));
      xs.forEach(x => S('rect', { x, y: Y0, width: W, height: H, rx: 10, class: 'node' }, svg));
      text(svg, xs[0] + W / 2, 20, 'Raw drawdown', 'n-title'); text(svg, xs[1] + W / 2, 20, 'Unit box', 'n-title'); text(svg, xs[2] + W / 2, 20, 'Modes', 'n-title');
      // 1. raw: price relative to the peak
      const depth = ep ? ep.depth : 0.34, price = curve.map(y => 1 + depth * y);
      const r = spark(svg, sv, price, [xs[0] + 12, Y0 + 22, W - 24, H - 48], 'st1', [1 - depth, 1]);
      const iMin = curve.indexOf(Math.min(...curve));
      [[0, 'peak'], [iMin, 'trough'], [curve.length - 1, 'back']].forEach(([i, lab], k) => {
        S('circle', { cx: r.X(i), cy: r.Y(price[i]), r: 3.5, class: 'fill2' }, svg);
        text(svg, r.X(i) + (k === 0 ? 4 : k === 2 ? -4 : 0), k === 1 ? r.Y(price[i]) + 16 : r.Y(price[i]) - 7, lab, 'n-sub', k === 0 ? 'start' : k === 2 ? 'end' : 'middle');
      });
      text(svg, xs[0] + W / 2, Y0 + H + 18, ep ? `${ep.ticker} ${ep.peak.slice(0, 4)}: −${pct(ep.depth, 0)}, ${int(ep.days)} days` : 'any depth, any length', 'n-sub');
      edge(svg, [xs[0] + W + 6, Y0 + H / 2], [xs[1] - 8, Y0 + H / 2], { width: 2, label: 'scale', labelAt: [(xs[0] + W + xs[1]) / 2, Y0 + H / 2 - 10] });
      // 2. unit box with landmarks
      const bx = [xs[1] + 22, Y0 + 16, W - 36, H - 40];
      S('rect', { x: bx[0], y: bx[1], width: bx[2], height: bx[3], fill: 'none', class: 'ln-m', 'stroke-dasharray': '3 3' }, svg);
      const u = spark(svg, sv, curve, bx, 'st2', [-1, 0]);
      const rr = L > 64 ? 1.4 : L > 32 ? 2 : 2.8;
      curve.forEach((y, i) => S('circle', { cx: u.X(i), cy: u.Y(y), r: rr, class: 'fill1' }, svg));
      text(svg, bx[0] - 4, bx[1] + 4, '0', 'n-sub', 'end'); text(svg, bx[0] - 4, bx[1] + bx[3] + 4, '−1', 'n-sub', 'end');
      text(svg, bx[0], bx[1] + bx[3] + 15, '0', 'n-sub'); text(svg, bx[0] + bx[2], bx[1] + bx[3] + 15, '1', 'n-sub');
      text(svg, xs[1] + W / 2, Y0 + H + 18, `${L} landmarks`, 'n-sub');
      edge(svg, [xs[1] + W + 6, Y0 + H / 2], [xs[2] - 8, Y0 + H / 2], { width: 2, label: 'PCA', labelAt: [(xs[1] + W + xs[2]) / 2, Y0 + H / 2 - 10] });
      // 3. modes
      if (d) {
        for (let k = 0; k < 3; k++) {
          const y = Y0 + 8 + k * 44, plus = d.mean.map((m, i) => m + 2 * d.sd[k] * d.modes[k][i]), minus = d.mean.map((m, i) => m - 2 * d.sd[k] * d.modes[k][i]);
          const yr = extent(plus, minus);
          spark(svg, d.s, minus, [xs[2] + 64, y + 4, W - 74, 30], 'st2', yr);
          spark(svg, d.s, plus, [xs[2] + 64, y + 4, W - 74, 30], 'st1', yr);
          text(svg, xs[2] + 10, y + 18, `mode ${k + 1}`, 'n-sub', 'start');
          text(svg, xs[2] + 10, y + 33, pct(d.variance[k], 0), 'edge-lbl', 'start');
        }
        text(svg, xs[2] + W / 2, Y0 + H + 18, `3 modes: ${pct(d.cumulative[2], 0)}`, 'n-sub');
      } else text(svg, xs[2] + W / 2, Y0 + H / 2, 'fitting…', 'n-sub');
    },
  },
  chartLayout: 'two',
  charts: [
    { id: 'model', title: 'The crash shape model', height: 300,
      sub: 'Move along each mode with the sliders, or click a crash in shape space to load it.',
      legend: [['neutral', 'dash', 'Mean'], ['s2', 'line', 'Model'], ['s1', 'line', 'Selected crash']],
      caption: (d, s, ctx) => { const i = ctx?.extra?.sel; return i != null && d.episodes[i] ? describe(d.episodes[i]) : 'No crash selected. Click a dot in shape space.'; },
      draw(host, d, s, ctx) {
        const c = d.sample, sel = ctx.extra.sel, ep = sel != null ? d.episodes[sel] : null;
        const P = frame(host, { h: 300, x: [0, 1], y: [-1.4, 0.3], yt: [-1, -0.5, 0], yf: v => Math.round(v * 100) + '%', xt: [0, 0.25, 0.5, 0.75, 1], xf: v => Math.round(v * 100) + '%', xl: 'Time from peak (0%) to recovery (100%)', yl: 'Drop, share of the crash depth' });
        P.hline(0); P.line(zip(d.s, d.mean), 'sn dashed');
        if (ep) P.line(zip(d.s, ep.curve), 's1');
        P.line(zip(c.s, c.curve), 's2');
        hover(host, P, px => {
          const i = nearest(d.s, P.xInv(px));
          return { cross: P.X(d.s[i]), pts: [[P.X(d.s[i]), P.Y(c.curve[i]), 'f2']].concat(ep ? [[P.X(d.s[i]), P.Y(ep.curve[i]), 'f1']] : []), html: tip(`${Math.round(d.s[i] * 100)}% of the way`, [['Mean', pct(d.mean[i], 0)], ['Model', pct(c.curve[i], 0)]].concat(ep ? [[ep.ticker, pct(ep.curve[i], 0)]] : [])) };
        });
      } },
    { id: 'space', title: 'Crashes in shape space', height: 300,
      sub: d => `${d.episodes.length} crashes by their first two mode scores. Click a dot to load its shape.`,
      legend: [['s1', 'dot', 'SPY'], ['s3', 'dot', 'Single stocks']],
      draw(host, d, s, ctx) {
        const sel = ctx.extra.sel, xs = d.episodes.map(e => e.scores[0]), ys = d.episodes.map(e => e.scores[1]);
        const [x0, x1] = extent(xs, [-3, 3]), [y0, y1] = extent(ys, [-2.5, 2.5]);
        const P = frame(host, { h: 300, x: [x0 - 0.3, x1 + 0.3], y: [y0 - 0.3, y1 + 0.3], xl: 'Mode 1: U-shaped ← → V-shaped', yl: 'Mode 2: early ↓ / late ↑ bottom' });
        P.vline(0); P.hline(0);
        const dots = d.episodes.map((e, i) => ({ i, e, x: P.X(e.scores[0]), y: P.Y(e.scores[1]) }));
        dots.slice().sort((a, b) => (a.e.ticker === 'SPY') - (b.e.ticker === 'SPY') || (a.i === sel) - (b.i === sel))
          .forEach(t => P.dot(t.e.scores[0], t.e.scores[1], (t.e.ticker === 'SPY' ? 'f1' : 'f3') + (t.i === sel ? ' sel' : ''), t.i === sel ? 7 : 5));
        // Fixed offsets for the three big SPY crashes so their labels never collide; other deep ones get a generic label.
        const LAB = { '2007-10-09': ['SPY 2008', 9, -8], '2020-02-19': ['SPY 2020', 9, 16], '2000-03-24': ['SPY dot-com', -9, -8, 'end'] };
        dots.filter(t => t.e.ticker === 'SPY' && (LAB[t.e.peak] || t.e.depth >= 0.3)).forEach(t => {
          const left = t.x > P.w * 0.7, [lab, dx, dy, anchor] = LAB[t.e.peak] || [`SPY ${t.e.peak.slice(0, 4)}`, left ? -9 : 9, -8, left ? 'end' : null];
          P.text(t.x + dx, t.y + dy, lab, anchor ? { 'text-anchor': anchor } : {});
        });
        const near = (px, py) => { let best = null, bd = 400; dots.forEach(t => { const dd = (t.x - px) ** 2 + (t.y - py) ** 2; if (dd < bd) { bd = dd; best = t; } }); return best; };
        hover(host, P, (px, py) => {
          const t = near(px, py); if (!t) return null; const e = t.e;
          return { pts: [[t.x, t.y, e.ticker === 'SPY' ? 'f1' : 'f3']], html: tip(`${e.ticker} ${e.peak.slice(0, 4)}–${e.recovered.slice(0, 4)}`, [['Depth', pct(e.depth, 0)], ['Days', int(e.days)], ['Bottom at', pct(e.trough_at, 0) + ' of the way'], ['Modes', `${signed(e.scores[0])}, ${signed(e.scores[1])}, ${signed(e.scores[2])}`]]) + '<br>Click to load' };
        }, { onClick: (px, py) => {
          const t = near(px, py); if (!t) return;
          ctx.extra.sel = t.i;
          const [b1, b2, b3] = t.e.scores.slice(0, 3).map(v => clamp(+v.toFixed(1), -3, 3));
          ctx.set({ b1, b2, b3 }, { immediate: true });
        } });
      } },
    { id: 'all', title: 'Every crash, normalised', height: 260,
      sub: 'Each crash runs from 0 at the peak down to −100% at its trough and back to 0.',
      legend: [['s1', 'line', 'SPY'], ['s3', 'line', 'Single stocks'], ['neutral', 'dash', 'Mean']],
      draw(host, d, s, ctx) {
        const P = frame(host, { h: 260, x: [0, 1], y: [-1.05, 0.05], yt: [-1, -0.5, 0], yf: v => Math.round(v * 100) + '%', xt: [0, 0.25, 0.5, 0.75, 1], xf: v => Math.round(v * 100) + '%', xl: 'Time from peak to recovery' });
        d.episodes.forEach(e => P.line(zip(d.s, e.curve), (e.ticker === 'SPY' ? 's1' : 's3') + ' thin').setAttribute('opacity', '.3'));
        const sel = ctx.extra.sel; if (sel != null && d.episodes[sel]) P.line(zip(d.s, d.episodes[sel].curve), 's1');
        P.line(zip(d.s, d.mean), 'sn dashed');
        hover(host, P, px => {
          const i = nearest(d.s, P.xInv(px)), col = d.episodes.map(e => e.curve[i]).sort((a, b) => a - b), q = f => col[Math.floor(f * (col.length - 1))];
          return { cross: P.X(d.s[i]), pts: [[P.X(d.s[i]), P.Y(d.mean[i]), 'fn']], html: tip(`${Math.round(d.s[i] * 100)}% of the way`, [['Mean', pct(d.mean[i], 0)], ['Middle 50%', `${pct(q(0.25), 0)} to ${pct(q(0.75), 0)}`]]) };
        });
      } },
    { id: 'variance', title: 'How much each mode explains', height: 260,
      sub: d => `Share of the variation between crashes. Three modes explain ${pct(d.cumulative[2], 0)}.`,
      legend: [['s1', 'box', 'Mode share'], ['s2', 'line', 'Cumulative']],
      draw(host, d) {
        const v = d.variance, c = d.cumulative, ks = v.map((_, i) => i + 1);
        const P = frame(host, { h: 260, x: [0.4, v.length + 0.6], y: [0, 1], xt: ks, yt: [0, 0.25, 0.5, 0.75, 1], yf: v => Math.round(v * 100) + '%', xl: 'Mode', yl: 'Share of variance' });
        v.forEach((x, i) => P.bars([i + 1 - 0.32, i + 1 + 0.32], [x], i < 3 ? 'f1' : 'fn', 0));
        P.line(zip(ks, c), 's2'); ks.forEach((k, i) => P.dot(k, c[i], 'f2', 3.5));
        hover(host, P, px => {
          const i = nearest(ks, P.xInv(px));
          return { cross: P.X(ks[i]), pts: [[P.X(ks[i]), P.Y(c[i]), 'f2']], html: tip(`Mode ${ks[i]}`, [['Share', pct(v[i], 1)], ['Cumulative', pct(c[i], 1)], ['Rebuild error', d.rebuild_rmse[i + 1] != null ? fmt(d.rebuild_rmse[i + 1], 3) : '—']]) };
        });
      } },
  ],
  insights: d => {
    const C = d.correlations;
    return [
      `<b>A loose typical shape.</b> Three modes explain ${pct(d.cumulative[2], 0)} of the variation between ${d.episodes.length} crashes. ShapeLab's jawlines reach about 90% with three.`,
      `<b>Where the bottom falls sets the shape.</b> Mode 2 tracks the timing of the trough (r = ${fmt(C.mode2_trough_at, 2)}). Mode 1 separates quick V's from long U's.`,
      Math.abs(C.mode1_depth) > 0.2
        ? `<b>Deep crashes look different.</b> Mode 1 against depth gives r = ${fmt(C.mode1_depth, 2)}: deeper crashes lean ${C.mode1_depth < 0 ? 'U-shaped, with a long flat bottom' : 'V-shaped'}.`
        : `<b>Depth barely sets the shape.</b> Mode 1 against depth gives r = ${fmt(C.mode1_depth, 2)}.`,
    ];
  },
  maths: { eq: 'normalise   y(s) = (P(t₀ + s T) / P(t₀) − 1) / depth,   s ∈ [0, 1]\nlandmarks   x = (y(s₁), …, y(s_N)),   s_j = (j − 1) / (N − 1)\nPCA         x ≈ x̄ + Σₖ bₖ σₖ φₖ,   bₖ in standard deviations',
    notes: ['A crash runs from a peak to the first day the price is back at it, so every shape starts and ends at 0 and touches −1 at the trough. Rotation is left out because time and price cannot be swapped.',
      'Modes are oriented so that +mode 1 is a sharper V and +mode 2 a later trough.'] },
};

export default [randomWalk, dose, crashShapes];
