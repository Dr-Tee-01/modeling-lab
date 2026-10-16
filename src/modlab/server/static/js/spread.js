// Spread tab: the six MC312 models, each re-solved on the server as you drag.
import { $, $$, bind, call, fmt, int, livePanel, pct, readouts, segmented } from './api.js';
import { extent, frame, hover, nearest, niceTicks, responsive, zip } from './plot.js';

const tip = (title, rows) => `<b>${title}</b>` + rows.map(([k, v]) => `<br>${k}: <span class="n">${v}</span>`).join('');

/* ------------------------------------------------------------ dosing */
function dosing(root) {
  const s = { model: 'one', dose: 100, tau: 8, half_life: 22, absorbed: 0.12, ka: 0.6, load: 1, mec: 10, mtc: 20 };
  const panel = $('[data-panel="dosing"]', root);
  const live = livePanel(panel, { request: sig => call('/api/spread/dosing', s, { signal: sig }), render });
  const ch = () => live.refresh();
  const ctl = {};
  for (const k of ['dose', 'tau', 'half_life', 'absorbed', 'ka', 'load', 'mec', 'mtc'])
    ctl[k] = bind(panel, s, k, ch, { dose: v => v + ' mg', tau: v => v + ' h', half_life: v => v + ' h', absorbed: v => Math.round(v * 100) + '%', ka: v => fmt(v) + ' /h', load: v => '×' + fmt(v, 1), mec: v => v + ' µg/mL', mtc: v => v + ' µg/mL' }[k]);
  segmented($('[data-seg="model"]', panel), v => { s.model = v; $('[data-k="ka"]', panel).closest('.ctl').hidden = v !== 'two'; ch(); });
  $('[data-k="ka"]', panel).closest('.ctl').hidden = true;
  $('[data-act="loading"]', panel).addEventListener('click', () => { const d = live.data; if (d) { ctl.load.set(Math.min(20, +d.loading_multiplier.toFixed(1))); ch(); } });
  const host = $('.plot-host', panel);
  const draw = responsive(host, () => {
    const d = live.data; if (!d) return;
    const [, hi] = extent(d.blood, d.gut || [], [s.mtc]);
    const P = frame(host, { h: 300, x: [0, d.t[d.t.length - 1]], y: [0, hi * 1.08], xl: 'Hours', yl: 'Concentration in the blood (µg/mL)', xf: v => v });
    P.band(s.mec, s.mtc);
    P.hline(s.mec); P.hline(s.mtc);
    if (d.gut) P.line(zip(d.t, d.gut), 's4 thin');
    P.line(zip(d.t, d.blood), 's1');
    P.text(P.w - P.M.r - 4, P.Y(s.mtc) - 5, 'MTC', { 'text-anchor': 'end' }); P.text(P.w - P.M.r - 4, P.Y(s.mec) + 14, 'MEC', { 'text-anchor': 'end' });
    hover(host, P, px => { const i = nearest(d.t, P.xInv(px)); return { cross: P.X(d.t[i]), pts: [[P.X(d.t[i]), P.Y(d.blood[i]), 'f1']], html: tip(`Hour ${fmt(d.t[i], 1)}`, [['Blood', fmt(d.blood[i], 2) + ' µg/mL']].concat(d.gut ? [['Gut', fmt(d.gut[i], 2)]] : [])) }; });
  });
  function render(d) {
    const st = { in_window: ['good', 'Settles inside the window'], too_high: ['bad', 'Peaks exceed the safe maximum'], too_low: ['warn', 'Troughs dip below the effective minimum'] }[d.status];
    $('[data-out="chip"]', panel).className = 'chip ' + st[0]; $('[data-out="chip"]', panel).textContent = st[1];
    readouts($('.readout', panel), [
      [s.model === 'one' ? 'Plateau peak / trough' : 'Last-dose peak / trough', `${fmt(d.steady_peak, 1)} / ${fmt(d.steady_trough, 1)} µg/mL`],
      ['First effective', d.first_effective_h == null ? 'never' : `${Math.round(d.first_effective_h)} h`],
      ['Time in window', pct(d.share_in_window, 0)],
      ['Loading dose for an instant plateau', `${int(d.loading_dose_mg)} mg (×${fmt(d.loading_multiplier, 1)})`],
      ['Solver', d.solver === 'exact' ? 'exact solution' : 'RK4, h = 0.05 h'],
    ]);
    $('[data-out="note"]', panel).textContent = `Each dose has decayed to ${pct(d.carryover, 0)} of itself when the next arrives (e^−kτ), so levels climb to a plateau of D / (1 − e^−kτ). Reaching 90% of it takes about ${Math.round(d.hours_to_90pct)} h without a loading dose.`;
    draw();
  }
  return live;
}

/* ------------------------------------------------------------ Bass */
function bass(root) {
  const s = { p: 0.03, q: 0.38, market: 100, years: 25 };
  const panel = $('[data-panel="bass"]', root);
  const live = livePanel(panel, { request: sig => call('/api/spread/bass', s, { signal: sig }), render });
  const ch = () => live.refresh();
  bind(panel, s, 'p', ch, v => fmt(v, 3)); bind(panel, s, 'q', ch, v => fmt(v, 2));
  bind(panel, s, 'market', ch, v => v + 'M'); bind(panel, s, 'years', ch, v => v + ' yrs');
  const [h1, h2] = $$('.plot-host', panel);
  const draw = () => {
    const d = live.data; if (!d) return;
    const T = d.t[d.t.length - 1];
    let P = frame(h1, { h: 240, x: [0, T], y: [0, s.market * 1.05], xl: 'Years', yl: 'Adopters (millions)' });
    P.line(zip(d.t, d.exact), 's2 dashed'); P.line(zip(d.t, d.adopters), 's1');
    hover(h1, P, px => { const i = nearest(d.t, P.xInv(px)); return { cross: P.X(d.t[i]), pts: [[P.X(d.t[i]), P.Y(d.adopters[i]), 'f1']], html: tip(`Year ${fmt(d.t[i], 1)}`, [['RK4', fmt(d.adopters[i], 2)], ['Closed form', fmt(d.exact[i], 2)]]) }; });
    const [, hi] = extent(d.sales_innovators, d.sales_imitators);
    P = frame(h2, { h: 240, x: [0, T], y: [0, hi * 1.1], xl: 'Years', yl: 'Sales per year (millions)' });
    P.line(zip(d.t, d.sales_imitators), 's3'); P.line(zip(d.t, d.sales_innovators), 's2');
    P.vline(d.peak_year);
    hover(h2, P, px => { const i = nearest(d.t, P.xInv(px)); return { cross: P.X(d.t[i]), pts: [[P.X(d.t[i]), P.Y(d.sales_innovators[i]), 'f2'], [P.X(d.t[i]), P.Y(d.sales_imitators[i]), 'f3']], html: tip(`Year ${fmt(d.t[i], 1)}`, [['Outside influence', fmt(d.sales_innovators[i], 2)], ['Word of mouth', fmt(d.sales_imitators[i], 2)]]) }; });
  };
  responsive(h1, draw); responsive(h2, draw);
  function render(d) {
    readouts($('.readout', panel), [['Sales peak', d.q > d.p || s.q > s.p ? `year ${fmt(d.peak_year, 1)}` : 'at launch'], ['Peak sales', fmt(d.peak_sales, 1) + 'M / yr'],
      ['Bought because of outside influence', pct(d.share_innovators, 0)], ['Largest RK4 vs closed-form gap', d.max_solver_error.toExponential(1)]]);
    draw();
  }
  return live;
}

/* ------------------------------------------------------------ SIR */
function sir(root) {
  const s = { beta: 1.66, alpha: 0.5, vaccinate: 0, lockdown: false, lock_start: 3, lock_days: 10, lock_cut: 50, days: 20 };
  const panel = $('[data-panel="sir"]', root);
  const live = livePanel(panel, { request: sig => call('/api/spread/sir', s, { signal: sig }), render });
  const ch = () => live.refresh();
  const c = {
    beta: bind(panel, s, 'beta', ch, v => fmt(v) + ' /day'), alpha: bind(panel, s, 'alpha', ch, v => `${fmt(v)} /day · ${fmt(1 / v, 1)} d ill`),
  };
  bind(panel, s, 'vaccinate', ch, v => v + '% / day'); bind(panel, s, 'lockdown', () => { $$('.lock', panel).forEach(e => e.hidden = !s.lockdown); ch(); });
  bind(panel, s, 'lock_start', ch, v => 'day ' + v); bind(panel, s, 'lock_days', ch, v => v + ' days'); bind(panel, s, 'lock_cut', ch, v => v + '%');
  $$('.lock', panel).forEach(e => e.hidden = true);
  const fitBtn = $('[data-act="fit"]', panel), fitOut = $('[data-out="fit"]', panel);
  fitBtn.addEventListener('click', async () => {
    fitBtn.disabled = true; fitOut.textContent = 'Fitting on the server…';
    try {
      const f = await call('/api/spread/sir/fit', {}, { method: 'POST' });
      c.beta.set(+f.beta.toFixed(2)); c.alpha.set(+f.alpha.toFixed(2));
      fitOut.textContent = `Least squares: β = ${fmt(f.beta, 3)}, α = ${fmt(f.alpha, 3)}, RMSE ${fmt(f.rmse, 1)} after ${f.evaluations} RK4 solves (${f.meta.compute_ms} ms).`;
      live.now();
    } catch (e) { fitOut.textContent = e.message; } finally { fitBtn.disabled = false; }
  });
  const host = $('.plot-host', panel);
  const draw = responsive(host, () => {
    const d = live.data; if (!d) return;
    const P = frame(host, { h: 320, x: [0, s.days], y: [0, 780], xl: 'Day of outbreak', yl: 'Boys' });
    if (s.lockdown) P.band(0, 780, 'zone').setAttribute('x', P.X(s.lock_start)), P.body.lastChild.setAttribute('width', Math.max(0, P.X(Math.min(s.days, s.lock_start + s.lock_days)) - P.X(s.lock_start)));
    P.line(zip(d.t, d.S), 's1'); P.line(zip(d.t, d.R), 's3'); P.line(zip(d.t, d.I), 's2');
    if (s.vaccinate > 0) P.line(zip(d.t, d.V), 's4 dashed');
    d.data.day.forEach((x, i) => P.dot(x, d.data.in_bed[i], 'f2', 4));
    hover(host, P, px => { const i = nearest(d.t, P.xInv(px)); return { cross: P.X(d.t[i]), pts: [[P.X(d.t[i]), P.Y(d.I[i]), 'f2']], html: tip(`Day ${fmt(d.t[i], 1)}`, [['Susceptible', int(d.S[i])], ['Infected', int(d.I[i])], ['Recovered', int(d.R[i])]].concat(Number.isInteger(+d.t[i]) && d.t[i] >= 1 && d.t[i] <= 14 ? [['In bed, 1978', d.data.in_bed[d.t[i] - 1]]] : [])) }; });
  });
  function render(d) {
    readouts($('.readout', panel), [['R₀ = β/α', fmt(d.r0, 2)], ['RMSE vs 1978', fmt(d.rmse, 1) + ' boys'], ['Peak', `${int(d.peak)} on day ${fmt(d.peak_day, 1)}`], ['Ever infected', `${int(d.ever_infected)} (${pct(d.share_infected, 0)})`]]);
    draw();
  }
  return live;
}

/* ------------------------------------------------------------ rumour */
function rumour(root) {
  const s = { beta: 0.3, gamma: 0.1, alpha: 0, newcomers: 0 };
  const panel = $('[data-panel="rumour"]', root);
  const live = livePanel(panel, { request: sig => call('/api/spread/rumour', s, { signal: sig }), render });
  const ch = () => live.refresh();
  bind(panel, s, 'beta', ch, v => fmt(v)); bind(panel, s, 'gamma', ch, v => fmt(v)); bind(panel, s, 'alpha', ch, v => fmt(v, 3)); bind(panel, s, 'newcomers', ch, v => v + ' / day');
  const host = $('.plot-host', panel);
  const draw = responsive(host, () => {
    const d = live.data; if (!d) return;
    const [, hi] = extent(d.ignorant, d.spreader, d.stifler);
    const P = frame(host, { h: 300, x: [0, 100], y: [0, hi * 1.05], xl: 'Days', yl: 'People' });
    P.line(zip(d.t, d.ignorant), 's1'); P.line(zip(d.t, d.stifler), 's3'); P.line(zip(d.t, d.spreader), 's2');
    if (d.theory.s_max != null) { P.hline(d.theory.s_max); P.text(P.w - P.M.r - 4, P.Y(d.theory.s_max) - 5, 'theory: peak spreaders', { 'text-anchor': 'end' }); }
    hover(host, P, px => { const i = nearest(d.t, P.xInv(px)); return { cross: P.X(d.t[i]), pts: [[P.X(d.t[i]), P.Y(d.spreader[i]), 'f2']], html: tip(`Day ${fmt(d.t[i], 0)}`, [['Never heard', int(d.ignorant[i])], ['Spreading', int(d.spreader[i])], ['Stopped', int(d.stifler[i])]]) }; });
  });
  function render(d) {
    const th = d.theory;
    readouts($('.readout', panel), [['Peak spreaders', `${int(d.peak_spreaders)}` + (th.s_max != null ? ` (theory ${int(th.s_max)})` : '')], ['Reach', pct(d.reach, 0) + (th.r_inf != null ? ` (theory ${pct(th.r_inf, 0)})` : '')],
      ['Conserved quantity drift', d.conserved_drift == null ? 'n/a with campaign or newcomers' : d.conserved_drift.toExponential(1)]]);
    draw();
  }
  return live;
}

/* ------------------------------------------------------------ Monte Carlo */
function chance(root) {
  const s = { lam: 0.05, runs: 100, seed: 9, n: 5000 };
  const panel = $('[data-panel="chance"]', root);
  const live = livePanel(panel, { request: async sig => {
    const [a, b] = await Promise.all([call('/api/spread/decay', s, { signal: sig }), call('/api/spread/coins', { n: s.n, seed: s.seed }, { signal: sig })]);
    return Object.assign(a, { coins: b, meta: { compute_ms: a.meta.compute_ms + b.meta.compute_ms } });
  }, render });
  const ch = () => live.refresh();
  bind(panel, s, 'lam', ch, v => fmt(v, 3) + ' /s'); bind(panel, s, 'runs', ch, v => v + ' runs'); bind(panel, s, 'n', ch, v => int(v) + ' tosses');
  $('[data-act="reroll"]', panel).addEventListener('click', () => { s.seed = Math.floor(Math.random() * 1e9); live.now(); });
  const [h1, h2] = $$('.plot-host', panel);
  const draw = () => {
    const d = live.data; if (!d) return;
    let P = frame(h1, { h: 260, x: [0, 100], y: [0, 1000], xl: 'Seconds', yl: 'Atoms not yet decayed' });
    d.runs_shown.forEach(r => P.line(zip(d.t, r), 'sn thin').setAttribute('opacity', '.35'));
    P.area(zip(d.t, d.mean.map((m, i) => m + d.sd[i])), zip(d.t, d.mean.map((m, i) => m - d.sd[i])), 'band');
    P.line(zip(d.t, d.theory), 's2 dashed'); P.line(zip(d.t, d.mean), 's1');
    hover(h1, P, px => { const i = nearest(d.t, P.xInv(px)); return { cross: P.X(i), pts: [[P.X(i), P.Y(d.mean[i]), 'f1']], html: tip(`t = ${i} s`, [['Mean of runs', fmt(d.mean[i], 1)], ['± sd', fmt(d.sd[i], 1)], ['N₀e^−λt', fmt(d.theory[i], 1)]]) }; });
    const c = d.coins;
    P = frame(h2, { h: 260, x: [1, c.n[c.n.length - 1]], xlog: true, y: [0, 1], xt: niceLog(c.n[c.n.length - 1]), xf: v => int(v), xl: 'Tosses (log scale)', yl: 'Share of heads' });
    P.area(zip(c.n, c.band.map(b => 0.5 + b)), zip(c.n, c.band.map(b => 0.5 - b)), 'band-null'); P.hline(0.5);
    c.share.forEach((sq, i) => P.line(zip(c.n, sq), ['s1', 's2', 's3', 's4', 's1', 's2', 's3', 's4', 's1', 's2', 's3', 's4'][i] + ' thin'));
  };
  responsive(h1, draw); responsive(h2, draw);
  function render(d) {
    readouts($('.readout', panel), [['Half-life measured', d.half_life_measured == null ? '> 100 s' : d.half_life_measured + ' s'], ['Theory ln 2 / λ', fmt(d.half_life_theory, 1) + ' s'],
      ['Spread at the half-life', `±${fmt(d.sd_at_half_life, 1)} (theory ±${fmt(d.sd_theory_at_half_life, 1)})`], ['Worst gap, mean vs theory', fmt(d.max_gap_to_theory, 1) + ' atoms']]);
    draw();
  }
  return live;
}
const niceLog = n => [1, 10, 100, 1000, 10000, 100000].filter(v => v <= n);

/* ------------------------------------------------------------ random walk */
function walk(root) {
  const s = { p: 0.5, steps: 200, walkers: 2000, seed: 10 };
  const panel = $('[data-panel="walk"]', root);
  const live = livePanel(panel, { request: async sig => {
    const [a, b] = await Promise.all([call('/api/spread/walk', s, { signal: sig }), call('/api/spread/walk2d', { seed: s.seed }, { signal: sig })]);
    return Object.assign(a, { w2: b, meta: { compute_ms: a.meta.compute_ms + b.meta.compute_ms } });
  }, render });
  const ch = () => live.refresh();
  bind(panel, s, 'p', ch, v => fmt(v)); bind(panel, s, 'steps', ch, v => v); bind(panel, s, 'walkers', ch, v => int(v));
  $('[data-act="reroll"]', panel).addEventListener('click', () => { s.seed = Math.floor(Math.random() * 1e9); live.now(); });
  const [h1, h2, h3] = $$('.plot-host', panel);
  const draw = () => {
    const d = live.data; if (!d) return;
    const [lo, hi] = extent(...d.paths, d.mean_theory);
    let P = frame(h1, { h: 260, x: [0, s.steps], y: [Math.min(lo, -5), Math.max(hi, 5)], xl: 'Step', yl: 'Position' });
    d.paths.forEach(r => P.line(zip(d.n, r), 'sn thin').setAttribute('opacity', '.35'));
    const sdT = d.var_theory.map(Math.sqrt);
    P.line(zip(d.n, d.mean_theory.map((m, i) => m + 2 * sdT[i])), 's2 dashed'); P.line(zip(d.n, d.mean_theory.map((m, i) => m - 2 * sdT[i])), 's2 dashed');
    P.line(zip(d.n, d.mean), 's1');
    hover(h1, P, px => { const i = nearest(d.n, P.xInv(px)); return { cross: P.X(i), html: tip(`Step ${i}`, [['Mean', fmt(d.mean[i], 2)], ['Theory n(2p−1)', fmt(d.mean_theory[i], 2)], ['Variance', fmt(d.var[i], 1)], ['Theory 4np(1−p)', fmt(d.var_theory[i], 1)]]) }; });
    const vals = d.final_values, cnt = d.final_counts, edges = vals.map(v => v - 1).concat([vals[vals.length - 1] + 1]);
    P = frame(h2, { h: 260, x: [edges[0] - 2, edges[edges.length - 1] + 2], y: [0, Math.max(...cnt, ...(d.normal_expected || [0])) * 1.1], xl: 'Final position', yl: 'Walkers' });
    P.bars(edges, cnt, 'f1');
    if (d.normal_expected) P.line(zip(vals, d.normal_expected), 's2');
    hover(h2, P, px => { const i = nearest(vals, P.xInv(px)); return { html: tip(`Position ${vals[i]}`, [['Walkers', cnt[i]], ['Normal approx.', d.normal_expected ? fmt(d.normal_expected[i], 1) : '—']]) }; });
    const w = d.w2, all = w.paths.flat();
    const ext = Math.max(10, ...all.map(p => Math.max(Math.abs(p[0]), Math.abs(p[1])))) * 1.05;
    const size = Math.min(h3.clientWidth, 340);
    P = frame(h3, { h: size, x: [-ext, ext], y: [-ext, ext], m: { l: 36, r: 8, t: 8, b: 24 } });
    w.paths.forEach((path, i) => P.line(path, ['s1', 's2', 's3', 's4'][i % 4] + ' thin'));
  };
  responsive(h1, draw); responsive(h2, draw); responsive(h3, draw);
  function render(d) {
    const N = d.n.length - 1, w = d.w2;
    readouts($('.readout', panel), [['Mean after n steps', `${fmt(d.mean[N], 2)} (theory ${fmt(d.mean_theory[N], 2)})`], ['Variance', `${fmt(d.var[N], 1)} (theory ${fmt(d.var_theory[N], 1)})`],
      ['2-D mean squared distance', `${fmt(w.msd[w.msd.length - 1], 0)} (theory ${fmt(w.msd_theory[w.msd_theory.length - 1], 0)})`]]);
    draw();
  }
  return live;
}

export function initSpread(root) {
  const models = { dosing, bass, sir, rumour, chance, walk };
  const lives = {};
  const show = name => {
    $$('[data-panel]', root).forEach(p => p.hidden = p.dataset.panel !== name);
    if (!lives[name]) { lives[name] = models[name](root); lives[name].now(); }
    else lives[name].redraw();
  };
  const sel = segmented($('.subnav', root), v => { show(v); try { localStorage.setItem('ml-spread', v); } catch (e) {} });
  let first = 'sir'; try { first = localStorage.getItem('ml-spread') || 'sir'; } catch (e) {}
  if (!models[first]) first = 'sir';
  sel(first); show(first);
}
