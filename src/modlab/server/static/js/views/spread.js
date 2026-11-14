// Pillar 1 · Spread: the six MC312 models as workbench specs (the reference format for every view).
import { call, fmt, int, pct } from '../api.js';
import { extent, frame, hover, nearest, niceTicks, zip } from '../plot.js';
import { S, edge, node, spark, text, widthFor } from '../ui/illustrate.js';

const tip = (title, rows) => `<b>${title}</b>` + rows.map(([k, v]) => `<br>${k}: <span class="n">${v}</span>`).join('');
const get = (path, s, sig, keys) => call(path, keys ? Object.fromEntries(keys.map(k => [k, s[k]])) : s, { signal: sig });

/* ================================================================ Lab 2: dosing */
const dosing = {
  id: 'dosing', group: 'spread', nav: 'Drug dosing', tag: 'Lab 2', title: 'Repeated dosing and the therapeutic window',
  question: 'How should a drug be dosed so that its level in the blood stays between "too little to work" and "toxic"?',
  params: [
    { k: 'model', type: 'seg', label: 'Model', value: 'one', options: [['one', 'One compartment'], ['two', 'Gut → blood']], group: 'Model',
      help: 'One compartment: each dose enters the blood at once. Gut → blood: the dose is absorbed gradually from the gut.' },
    { k: 'dose', label: 'Dose', min: 25, max: 400, step: 5, value: 100, unit: 'mg', group: 'Regimen', help: 'Amount of drug taken each time.' },
    { k: 'tau', label: 'Interval τ', min: 2, max: 24, step: 1, value: 8, unit: 'h', group: 'Regimen', help: 'Hours between doses.' },
    { k: 'load', label: 'First dose multiplier', min: 1, max: 10, step: 0.1, value: 1, unit: '×', group: 'Regimen', help: 'A larger first (loading) dose reaches the steady level immediately.' },
    { k: 'half_life', label: 'Elimination half-life', min: 2, max: 72, step: 1, value: 22, unit: 'h', group: 'Drug', help: 'Time for the body to clear half the drug. Phenytoin: about 22 h.' },
    { k: 'absorbed', label: 'Fraction absorbed', min: 0.02, max: 1, step: 0.01, value: 0.12, scale: 100, unit: '%', group: 'Drug', help: 'Share of each dose that reaches the blood.' },
    { k: 'ka', label: 'Gut → blood rate k₁', min: 0.05, max: 3, step: 0.05, value: 0.6, unit: '/h', group: 'Drug', show: s => s.model === 'two', help: 'How fast the gut passes the drug into the blood.' },
    { k: 'mec', label: 'Minimum effective (MEC)', min: 1, max: 40, step: 0.5, value: 10, unit: 'µg/mL', group: 'Therapeutic window', help: 'Below this level the drug does not work.' },
    { k: 'mtc', label: 'Maximum safe (MTC)', min: 2, max: 60, step: 0.5, value: 20, unit: 'µg/mL', group: 'Therapeutic window', help: 'Above this level the drug is toxic.' },
  ],
  presets: [
    { label: 'Phenytoin', values: { model: 'one', dose: 100, tau: 8, half_life: 22, absorbed: 0.12, load: 1 } },
    { label: 'With loading dose', values: { model: 'one', dose: 100, tau: 8, half_life: 22, absorbed: 0.12, load: 4.5 } },
    { label: 'Once a day', values: { model: 'one', dose: 300, tau: 24, half_life: 22, absorbed: 0.12, load: 1 } },
    { label: 'Absorbed via the gut', values: { model: 'two', dose: 100, tau: 8, half_life: 22, absorbed: 0.12, ka: 0.6, load: 1 } },
  ],
  request: (s, sig) => {
    if (s.mtc <= s.mec) return Promise.reject(new Error('The maximum safe level must be above the minimum effective level.'));
    return get('/api/spread/dosing', s, sig);
  },
  kpis: d => {
    const tone = { in_window: 'good', too_high: 'bad', too_low: 'warn' }[d.status];
    return [
      { label: d.solver === 'exact' ? 'Plateau peak / trough' : 'Last-dose peak / trough', value: `${fmt(d.steady_peak, 1)} / ${fmt(d.steady_trough, 1)}`, unit: 'µg/mL', tone,
        note: { in_window: 'Settles inside the window', too_high: 'Peaks exceed the safe maximum', too_low: 'Troughs fall below the effective level' }[d.status] },
      { label: 'Time inside the window', value: pct(d.share_in_window, 0), note: 'over 10 days', tone: d.share_in_window > 0.8 ? 'good' : d.share_in_window > 0.4 ? 'warn' : 'bad' },
      { label: 'First effective', value: d.first_effective_h == null ? 'never' : Math.round(d.first_effective_h), unit: d.first_effective_h == null ? '' : 'h', note: `90% of the plateau after ${Math.round(d.hours_to_90pct)} h` },
      { label: 'Loading dose for an instant plateau', value: int(d.loading_dose_mg), unit: 'mg', note: `first dose ×${fmt(d.loading_multiplier, 1)}` },
    ];
  },
  illustration: {
    title: 'Where the drug goes', height: 200,
    caption: 'Arrow widths follow the rates. The blood tank fills to the steady peak, coloured by whether it sits inside the window.',
    draw(svg, d, s) {
      const k = Math.LN2 / s.half_life, D = s.dose * 1000 * s.absorbed / 3000, p = Math.exp(-k * s.tau);
      const peak = d ? d.steady_peak : D / (1 - p);
      const status = peak > s.mtc ? 'fill2' : (d ? d.steady_trough : D * p / (1 - p)) < s.mec ? 'fill4' : 'fill3';
      const pill = node(svg, { x: 10, y: 66, w: 110, h: 64, title: 'Dose', value: `${s.dose} mg`, sub: `every ${s.tau} h` });
      let src = pill;
      if (s.model === 'two') {
        const gut = node(svg, { x: 170, y: 66, w: 110, h: 64, title: 'Gut', sub: `k₁ = ${fmt(s.ka)}/h`, fill: 0.5, fillCls: 'fill4' });
        edge(svg, [pill.r + 4, pill.cy], [gut.x - 6, gut.cy], { label: `${pct(s.absorbed, 0)} absorbed`, width: 3, flow: true });
        src = gut;
      }
      const bx = s.model === 'two' ? 330 : 230;
      const blood = node(svg, { x: bx, y: 40, w: 130, h: 116, title: 'Blood', value: `${fmt(peak, 1)} µg/mL`, sub: `window ${s.mec}–${s.mtc}`, fill: peak / (s.mtc * 1.3), fillCls: status, hot: true });
      // therapeutic window marks on the tank
      const yOf = v => blood.b - 1 - (blood.h - 2) * Math.min(1, v / (s.mtc * 1.3));
      [s.mec, s.mtc].forEach(v => S('line', { x1: blood.x, x2: blood.r, y1: yOf(v), y2: yOf(v), class: 'ln-m', 'stroke-dasharray': '3 3' }, svg));
      edge(svg, [src.r + 4, src.cy], [blood.x - 6, blood.cy], { label: s.model === 'two' ? `k₁·gut` : `${pct(s.absorbed, 0)} absorbed`, width: 3, flow: true });
      const out = node(svg, { x: 520 - (s.model === 'two' ? 0 : 0), y: 66, w: 74, h: 64, title: 'Cleared', sub: `t½ ${s.half_life} h` });
      edge(svg, [blood.r + 4, blood.cy], [out.x - 6, out.cy], { label: `k = ${fmt(k, 3)}/h`, width: widthFor(k, 0.35), flow: true });
    },
  },
  charts: [{
    id: 'level', title: 'Concentration in the blood', height: 300,
    sub: (d, s) => d.solver === 'exact' ? 'Exact solution of the one-compartment model.' : 'Gut and blood integrated with RK4 (h = 0.05 h).',
    legend: (d, s) => [['s1', 'line', 'Blood'], ...(s.model === 'two' ? [['s4', 'line', 'Gut']] : []), ['s3', 'box', 'Therapeutic window']],
    draw(host, d, s) {
      const [, hi] = extent(d.blood, d.gut || [], [s.mtc]);
      const P = frame(host, { h: 300, x: [0, d.t[d.t.length - 1]], y: [0, hi * 1.08], xl: 'Hours', yl: 'µg/mL' });
      P.band(s.mec, s.mtc); P.hline(s.mec); P.hline(s.mtc);
      if (d.gut) P.line(zip(d.t, d.gut), 's4 thin');
      P.line(zip(d.t, d.blood), 's1');
      P.text(P.w - P.M.r - 4, P.Y(s.mtc) - 5, 'MTC', { 'text-anchor': 'end' }); P.text(P.w - P.M.r - 4, P.Y(s.mec) + 14, 'MEC', { 'text-anchor': 'end' });
      hover(host, P, px => { const i = nearest(d.t, P.xInv(px)); return { cross: P.X(d.t[i]), pts: [[P.X(d.t[i]), P.Y(d.blood[i]), 'f1']], html: tip(`Hour ${fmt(d.t[i], 1)}`, [['Blood', fmt(d.blood[i], 2) + ' µg/mL']].concat(d.gut ? [['Gut', fmt(d.gut[i], 2)]] : [])) }; });
    },
  }],
  insights: (d, s) => [
    `Each dose has decayed to <b>${pct(d.carryover, 0)}</b> of itself when the next arrives (e<sup>−kτ</sup>), so levels climb to a plateau of D / (1 − e<sup>−kτ</sup>) instead of growing forever.`,
    s.load < 1.05 ? `Without a loading dose, reaching 90% of the plateau takes about <b>${Math.round(d.hours_to_90pct)} h</b>. A first dose ${fmt(d.loading_multiplier, 1)}× larger starts the patient on the plateau.` : `The loading dose puts the patient near the plateau from the first hours.`,
    d.status === 'too_high' ? 'Peaks exceed the safe maximum: lower the dose or split it into smaller, more frequent doses.' : d.status === 'too_low' ? 'Troughs dip below the effective level: shorten the interval or raise the dose.' : 'This regimen stays inside the therapeutic window once it settles.',
  ],
  maths: { eq: 'one compartment   dC/dt = −k C + D Σ δ(t − nτ)\n                  k = ln 2 / t½\nplateau           C_peak = D / (1 − e^(−kτ))\ngut → blood       dG/dt = −k₁ G,   dB/dt = k₁ G − k B',
    notes: ['D is the dose that reaches the blood per unit volume: dose × fraction absorbed / 3 L.'] },
};

/* ================================================================ Lab 4: Bass */
const bass = {
  id: 'bass', group: 'spread', nav: 'Product adoption', tag: 'Lab 4', title: 'Bass diffusion: advertising and word of mouth',
  question: 'Why do new products sell slowly, then explode, then saturate?',
  params: [
    { k: 'p', label: 'Innovation p', min: 0.001, max: 0.15, step: 0.001, value: 0.03, group: 'Behaviour', help: 'Share of non-owners who buy each year because of advertising or outside influence.' },
    { k: 'q', label: 'Imitation q', min: 0, max: 1, step: 0.01, value: 0.38, group: 'Behaviour', help: 'Strength of word of mouth: buying because others already own it.' },
    { k: 'market', label: 'Market size M', min: 5, max: 500, step: 5, value: 100, unit: 'M people', group: 'Market' },
    { k: 'years', label: 'Horizon', min: 10, max: 40, step: 1, value: 25, unit: 'years', group: 'Market' },
  ],
  presets: [
    { label: 'Typical product', values: { p: 0.03, q: 0.38 } },
    { label: 'Word-of-mouth hit', values: { p: 0.01, q: 0.7 } },
    { label: 'Advertising-led', values: { p: 0.1, q: 0.1 } },
  ],
  request: (s, sig) => get('/api/spread/bass', s, sig),
  kpis: (d, s) => [
    { label: 'Sales peak', value: s.q > s.p ? fmt(d.peak_year, 1) : '0', unit: 'years in', note: 'peak at ln(q/p) / (p + q)' },
    { label: 'Peak sales', value: fmt(d.peak_sales, 1), unit: 'M / year' },
    { label: 'Bought because of outside influence', value: pct(d.share_innovators, 0), note: 'the rest followed word of mouth' },
    { label: 'Solver vs closed form', value: d.max_solver_error.toExponential(1), unit: 'M', tone: d.max_solver_error < 1e-4 ? 'good' : 'warn', note: 'largest RK4 error' },
  ],
  illustration: {
    title: 'Two reasons to buy', height: 190,
    draw(svg, d, s) {
      const share = d ? d.share_innovators : s.p / (s.p + s.q);
      const pool = node(svg, { x: 20, y: 60, w: 150, h: 70, title: 'Not yet bought', sub: `M = ${s.market}M` });
      const own = node(svg, { x: 420, y: 60, w: 160, h: 70, title: 'Adopters N(t)', sub: d ? `${fmt(d.adopters[d.adopters.length - 1], 0)}M after ${s.years} y` : '', fill: d ? d.adopters[d.adopters.length - 1] / s.market : 0.6, fillCls: 'fill1', hot: true });
      edge(svg, [pool.r + 4, pool.cy - 14], [own.x - 6, own.cy - 14], { label: `advertising  p = ${fmt(s.p, 3)}`, width: widthFor(share, 1), cls: 'st2', curve: [0, -40], flow: true });
      edge(svg, [pool.r + 4, pool.cy + 14], [own.x - 6, own.cy + 14], { label: `word of mouth  q·N/M, q = ${fmt(s.q)}`, width: widthFor(1 - share, 1), cls: 'st3', curve: [0, 40], labelAt: [295, 185], flow: true });
    },
  },
  chartLayout: 'two',
  charts: [
    { id: 'cum', title: 'Adopters over time', sub: 'RK4 solution (solid) over the closed form (dashed).', legend: [['s1', 'line', 'RK4'], ['s2', 'dash', 'Closed form']],
      draw(host, d, s) {
        const P = frame(host, { h: 260, x: [0, s.years], y: [0, s.market * 1.05], xl: 'Years', yl: 'Millions' });
        P.line(zip(d.t, d.exact), 's2 dashed'); P.line(zip(d.t, d.adopters), 's1');
        hover(host, P, px => { const i = nearest(d.t, P.xInv(px)); return { cross: P.X(d.t[i]), pts: [[P.X(d.t[i]), P.Y(d.adopters[i]), 'f1']], html: tip(`Year ${fmt(d.t[i], 1)}`, [['RK4', fmt(d.adopters[i], 2)], ['Closed form', fmt(d.exact[i], 2)]]) }; });
      } },
    { id: 'sales', title: 'Sales per year, by reason', sub: 'The vertical line marks the sales peak.', legend: [['s2', 'line', 'Outside influence'], ['s3', 'line', 'Word of mouth']],
      draw(host, d, s) {
        const [, hi] = extent(d.sales_innovators, d.sales_imitators);
        const P = frame(host, { h: 260, x: [0, s.years], y: [0, hi * 1.1], xl: 'Years', yl: 'Millions / year' });
        P.line(zip(d.t, d.sales_imitators), 's3'); P.line(zip(d.t, d.sales_innovators), 's2'); P.vline(d.peak_year);
        hover(host, P, px => { const i = nearest(d.t, P.xInv(px)); return { cross: P.X(d.t[i]), pts: [[P.X(d.t[i]), P.Y(d.sales_innovators[i]), 'f2'], [P.X(d.t[i]), P.Y(d.sales_imitators[i]), 'f3']], html: tip(`Year ${fmt(d.t[i], 1)}`, [['Outside influence', fmt(d.sales_innovators[i], 2)], ['Word of mouth', fmt(d.sales_imitators[i], 2)]]) }; });
      } },
  ],
  insights: (d, s) => [
    `Early on there is nobody to imitate, so <b>advertising drives the first sales</b> even when word of mouth is strong.`,
    s.q > s.p ? `Sales peak after <b>${fmt(d.peak_year, 1)} years</b>, when word of mouth outweighs the shrinking pool of non-owners.` : 'With q ≤ p, word of mouth never takes over and sales fall from launch.',
    `RK4 matches the closed form to <b>${d.max_solver_error.toExponential(0)}</b> million people, a check on the solver.`,
  ],
  maths: { eq: 'dN/dt = (p + q·N/M)(M − N)\nN(t)  = M (1 − e^(−(p+q)t)) / (1 + (q/p) e^(−(p+q)t))\nt*    = ln(q/p) / (p + q)', notes: ['Innovators buy at rate p(M − N); imitators at rate q(N/M)(M − N).'] },
};

/* ================================================================ Labs 5-6: SIR */
const sir = {
  id: 'sir', group: 'spread', nav: 'Epidemic (SIR)', tag: 'Labs 5–6', title: 'Flu in a boarding school, January 1978',
  question: '763 boys, one initial case. How fast did flu spread, and what would vaccination or a lockdown have changed?',
  params: [
    { k: 'beta', label: 'Transmission β', min: 0.2, max: 3, step: 0.01, value: 1.66, unit: '/day', group: 'Disease', help: 'Infections caused per infected boy per day, in a fully susceptible school.' },
    { k: 'alpha', label: 'Recovery α', min: 0.1, max: 1, step: 0.01, value: 0.5, unit: '/day', group: 'Disease', help: '1/α is the average number of days a boy stays ill.' },
    { k: 'vaccinate', label: 'Vaccinate', min: 0, max: 30, step: 1, value: 0, unit: '% / day', group: 'Interventions', help: 'Share of the remaining susceptible boys vaccinated each day.' },
    { k: 'lockdown', type: 'switch', label: 'Lockdown', value: false, group: 'Interventions', help: 'Cut contacts for a window of days.' },
    { k: 'lock_start', label: 'Starts on day', min: 0, max: 15, step: 1, value: 3, group: 'Interventions', show: s => s.lockdown },
    { k: 'lock_days', label: 'Lasts', min: 1, max: 20, step: 1, value: 10, unit: 'days', group: 'Interventions', show: s => s.lockdown },
    { k: 'lock_cut', label: 'Contacts cut by', min: 0, max: 90, step: 5, value: 50, unit: '%', group: 'Interventions', show: s => s.lockdown },
  ],
  presets: [
    { label: 'Best fit to 1978', values: { beta: 1.67, alpha: 0.44, vaccinate: 0, lockdown: false } },
    { label: 'Early lockdown', values: { beta: 1.67, alpha: 0.44, vaccinate: 0, lockdown: true, lock_start: 2, lock_days: 10, lock_cut: 60 } },
    { label: 'Late lockdown', values: { beta: 1.67, alpha: 0.44, vaccinate: 0, lockdown: true, lock_start: 6, lock_days: 10, lock_cut: 60 } },
    { label: 'Vaccinate 10%/day', values: { beta: 1.67, alpha: 0.44, vaccinate: 10, lockdown: false } },
  ],
  request: (s, sig) => get('/api/spread/sir', s, sig),
  actions: [{ label: 'Fit β and α to the 1978 data', icon: 'fit', primary: true, async run(ctx, ui) {
    ui.note('Fitting on the server…');
    const f = await call('/api/spread/sir/fit', {}, { method: 'POST' });
    ctx.set({ beta: +f.beta.toFixed(2), alpha: +f.alpha.toFixed(2), vaccinate: 0, lockdown: false }, { immediate: true });
    ui.note(`Least squares: β = ${fmt(f.beta, 3)}, α = ${fmt(f.alpha, 3)}, RMSE ${fmt(f.rmse, 1)} after ${f.evaluations} RK4 solves (${Math.round(f.meta.compute_ms)} ms).`);
  } }],
  kpis: d => [
    { label: 'Basic reproduction number R₀', value: fmt(d.r0, 2), note: 'infections per case, early on', tone: d.r0 > 1 ? 'warn' : 'good' },
    { label: 'Peak in bed', value: int(d.peak), unit: 'boys', note: `on day ${fmt(d.peak_day, 1)}` },
    { label: 'Ever infected', value: pct(d.share_infected, 0), note: `${int(d.ever_infected)} of 763` },
    { label: 'Fit to the 1978 counts', value: fmt(d.rmse, 1), unit: 'boys RMSE', tone: d.rmse < 20 ? 'good' : d.rmse < 50 ? 'warn' : 'bad', note: 'root-mean-square error, days 1–14' },
  ],
  illustration: {
    title: 'Who moves where', height: 210,
    caption: 'Box values are at the peak of the outbreak. Arrow widths follow the transmission and recovery rates.',
    draw(svg, d, s) {
      const pk = d ? nearest(d.t, d.peak_day) : null;
      const v = arr => d ? int(arr[pk]) : null;
      const Sx = node(svg, { x: 10, y: 54, w: 124, h: 76, title: 'Susceptible', value: v(d?.S), sub: 'can catch it' });
      const Ix = node(svg, { x: 238, y: 54, w: 124, h: 76, title: 'Infected', value: v(d?.I), sub: 'in bed', hot: true });
      const Rx = node(svg, { x: 466, y: 54, w: 124, h: 76, title: 'Recovered', value: v(d?.R), sub: 'immune' });
      const cut = s.lockdown ? 1 - s.lock_cut / 100 : 1;
      edge(svg, [Sx.r + 4, Sx.cy], [Ix.x - 6, Ix.cy], { label: `β = ${fmt(s.beta)}`, below: 'infection', width: widthFor(s.beta, 3), cls: 'st2', flow: true });
      edge(svg, [Ix.r + 4, Ix.cy], [Rx.x - 6, Rx.cy], { label: `α = ${fmt(s.alpha)}`, below: `${fmt(1 / s.alpha, 1)} days ill`, width: widthFor(s.alpha, 1), cls: 'st3', flow: true });
      if (s.vaccinate > 0) {
        const V = node(svg, { x: 10, y: 156, w: 124, h: 48, title: 'Vaccinated', sub: `${s.vaccinate}% / day` });
        edge(svg, [Sx.cx, Sx.b + 4], [V.cx, V.y - 4], { width: 2, cls: 'st4' });
      }
      if (s.lockdown) text(svg, 300, 172, `lockdown day ${s.lock_start}–${s.lock_start + s.lock_days}: β × ${fmt(cut, 2)}`, 'edge-lbl');
      text(svg, 300, 28, `R₀ = β/α = ${fmt(s.beta / s.alpha, 2)}`, 'n-title');
    },
  },
  charts: [{
    id: 'sir', title: 'The outbreak, day by day', height: 320,
    sub: 'Dots are the real counts of boys in bed. Lines are the SIR model solved with RK4.',
    legend: (d, s) => [['s1', 'line', 'Susceptible'], ['s2', 'line', 'Infected'], ['s3', 'line', 'Recovered'], ...(s.vaccinate > 0 ? [['s4', 'dash', 'Vaccinated']] : []), ['s2', 'dot', 'In bed, 1978']],
    draw(host, d, s) {
      const P = frame(host, { h: 320, x: [0, 20], y: [0, 780], xt: niceTicks(0, 20, 10), xl: 'Day of outbreak', yl: 'Boys' });
      if (s.lockdown) { const r = P.band(0, 780); r.setAttribute('x', P.X(s.lock_start)); r.setAttribute('width', Math.max(0, P.X(Math.min(20, s.lock_start + s.lock_days)) - P.X(s.lock_start))); }
      P.line(zip(d.t, d.S), 's1'); P.line(zip(d.t, d.R), 's3'); P.line(zip(d.t, d.I), 's2');
      if (s.vaccinate > 0) P.line(zip(d.t, d.V), 's4 dashed');
      d.data.day.forEach((x, i) => P.dot(x, d.data.in_bed[i], 'f2', 4.5));
      hover(host, P, px => { const i = nearest(d.t, P.xInv(px)); const day = Math.round(d.t[i]); return { cross: P.X(d.t[i]), pts: [[P.X(d.t[i]), P.Y(d.I[i]), 'f2']], html: tip(`Day ${fmt(d.t[i], 1)}`, [['Susceptible', int(d.S[i])], ['Infected', int(d.I[i])], ['Recovered', int(d.R[i])]].concat(Math.abs(d.t[i] - day) < 0.06 && day >= 1 && day <= 14 ? [['In bed, 1978', d.data.in_bed[day - 1]]] : [])) }; });
    },
  }],
  insights: (d, s) => [
    `With R₀ = <b>${fmt(d.r0, 2)}</b>, each early case infects about ${fmt(d.r0, 1)} others; the outbreak burns out once too few susceptible boys remain, after infecting <b>${pct(d.share_infected, 0)}</b> of the school.`,
    d.rmse < 20 ? `This setting fits the 1978 counts closely (RMSE ${fmt(d.rmse, 1)} boys).` : `Press <b>Fit β and α</b> to let the server find the parameters that match the 1978 counts.`,
    s.lockdown ? `The lockdown cuts contacts by ${s.lock_cut}% for ${s.lock_days} days from day ${s.lock_start}. Starting earlier lowers the peak far more than starting later.` : 'Turn on a lockdown or vaccination to see how the peak and the total change.',
  ],
  maths: { eq: "S' = −β S I / N − v S\nI' =  β S I / N − α I\nR' =  α I\nR₀ = β / α", notes: ['Integrated with fourth-order Runge–Kutta (h = 0.01 day). The fit minimises squared error against the 14 daily bed counts with a trust-region least-squares solver.'] },
};

/* ================================================================ Labs 7-8: rumour */
const rumour = {
  id: 'rumour', group: 'spread', nav: 'Rumour', tag: 'Labs 7–8', title: 'How a rumour spreads and stops',
  question: 'A rumour dies when the people spreading it keep meeting people who already know. How far does it get?',
  params: [
    { k: 'beta', label: 'Spreading β', min: 0.05, max: 1, step: 0.01, value: 0.3, group: 'Rumour', help: 'Rate at which a spreader tells someone who has not heard it.' },
    { k: 'gamma', label: 'Stifling γ', min: 0.02, max: 0.5, step: 0.01, value: 0.1, group: 'Rumour', help: 'Rate at which a spreader stops after meeting someone who already knows.' },
    { k: 'alpha', label: 'Education campaign α', min: 0, max: 0.1, step: 0.005, value: 0, unit: '/day', group: 'Interventions', help: 'Share of people who have not heard it who are taught to ignore it, each day.' },
    { k: 'newcomers', label: 'Newcomers', min: 0, max: 10, step: 1, value: 0, unit: '/ day', group: 'Interventions', help: 'People joining the population each day.' },
  ],
  presets: [
    { label: 'Classic', values: { beta: 0.3, gamma: 0.1, alpha: 0, newcomers: 0 } },
    { label: 'Quick to stop', values: { beta: 0.3, gamma: 0.3, alpha: 0, newcomers: 0 } },
    { label: 'Education campaign', values: { beta: 0.3, gamma: 0.1, alpha: 0.05, newcomers: 0 } },
    { label: 'Growing town', values: { beta: 0.3, gamma: 0.1, alpha: 0, newcomers: 5 } },
  ],
  request: (s, sig) => get('/api/spread/rumour', s, sig),
  kpis: d => [
    { label: 'Peak spreaders', value: int(d.peak_spreaders), note: d.theory.s_max != null ? `theory ${int(d.theory.s_max)} · day ${fmt(d.peak_day, 0)}` : `on day ${fmt(d.peak_day, 0)}` },
    { label: 'Ever heard it', value: pct(d.reach, 0), note: d.theory.r_inf != null ? `theory ${pct(d.theory.r_inf, 0)}` : '' },
    { label: 'R₀ = 1 + β/γ', value: fmt(d.theory.R0, 2) },
    { label: 'Conserved quantity drift', value: d.conserved_drift == null ? 'n/a' : d.conserved_drift.toExponential(1), tone: d.conserved_drift == null ? null : d.conserved_drift < 1e-6 ? 'good' : 'warn', note: d.conserved_drift == null ? 'only without campaign or newcomers' : 'solver accuracy check' },
  ],
  illustration: {
    title: 'Three kinds of people', height: 200,
    draw(svg, d, s) {
      const pk = d ? nearest(d.t, d.peak_day) : null;
      const v = a => d ? int(a[pk]) : null;
      const I = node(svg, { x: 10, y: 60, w: 124, h: 70, title: 'Never heard', value: v(d?.ignorant) });
      const Sp = node(svg, { x: 238, y: 60, w: 124, h: 70, title: 'Spreading', value: v(d?.spreader), hot: true });
      const St = node(svg, { x: 466, y: 60, w: 124, h: 70, title: 'Stopped', value: v(d?.stifler) });
      edge(svg, [I.r + 4, I.cy], [Sp.x - 6, Sp.cy], { label: `β = ${fmt(s.beta)}`, below: 'is told', width: widthFor(s.beta, 1), cls: 'st2', flow: true });
      edge(svg, [Sp.r + 4, Sp.cy], [St.x - 6, St.cy], { label: `γ = ${fmt(s.gamma)}`, below: 'meets a knower', width: widthFor(s.gamma, 0.5), cls: 'st3', flow: true });
      if (s.alpha > 0) edge(svg, [I.cx, I.b + 4], [St.cx, St.b + 4], { label: `campaign α = ${fmt(s.alpha, 3)}`, width: widthFor(s.alpha, 0.1), cls: 'st4', curve: [0, 70], labelAt: [300, 188] });
      if (s.newcomers > 0) edge(svg, [I.cx, 10], [I.cx, I.y - 6], { label: `+${s.newcomers}/day`, labelAt: [I.cx + 44, 30], width: 2 });
      if (d) text(svg, 300, 36, 'values at the peak', 'n-sub');
    },
  },
  charts: [{
    id: 'rum', title: 'The rumour over 100 days', height: 300, sub: 'Without a campaign or newcomers, the dashed line is the closed-form peak.',
    legend: [['s1', 'line', 'Never heard'], ['s2', 'line', 'Spreading'], ['s3', 'line', 'Stopped']],
    draw(host, d) {
      const [, hi] = extent(d.ignorant, d.spreader, d.stifler);
      const P = frame(host, { h: 300, x: [0, 100], y: [0, hi * 1.05], xl: 'Days', yl: 'People' });
      P.line(zip(d.t, d.ignorant), 's1'); P.line(zip(d.t, d.stifler), 's3'); P.line(zip(d.t, d.spreader), 's2');
      if (d.theory.s_max != null) { P.hline(d.theory.s_max); P.text(P.w - P.M.r - 4, P.Y(d.theory.s_max) - 5, 'theory: peak spreaders', { 'text-anchor': 'end' }); }
      hover(host, P, px => { const i = nearest(d.t, P.xInv(px)); return { cross: P.X(d.t[i]), pts: [[P.X(d.t[i]), P.Y(d.spreader[i]), 'f2']], html: tip(`Day ${fmt(d.t[i], 0)}`, [['Never heard', int(d.ignorant[i])], ['Spreading', int(d.spreader[i])], ['Stopped', int(d.stifler[i])]]) }; });
    },
  }],
  insights: d => [
    d.theory.s_max != null ? `The solver's peak (${int(d.peak_spreaders)}) lands on the closed-form peak (${int(d.theory.s_max)}), derived from a quantity the equations conserve.` : 'With a campaign or newcomers the conserved quantity no longer holds, so there is no closed-form peak.',
    `The rumour reaches <b>${pct(d.reach, 0)}</b> of people even though every spreader eventually stops: stopping depends on meeting people who know, which is rare early on.`,
  ],
  maths: { eq: "i' = −β s i / N − α i + Λ\ns' =  β s i / N − γ s (s + r) / N\nr' =  γ s (s + r) / N + α i\ns_max = 1 − (1/c)(1 + ln i₀ − i₀ + ln R₀),  c = β/γ", notes: ['i: never heard, s: spreading, r: stopped. Solved with RK4 (h = 0.05 day).'] },
};

/* ================================================================ Lab 9: chance */
const chance = {
  id: 'chance', group: 'spread', nav: 'Monte Carlo', tag: 'Lab 9', title: 'Monte Carlo: radioactive decay and coin tosses',
  question: 'Each atom decays at a random moment. Why does the average of many runs follow a smooth exponential?',
  params: [
    { k: 'lam', label: 'Decay constant λ', min: 0.01, max: 0.3, step: 0.005, value: 0.05, unit: '/s', group: 'Decay', help: 'Probability per second that an atom decays.' },
    { k: 'runs', label: 'Repeat the experiment', min: 10, max: 500, step: 10, value: 100, unit: 'runs', group: 'Decay', help: 'More runs make the average smoother.' },
    { k: 'n', label: 'Coin tosses', min: 100, max: 50000, step: 100, value: 5000, group: 'Coins' },
  ],
  hidden: { seed: 9 },
  presets: [
    { label: 'Fast decay', values: { lam: 0.2 } }, { label: 'Slow decay', values: { lam: 0.02 } },
    { label: 'Few runs', values: { runs: 10 } }, { label: 'Many runs', values: { runs: 500 } },
  ],
  request: async (s, sig) => {
    const [a, b] = await Promise.all([call('/api/spread/decay', { lam: s.lam, runs: s.runs, seed: s.seed }, { signal: sig }), call('/api/spread/coins', { n: s.n, seed: s.seed }, { signal: sig })]);
    return Object.assign(a, { coins: b, meta: { compute_ms: a.meta.compute_ms + b.meta.compute_ms, cached: false } });
  },
  actions: [{ label: 'Draw new random numbers', icon: 'dice', run(ctx) { ctx.state.seed = Math.floor(Math.random() * 1e9); return ctx.now(); } }],
  kpis: d => [
    { label: 'Half-life, measured', value: d.half_life_measured == null ? '> 100' : d.half_life_measured, unit: 's', note: `theory ln 2 / λ = ${fmt(d.half_life_theory, 1)} s`, tone: d.half_life_measured != null && Math.abs(d.half_life_measured - d.half_life_theory) <= 1 ? 'good' : null },
    { label: 'Spread at the half-life', value: '±' + fmt(d.sd_at_half_life, 1), unit: 'atoms', note: `binomial theory ±${fmt(d.sd_theory_at_half_life, 1)}` },
    { label: 'Worst gap, mean vs theory', value: fmt(d.max_gap_to_theory, 1), unit: 'atoms' },
    { label: 'Coins after the last toss', value: pct(Math.min(...d.coins.final), 1) + '–' + pct(Math.max(...d.coins.final), 1), note: 'share of heads, six sequences' },
  ],
  illustration: {
    title: 'One run, at the half-life', height: 210,
    caption: 'Each square is an atom from the first run. Orange ones have decayed by t = ln 2 / λ; about half remain.',
    draw(svg, d, s) {
      const th = Math.round(Math.LN2 / s.lam), cols = 50, rows = 20, cell = 11, ox = (600 - cols * cell) / 2;
      for (let i = 0; i < 1000; i++) {
        const decayed = d ? d.first_run_decay_times[i] <= th : false;
        S('rect', { x: ox + (i % cols) * cell, y: 8 + Math.floor(i / cols) * cell, width: cell - 2, height: cell - 2, rx: 2, class: decayed ? 'fill2' : 'fill1', opacity: decayed ? 0.85 : 0.35 }, svg);
      }
      if (d) text(svg, 300, 8 + rows * cell + 16, `t = ${th} s · ${d.first_run_decay_times.filter(t => t > th).length} of 1,000 left`, 'n-sub');
    },
  },
  chartLayout: 'two',
  charts: [
    { id: 'decay', title: 'Atoms remaining', sub: 'Grey: individual runs. Blue band: mean ± one standard deviation.', legend: [['s1', 'line', 'Mean of runs'], ['s2', 'dash', 'N₀e^−λt'], ['neutral', 'line', 'Single runs']],
      draw(host, d) {
        const P = frame(host, { h: 260, x: [0, 100], y: [0, 1000], xl: 'Seconds', yl: 'Atoms' });
        d.runs_shown.forEach(r => P.line(zip(d.t, r), 'sn thin').setAttribute('opacity', '.3'));
        P.area(zip(d.t, d.mean.map((m, i) => m + d.sd[i])), zip(d.t, d.mean.map((m, i) => m - d.sd[i])), 'band');
        P.line(zip(d.t, d.theory), 's2 dashed'); P.line(zip(d.t, d.mean), 's1');
        hover(host, P, px => { const i = nearest(d.t, P.xInv(px)); return { cross: P.X(i), pts: [[P.X(i), P.Y(d.mean[i]), 'f1']], html: tip(`t = ${i} s`, [['Mean of runs', fmt(d.mean[i], 1)], ['± sd', fmt(d.sd[i], 1)], ['N₀e^−λt', fmt(d.theory[i], 1)]]) }; });
      } },
    { id: 'coins', title: 'Share of heads as tosses add up', sub: 'Six fair-coin sequences settling toward 0.5 inside the 95% band.', legend: [['neutral', 'box', '95% band for a fair coin']],
      draw(host, d) {
        const c = d.coins, N = c.n[c.n.length - 1];
        const P = frame(host, { h: 260, x: [1, N], xlog: true, y: [0, 1], xt: [1, 10, 100, 1000, 10000, 100000].filter(v => v <= N), xf: v => int(v), xl: 'Tosses (log scale)', yl: 'Share of heads' });
        P.area(zip(c.n, c.band.map(b => 0.5 + b)), zip(c.n, c.band.map(b => 0.5 - b)), 'band-null'); P.hline(0.5);
        c.share.forEach((sq, i) => P.line(zip(c.n, sq), ['s1', 's2', 's3', 's4'][i % 4] + ' thin'));
      } },
  ],
  insights: d => [
    'Nothing in the simulation says "exponential". Each atom just gets a random decay time, yet the average of the runs traces <b>N₀e<sup>−λt</sup></b>.',
    `Single runs scatter by about ±${fmt(d.sd_at_half_life, 0)} atoms at the half-life, exactly the binomial spread √(N p (1 − p)).`,
    'The coin sequences wander early, then settle inside the shrinking 1.96 · 0.5 / √n band: the law of large numbers.',
  ],
  maths: { eq: 'P(atom survives to t) = e^(−λt)\nE[N(t)] = N₀ e^(−λt),   sd = √(N₀ e^(−λt)(1 − e^(−λt)))\nt½ = ln 2 / λ', notes: ['Each atom\'s decay step is ⌈−ln(1 − u) / λ⌉ for a uniform random u.'] },
};

/* ================================================================ Lab 10: random walk */
const walk = {
  id: 'walk', group: 'spread', nav: 'Random walk', tag: 'Lab 10', title: 'Random walks in one and two dimensions',
  question: 'Each step is a coin flip. Where do thousands of walkers end up, and how fast do they spread out?',
  params: [
    { k: 'p', label: 'Step right with probability p', min: 0, max: 1, step: 0.01, value: 0.5, group: '1-D walk', help: 'p = 0.5 is a fair walk; above 0.5 it drifts right.' },
    { k: 'steps', label: 'Steps', min: 10, max: 500, step: 10, value: 200, group: '1-D walk' },
    { k: 'walkers', label: 'Walkers', min: 100, max: 5000, step: 100, value: 2000, group: '1-D walk', help: 'More walkers make the histogram closer to the theory.' },
  ],
  hidden: { seed: 10 },
  presets: [
    { label: 'Fair coin', values: { p: 0.5, steps: 200 } },
    { label: 'Biased 60 / 40', values: { p: 0.6, steps: 200 } },
    { label: 'Long walk', values: { p: 0.5, steps: 500 } },
  ],
  request: async (s, sig) => {
    const [a, b] = await Promise.all([call('/api/spread/walk', { p: s.p, steps: s.steps, walkers: s.walkers, seed: s.seed }, { signal: sig }), call('/api/spread/walk2d', { seed: s.seed }, { signal: sig })]);
    return Object.assign(a, { w2: b, meta: { compute_ms: a.meta.compute_ms + b.meta.compute_ms, cached: false } });
  },
  actions: [{ label: 'Draw new random numbers', icon: 'dice', run(ctx) { ctx.state.seed = Math.floor(Math.random() * 1e9); return ctx.now(); } }],
  kpis: (d, s) => {
    const N = d.n.length - 1, w = d.w2;
    return [
      { label: 'Mean position', value: fmt(d.mean[N], 2), note: `theory n(2p − 1) = ${fmt(d.mean_theory[N], 2)}` },
      { label: 'Variance', value: fmt(d.var[N], 1), note: `theory 4np(1 − p) = ${fmt(d.var_theory[N], 1)}`, tone: d.var_theory[N] > 0 && Math.abs(d.var[N] / d.var_theory[N] - 1) < 0.1 ? 'good' : null },
      { label: 'Typical distance', value: fmt(Math.sqrt(d.var[N]), 1), unit: 'steps', note: `after ${s.steps} steps: grows like √n` },
      { label: '2-D mean squared distance', value: fmt(w.msd[w.msd.length - 1], 0), note: `theory 1.5 n = ${fmt(w.msd_theory[w.msd_theory.length - 1], 0)}` },
    ];
  },
  illustration: {
    title: 'One step at a time', height: 190,
    caption: 'The bell is where walkers should end up after n steps: centred on n(2p − 1), with width 2√(4np(1 − p)).',
    draw(svg, d, s) {
      const n = s.steps, mu = n * (2 * s.p - 1), sd = Math.sqrt(4 * n * s.p * (1 - s.p)) || 0.5;
      const span = Math.max(Math.abs(mu) + 3.5 * sd, 10), X = v => 300 + v / span * 270;
      S('line', { x1: 30, x2: 570, y1: 140, y2: 140, class: 'ln-m', 'stroke-width': 1.5 }, svg);
      const pts = []; for (let v = -span; v <= span; v += span / 80) pts.push([X(v), 140 - 90 * Math.exp(-((v - mu) ** 2) / (2 * sd * sd))]);
      S('path', { d: 'M' + pts.map(p => p.map(q => q.toFixed(1)).join(',')).join('L'), fill: 'none', class: 'st1', 'stroke-width': 2.5 }, svg);
      S('rect', { x: X(mu - 2 * sd), y: 146, width: X(mu + 2 * sd) - X(mu - 2 * sd), height: 6, rx: 3, class: 'fill1', opacity: 0.3 }, svg);
      text(svg, X(mu), 172, `95% of walkers: ${fmt(mu - 2 * sd, 0)} to ${fmt(mu + 2 * sd, 0)}`, 'n-sub');
      S('circle', { cx: X(0), cy: 140, r: 7, class: 'fill2' }, svg);
      edge(svg, [X(0) - 10, 118], [X(0) - 50, 118], { label: `1 − p = ${fmt(1 - s.p)}`, width: widthFor(1 - s.p, 1), labelAt: [X(0) - 40, 106] });
      edge(svg, [X(0) + 10, 118], [X(0) + 50, 118], { label: `p = ${fmt(s.p)}`, width: widthFor(s.p, 1), labelAt: [X(0) + 40, 106] });
      text(svg, X(0), 30, 'start', 'n-sub');
    },
  },
  chartLayout: 'two',
  charts: [
    { id: 'paths', title: 'Forty walkers and the theory', sub: 'Dashed: n(2p − 1) ± 2 standard deviations.', legend: [['s1', 'line', 'Mean of all walkers'], ['s2', 'dash', 'Theory ± 2 sd'], ['neutral', 'line', 'Walkers']],
      draw(host, d, s) {
        const [lo, hi] = extent(...d.paths, d.mean_theory);
        const P = frame(host, { h: 260, x: [0, s.steps], y: [Math.min(lo, -5), Math.max(hi, 5)], xl: 'Step', yl: 'Position' });
        d.paths.forEach(r => P.line(zip(d.n, r), 'sn thin').setAttribute('opacity', '.3'));
        const sdT = d.var_theory.map(Math.sqrt);
        P.line(zip(d.n, d.mean_theory.map((m, i) => m + 2 * sdT[i])), 's2 dashed'); P.line(zip(d.n, d.mean_theory.map((m, i) => m - 2 * sdT[i])), 's2 dashed');
        P.line(zip(d.n, d.mean), 's1');
        hover(host, P, px => { const i = nearest(d.n, P.xInv(px)); return { cross: P.X(i), html: tip(`Step ${i}`, [['Mean', fmt(d.mean[i], 2)], ['Theory', fmt(d.mean_theory[i], 2)], ['Variance', fmt(d.var[i], 1)], ['Theory', fmt(d.var_theory[i], 1)]]) }; });
      } },
    { id: 'hist', title: 'Where the walkers end up', sub: 'Final positions against the normal approximation.', legend: [['s1', 'box', 'Walkers'], ['s2', 'line', 'Normal approximation']],
      draw(host, d) {
        const vals = d.final_values, cnt = d.final_counts, edges = vals.map(v => v - 1).concat([vals[vals.length - 1] + 1]);
        const P = frame(host, { h: 260, x: [edges[0] - 2, edges[edges.length - 1] + 2], y: [0, Math.max(...cnt, ...(d.normal_expected || [0])) * 1.1], xl: 'Final position', yl: 'Walkers' });
        P.bars(edges, cnt, 'f1');
        if (d.normal_expected) P.line(zip(vals, d.normal_expected), 's2');
        hover(host, P, px => { const i = nearest(vals, P.xInv(px)); return { html: tip(`Position ${vals[i]}`, [['Walkers', cnt[i]], ['Normal approximation', d.normal_expected ? fmt(d.normal_expected[i], 1) : '—']]) }; });
      } },
    { id: 'w2', title: 'Twelve walkers in two dimensions', sub: 'Eight-neighbour steps from the origin.', legend: [],
      draw(host, d) {
        const w = d.w2, all = w.paths.flat(), ext = Math.max(10, ...all.map(p => Math.max(Math.abs(p[0]), Math.abs(p[1])))) * 1.05;
        const size = Math.min(host.clientWidth, 340);
        const P = frame(host, { h: size, x: [-ext, ext], y: [-ext, ext], m: { l: 36, r: 8, t: 8, b: 24 } });
        w.paths.forEach((path, i) => P.line(path, ['s1', 's2', 's3', 's4'][i % 4] + ' thin'));
        P.dot(0, 0, 'fn', 4);
      } },
    { id: 'msd', title: 'Mean squared distance in 2-D', sub: 'Grows linearly with the number of steps: diffusion.', legend: [['s1', 'line', 'Simulated'], ['s2', 'dash', 'Theory 1.5 n']],
      draw(host, d) {
        const w = d.w2, n = w.msd.map((_, i) => i);
        const P = frame(host, { h: 260, x: [0, n.length - 1], y: [0, Math.max(...w.msd, ...w.msd_theory) * 1.05], xl: 'Step', yl: 'Mean r²' });
        P.line(zip(n, w.msd_theory), 's2 dashed'); P.line(zip(n, w.msd), 's1');
      } },
  ],
  insights: (d, s) => [
    `After ${s.steps} steps the typical walker is only about <b>${fmt(Math.sqrt(d.var[d.var.length - 1]), 0)} steps</b> from the start: distance grows like √n, not n.`,
    'The histogram is close to a bell curve: the central limit theorem at work on a sum of coin flips.',
    'This is the null model the Crossover study tests stock prices against.',
  ],
  maths: { eq: 'Xₙ = ξ₁ + … + ξₙ,  ξ = +1 with prob. p, −1 otherwise\nE[Xₙ] = n(2p − 1),   Var[Xₙ] = 4np(1 − p)\n2-D, 8 neighbours:  E[|Xₙ|²] = 1.5 n', notes: [] },
};

export default [dosing, bass, sir, rumour, chance, walk];
