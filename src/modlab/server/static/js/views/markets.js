// Pillar 3 · Markets: walk-forward forecasting of volatility and returns, and neural nets as background jobs.
import { call, fdate, fmt, int, pct } from '../api.js';
import { E, extent, frame, hover, nearest, zip } from '../plot.js';
import { seg } from '../ui/dom.js';
import { S, edge, node, spark, text, widthFor } from '../ui/illustrate.js';

const tip = (title, rows) => `<b>${title}</b>` + rows.map(([k, v]) => `<br>${k}: <span class="n">${v}</span>`).join('');
const yearOf = s => +s.slice(0, 4) + (+s.slice(5, 7) - 1) / 12 + (+s.slice(8, 10)) / 365;
const pval = p => p == null ? '—' : p < 0.001 ? p.toExponential(1) : fmt(p, 3);
const kupTone = p => p == null ? null : p < 0.01 ? 'bad' : p < 0.05 ? 'warn' : 'good';
const r2pct = v => v == null ? '—' : fmt(v * 100, 2);

const END = 2026;
const TICKERS = [['SPY', 'SPY'], ['AAPL', 'AAPL'], ['MSFT', 'MSFT'], ['NVDA', 'NVDA'], ['JPM', 'JPM'], ['XOM', 'XOM']];
const DATA_START = { NVDA: 1999 };
const NAMES = { zero: 'Zero (random walk)', hist_mean: 'Historical mean', arima: 'ARIMA', ridge: 'Ridge', gbm: 'Gradient boosting', lstm: 'LSTM', gru: 'GRU',
  hist21: '21-day history', ewma: 'EWMA', garch: 'GARCH(1,1)', har: 'HAR-RV', har_x: 'HAR-RV + VIX', buy_hold: 'Buy and hold' };
const SHORT = { hist21: '21-day', ewma: 'EWMA', garch: 'GARCH', har: 'HAR', har_x: 'HAR + VIX' };
const VOL_MODELS = ['hist21', 'ewma', 'garch', 'har', 'har_x'];
const RET_MODELS = ['hist_mean', 'ridge', 'gbm', 'arima'];
const COLOR = { hist_mean: 's1', ridge: 's2', gbm: 's3', arima: 's4', lstm: 's2', gru: 's3', buy_hold: 'sn' };
const legendColor = k => COLOR[k] === 'sn' ? 'neutral' : COLOR[k];
const picked = (s, list) => list.filter(m => s['m_' + m]);
const tickerParam = { k: 'ticker', type: 'seg', label: 'Ticker', value: 'SPY', options: TICKERS, group: 'Data',
  help: 'SPY tracks the S&P 500. The others are single stocks: Apple, Microsoft, Nvidia, JPMorgan and Exxon.' };
const startParam = help => ({ k: 'start_year', label: 'Test years from', min: 2008, max: 2025, step: 1, value: 2015, group: 'Data', help });
const sup = n => String(n).split('').map(c => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+c]).join('');
const alphaLabel = a => a >= 10 && Number.isInteger(Math.log10(a)) ? '10' + sup(Math.log10(a)) : String(a);

/* ---------------- shared drawing */
function hbars(host, rows, max, f, tipf, bestKey) {
  const rowH = 30, h = rows.length * rowH + 40, w = Math.max(260, host.clientWidth), lw = Math.min(140, w * 0.38);
  const P = frame(host, { h, x: [0, max], y: [0, rows.length], yt: [], m: { l: lw, r: 50, t: 6, b: 30 }, xf: f, xn: 4 });
  rows.forEach((r, i) => {
    const y = P.Y(i + 0.5), x0 = P.X(0), x1 = P.X(Math.max(r.v, 0));
    E('rect', { x: x0, y: y - 9, width: Math.max(2, x1 - x0), height: 18, rx: 4, class: r.k === bestKey ? 'f1' : 'fn', opacity: r.k === bestKey ? 1 : 0.6 }, P.body);
    P.text(lw - 10, y + 4, NAMES[r.k] || r.k, { 'text-anchor': 'end' });
    P.text(x1 + 6, y + 4, f(r.v), { style: 'font-family:var(--f-mono)' });
  });
  hover(host, P, (px, py) => { const i = Math.floor((P.h - P.M.b - py) / rowH); return i >= 0 && i < rows.length ? { html: tipf(rows[i]) } : null; });
}

function equity(host, eq, names) {
  const yrs = eq.dates.map(yearOf);
  const [lo, hi] = extent(...names.map(n => eq.curves[n]));
  const y0 = Math.min(lo, 1) * 0.92, y1 = hi * 1.08;
  const P = frame(host, { h: 280, x: [yrs[0], yrs[yrs.length - 1]], y: [y0, y1], ylog: true,
    yt: [0.25, 0.5, 1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000].filter(v => v >= y0 && v <= y1), yf: v => v + '×',
    yl: 'Growth of $1 (log scale)', xf: v => Math.round(v) });
  P.hline(1);
  names.forEach(n => P.line(zip(yrs, eq.curves[n]), COLOR[n] || 's4'));
  hover(host, P, px => {
    const i = nearest(yrs, P.xInv(px));
    return { cross: P.X(yrs[i]), pts: names.map(n => [P.X(yrs[i]), P.Y(eq.curves[n][i]), (COLOR[n] || 's4').replace('s', 'f')]),
      html: tip(fdate(eq.dates[i]), names.map(n => [NAMES[n] || n, fmt(eq.curves[n][i], 2) + '×'])) };
  });
}

function scoreTable(d, names) {
  const bt = d.backtest || {};
  const best = names.filter(k => k !== 'zero').reduce((a, k) => (a == null || d.models[k].r2_oos > d.models[a].r2_oos ? k : a), null);
  const row = (k, m, b) => `<tr${k === best && m.r2_oos > 0 ? ' class="best"' : ''}><td>${NAMES[k] || k}</td>`
    + `<td class="num">${m ? r2pct(m.r2_oos) + '%' : '—'}</td><td class="num">${m?.hit_rate == null ? '—' : pct(m.hit_rate)}</td>`
    + `<td class="num">${pval(m?.hit_p)}</td><td class="num">${pval(m?.dm_p)}</td>`
    + `<td class="num">${b?.cagr == null ? '—' : pct(b.cagr)}</td><td class="num">${b ? fmt(b.sharpe) : '—'}</td><td class="num">${b?.max_dd == null ? '—' : pct(b.max_dd, 0)}</td></tr>`;
  const body = names.map(k => row(k, d.models[k], bt[k])).join('') + (bt.buy_hold ? row('buy_hold', null, bt.buy_hold) : '');
  return `<div class="tbl-wrap"><table><thead><tr><th>Model</th><th class="num">R² vs zero</th><th class="num">Direction right</th><th class="num">p vs always up</th>`
    + `<th class="num">DM p vs zero</th><th class="num">CAGR</th><th class="num">Sharpe</th><th class="num">Max drawdown</th></tr></thead><tbody>${body}</tbody></table></div>`;
}

/* ================================================================ Volatility */
const volPick = (d, s) => {
  const avail = Object.keys(d.series.forecasts);
  return avail.includes(s.vol_show) ? s.vol_show : avail.includes('har_x') ? 'har_x' : avail[avail.length - 1];
};
const levelOf = a => pct(1 - a, 0);

const volatility = {
  id: 'volatility', group: 'markets', nav: 'Volatility', tag: 'GARCH · HAR', title: 'Forecasting tomorrow\'s volatility',
  question: 'Tomorrow\'s return is close to a coin flip, but the size of tomorrow\'s move is not. How well does each model forecast it, and does the risk estimate built on it hold up?',
  params: [
    tickerParam,
    startParam('Every year from here to 2026 is forecast by models refit only on data before that year.'),
    { k: 'm_hist21', type: 'switch', label: NAMES.hist21, value: true, group: 'Models', help: 'Average daily variance over the last 21 trading days. The simplest baseline.' },
    { k: 'm_ewma', type: 'switch', label: 'EWMA', value: true, group: 'Models', help: 'RiskMetrics: exponentially weighted squared returns with λ = 0.94.' },
    { k: 'm_garch', type: 'switch', label: NAMES.garch, value: true, group: 'Models', help: 'Fitted by maximum likelihood each year on the 12 years before it.' },
    { k: 'm_har', type: 'switch', label: NAMES.har, value: true, group: 'Models', help: 'A linear regression of tomorrow\'s log variance on yesterday, last week and last month.' },
    { k: 'm_har_x', type: 'switch', label: NAMES.har_x, value: true, group: 'Models', help: 'HAR-RV plus the VIX implied variance and a term for yesterday\'s fall.' },
    { k: 'var_model', type: 'seg', label: 'VaR built on', value: 'har_x', options: VOL_MODELS.map(m => [m, SHORT[m]]), group: 'Value at Risk',
      help: 'The variance forecast that sets each day\'s Value-at-Risk threshold.' },
    { k: 'var_alpha', type: 'seg', label: 'Confidence', value: 0.01, options: [[0.01, '99%'], [0.05, '95%']], group: 'Value at Risk',
      help: 'A 99% one-day VaR should be breached on 1% of days.' },
  ],
  hidden: { vol_show: 'har_x' },
  presets: [
    { label: 'S&P 500 since 2015', values: { ticker: 'SPY', start_year: 2015, m_hist21: true, m_ewma: true, m_garch: true, m_har: true, m_har_x: true, var_model: 'har_x' } },
    { label: 'Through the 2008 crash', values: { ticker: 'SPY', start_year: 2008, m_hist21: true, m_ewma: true, m_garch: true, m_har: true, m_har_x: true, var_model: 'har_x' } },
    { label: 'Nvidia', values: { ticker: 'NVDA', start_year: 2015, m_hist21: true, m_ewma: true, m_garch: true, m_har: true, m_har_x: true, var_model: 'har_x' } },
    { label: 'Classic models only', values: { ticker: 'SPY', start_year: 2015, m_hist21: true, m_ewma: true, m_garch: true, m_har: false, m_har_x: false, var_model: 'garch' } },
  ],
  request: async (s, sig) => {
    const models = picked(s, VOL_MODELS);
    if (!models.length) throw new Error('Pick at least one volatility model.');
    const [v, st] = await Promise.all([
      call('/api/markets/volatility', { ticker: s.ticker, models, start_year: s.start_year, end_year: END, var_alpha: s.var_alpha, var_model: s.var_model }, { signal: sig }),
      call('/api/markets/stylized', { ticker: s.ticker }, { signal: sig }),
    ]);
    return Object.assign(v, { sty: st, meta: { compute_ms: v.meta.compute_ms + st.meta.compute_ms, cached: v.meta.cached && st.meta.cached } });
  },
  kpis: (d, s) => {
    const rows = Object.entries(d.models).sort((a, b) => b[1].r2_log - a[1].r2_log);
    const [bk, bm] = rows[0], base = d.models.hist21 ? ['hist21', d.models.hist21] : rows[rows.length - 1];
    const V = d.var, lvl = levelOf(V.alpha);
    const varTile = (label, x) => ({ label, value: pct(x.rate, 2), tone: kupTone(x.kupiec_p),
      note: `target ${pct(V.alpha, 0)} · ${x.exceed} of ${int(x.n)} days · Kupiec p ${pval(x.kupiec_p)}` });
    return [
      { label: 'Best model, R² of log variance', value: pct(bm.r2_log, 1), tone: 'good', note: `${NAMES[bk]}, out of sample` },
      rows.length > 1
        ? { label: base[0] === 'hist21' ? 'Baseline: 21-day average' : 'Weakest model selected', value: pct(base[1].r2_log, 1),
          note: `the best model adds ${fmt((bm.r2_log - base[1].r2_log) * 100, 1)} points` }
        : { label: 'Correlation with realised', value: fmt(bm.corr, 2), note: 'pick more models to compare' },
      varTile(`${lvl} VaR breaches, normal tails`, V.normal),
      varTile(`${lvl} VaR breaches, filtered historical`, V.fhs),
    ];
  },
  illustration: {
    title: 'HAR-RV: three horizons and the VIX', height: 240,
    caption: 'Arrow widths follow the coefficients fitted for the latest test year. HAR models traders who react on daily, weekly and monthly horizons.',
    draw(svg, d, s) {
      const key = s.m_har_x ? 'har_x' : 'har';
      const byYear = d?.params?.har?.[key], last = byYear ? Object.keys(byYear).sort().pop() : null;
      const coef = last ? byYear[last] : null;
      const ins = [['Yesterday', 'RV of day t'], ['Last week', '5-day mean'], ['Last month', '22-day mean'], ['VIX', 'implied variance']];
      const syms = ['β_d', 'β_w', 'β_m', 'γ'];
      // Inputs in one column, 52 apart. The regression box spans the column, so each arrow
      // runs straight across and lands on its own point of the box's left edge.
      const reg = node(svg, { x: 252, y: 10, w: 118, h: 198, title: 'Regression', sub: last ? `refit for ${last}` : 'log variance' });
      ins.forEach(([t, sub], i) => {
        const off = i === 3 && key !== 'har_x';
        const g = off ? S('g', { opacity: 0.35 }, svg) : svg;
        const n = node(g, { x: 8, y: 10 + i * 52, w: 150, h: 42, title: t, sub });
        const c = coef && !off ? coef[i + 1] : null;
        edge(off ? g : svg, [n.r + 4, n.cy], [reg.x - 6, n.cy], { width: c == null ? 2 : Math.min(8, widthFor(Math.abs(c), 0.6)), cls: off ? '' : 'st1', arrow: !off });
        text(off ? g : svg, n.r + 8, n.cy - 9, off ? 'off' : c == null ? syms[i] : `${syms[i]} ${fmt(c, 2)}`, 'edge-lbl', 'start');
      });
      const sc = d?.models?.[key];
      const out = node(svg, { x: 448, y: reg.cy - 48, w: 144, h: 96, hot: true, title: 'Tomorrow', value: sc ? `R² ${pct(sc.r2_log, 0)}` : 'log RV', sub: `${s.ticker} variance` });
      edge(svg, [reg.r + 4, reg.cy], [out.x - 6, out.cy], { width: 3, flow: true, label: 'forecast' });
      text(svg, 8, 232, key === 'har_x' ? 'HAR-RV + VIX also adds a term for yesterday\'s fall (the leverage effect).' : 'Switch on HAR-RV + VIX to add the VIX input.', 'n-sub', 'start');
    },
  },
  chartLayout: 'two',
  charts: [
    { id: 'r2', title: 'R² of log variance, out of sample', height: 190,
      sub: d => `Next-day log realised variance, ${d.years[0]}–${d.years[d.years.length - 1]}, ${int(d.n_test)} days. QLIKE loss in the tooltip.`,
      legend: [['s1', 'box', 'Best'], ['neutral', 'box', 'Other models']],
      draw(host, d) {
        const rows = Object.entries(d.models).map(([k, v]) => ({ k, v: v.r2_log })).sort((a, b) => a.v - b.v);
        hbars(host, rows, Math.max(0.7, ...rows.map(r => r.v)), v => Math.round(v * 100) + '%',
          r => tip(NAMES[r.k], [['R² log variance', pct(d.models[r.k].r2_log)], ['QLIKE', fmt(d.models[r.k].qlike, 3)], ['Correlation', fmt(d.models[r.k].corr, 2)]]), rows[rows.length - 1].k);
      } },
    { id: 'acf', title: 'Why the size of a move is forecastable', height: 240,
      sub: d => `Autocorrelation of ${d.sty.ticker} daily returns and of their absolute size, ${d.sty.start.slice(0, 4)}–${d.sty.end.slice(0, 4)}. Inside the grey band is noise.`,
      legend: [['s1', 'line', 'Returns'], ['s2', 'line', 'Absolute returns'], ['neutral', 'box', '95% noise band']],
      draw(host, d) {
        const st = d.sty, b = st.band, [lo, hi] = extent(st.acf_r, st.acf_abs);
        const P = frame(host, { h: 240, x: [1, st.lags[st.lags.length - 1]], y: [Math.min(lo, -b) - 0.03, hi + 0.03], xl: 'Lag (trading days)', yl: 'Autocorrelation' });
        P.area(st.lags.map(l => [l, b]), st.lags.map(l => [l, -b]), 'band-null'); P.hline(0);
        P.line(zip(st.lags, st.acf_r), 's1'); P.line(zip(st.lags, st.acf_abs), 's2');
        hover(host, P, px => { const i = nearest(st.lags, P.xInv(px)); return { cross: P.X(st.lags[i]), pts: [[P.X(st.lags[i]), P.Y(st.acf_r[i]), 'f1'], [P.X(st.lags[i]), P.Y(st.acf_abs[i]), 'f2']],
          html: tip(`Lag ${st.lags[i]} days`, [['Returns', fmt(st.acf_r[i], 3)], ['Absolute returns', fmt(st.acf_abs[i], 3)], ['Noise band', '±' + fmt(b, 3)]]) }; });
      } },
    { id: 'series', title: 'Realised and forecast variance', height: 280,
      sub: 'Log daily variance on each test day. Hover for annualised volatility. Pick the model above the chart.',
      legend: (d, s) => [['neutral', 'line', 'Realised'], ['s2', 'line', NAMES[volPick(d, s)]]],
      tools(el, ctx) {
        const sg = seg(VOL_MODELS.map(m => [m, SHORT[m]]), ctx.state.vol_show, v => { ctx.state.vol_show = v; ctx.redraw(); }, { label: 'Forecast shown' });
        ctx.extra.volSeg = sg; el.append(sg);
      },
      draw(host, d, s, ctx) {
        const S2 = d.series, show = volPick(d, s), f = S2.forecasts[show], yrs = S2.dates.map(yearOf);
        const sg = ctx?.extra?.volSeg;
        if (sg) { sg.set(show); [...sg.children].forEach((b, i) => { b.hidden = !(VOL_MODELS[i] in S2.forecasts); }); }
        const step = Math.max(1, Math.floor(yrs.length / 1500)), idx = yrs.map((_, i) => i).filter(i => i % step === 0);
        const [lo, hi] = extent(idx.map(i => S2.realized[i]), idx.map(i => f[i]));
        const P = frame(host, { h: 280, x: [yrs[0], yrs[yrs.length - 1]], y: [lo, hi], yl: 'Log daily variance', xf: v => Math.round(v) });
        P.line(idx.map(i => [yrs[i], S2.realized[i]]), 'sn thin').setAttribute('opacity', '.6');
        P.line(idx.map(i => [yrs[i], f[i]]), 's2');
        const vol = v => pct(Math.sqrt(Math.exp(v) * 252), 0);
        hover(host, P, px => { const i = nearest(yrs, P.xInv(px)); return { cross: P.X(yrs[i]), pts: [[P.X(yrs[i]), P.Y(f[i]), 'f2']],
          html: tip(fdate(S2.dates[i]), [['Realised, annualised', vol(S2.realized[i])], [`${SHORT[show]} forecast`, vol(f[i])]]) }; });
      } },
    { id: 'var', title: 'Value at Risk backtest', height: 280,
      sub: d => `Next-day returns against the ${levelOf(d.var.alpha)} VaR from ${NAMES[d.var.model]}. Dots mark the days the loss went past the threshold.`,
      legend: [['neutral', 'line', 'Daily return'], ['s2', 'line', 'VaR, normal tails'], ['s3', 'line', 'VaR, filtered historical'], ['s2', 'dot', 'Breach, normal'], ['s3', 'dot', 'Breach, filtered']],
      draw(host, d) {
        const S2 = d.series, yrs = S2.dates.map(yearOf), r = S2.next_return.map(v => v * 100);
        const vn = S2.var_normal.map(v => v * 100), vf = S2.var_fhs.map(v => v * 100);
        const [lo, hi] = extent(r, vn, vf);
        const P = frame(host, { h: 280, x: [yrs[0], yrs[yrs.length - 1]], y: [lo * 1.05, hi * 1.05], yl: 'Next-day return (%)', yf: v => v + '%', xf: v => Math.round(v) });
        P.hline(0);
        P.line(zip(yrs, r), 'sn thin').setAttribute('opacity', '.55');
        P.line(zip(yrs, vn), 's2 thin'); P.line(zip(yrs, vf), 's3 thin');
        r.forEach((v, i) => { if (v < vn[i]) P.dot(yrs[i], v, 'f2', 3.5); });
        r.forEach((v, i) => { if (v < vf[i]) P.dot(yrs[i], v, 'f3', 2.5); });
        hover(host, P, px => { const i = nearest(yrs, P.xInv(px)); return { cross: P.X(yrs[i]), pts: [[P.X(yrs[i]), P.Y(r[i]), 'fn']],
          html: tip(fdate(S2.dates[i]), [['Return', fmt(r[i], 2) + '%'], ['VaR, normal', fmt(vn[i], 2) + '%'], ['VaR, filtered', fmt(vf[i], 2) + '%']]) }; });
      } },
  ],
  insights: d => {
    const rows = Object.entries(d.models).sort((a, b) => b[1].r2_log - a[1].r2_log), [bk, bm] = rows[0], st = d.sty, V = d.var;
    const word = x => x.kupiec_p < 0.05 ? (x.rate > V.alpha ? 'too often' : 'too rarely') : 'consistent with the target';
    return [
      `<b>${NAMES[bk]}</b> explains <b>${pct(bm.r2_log, 0)}</b> of the day-to-day variation in ${d.ticker}'s log variance out of sample` + (d.models.hist21 && bk !== 'hist21' ? `, against ${pct(d.models.hist21.r2_log, 0)} for a plain 21-day average.` : '.'),
      `Today's return says almost nothing about tomorrow's (lag-1 autocorrelation ${fmt(st.acf_r[0], 2)}), but the size of the move does: absolute returns correlate <b>${fmt(st.acf_abs[0], 2)}</b> at lag 1 and still ${fmt(st.acf_abs[st.acf_abs.length - 1], 2)} after ${st.lags.length} days. Calm and turbulent spells persist.`,
      `Returns have fat tails: ${st.tail_counts.beyond_4sd} days beyond 4 standard deviations, where a normal distribution expects ${fmt(st.tail_counts.normal_expected_beyond_4sd, 1)}. Normal tails breach the ${levelOf(V.alpha)} VaR on <b>${pct(V.normal.rate, 2)}</b> of days, ${word(V.normal)}. Filtered historical simulation breaches on <b>${pct(V.fhs.rate, 2)}</b>, ${word(V.fhs)}.`,
    ];
  },
  maths: { eq: 'target   log RVₜ₊₁,  RV = overnight gap² + Garman–Klass range\nHAR-RV   log RVₜ₊₁ = c + β_d log RVₜ + β_w log RV⁽⁵⁾ₜ + β_m log RV⁽²²⁾ₜ\n + VIX    … + γ log(VIX²ₜ / 252) + δ min(rₜ, 0)² / RV⁽²²⁾ₜ\nGARCH    σ²ₜ₊₁ = ω + α (rₜ − μ)² + β σ²ₜ\nEWMA     σ²ₜ₊₁ = 0.94 σ²ₜ + 0.06 r²ₜ\nVaR      breach if rₜ₊₁ < q · σ̂ₜ₊₁,  q = Φ⁻¹(α) or the α-quantile of r / σ̂ in sample',
    notes: ['R² is measured on log variance over all test days. The level models (21-day, EWMA, GARCH) get a constant log-bias correction fitted in sample. Kupiec\'s test compares the breach count with a binomial at the target rate: a small p says the rate is wrong.'] },
};

/* ================================================================ Returns */
// Best model by R², by hit rate and by Sharpe. Models that are long every day (the historical
// mean, nearly always positive) are just buy and hold, so they are left out of the timing picks.
function retBest(d) {
  const ks = Object.keys(d.models).filter(k => k !== 'zero'), bt = d.backtest;
  const timers = ks.filter(k => !(bt[k]?.time_in_market > 0.999)), pool = timers.length ? timers : ks;
  const top = (list, f) => list.reduce((a, k) => (f(k) > f(a) ? k : a), list[0]);
  return { ks, bt, bh: bt.buy_hold, up: d.always_up_hit_rate,
    kr: top(ks, k => d.models[k].r2_oos), kh: top(pool, k => d.models[k].hit_rate ?? -1), kt: top(pool, k => bt[k].sharpe ?? -9) };
}

const returnsView = {
  id: 'returns', group: 'markets', nav: 'Returns', tag: 'Walk-forward', title: 'Forecasting the next day\'s return',
  question: 'Can any model forecast tomorrow\'s return well enough to beat buying and holding? Each change refits every model, year by year, on the server. That takes a few seconds.',
  params: [
    tickerParam,
    startParam('Each test year is forecast by models refit on all earlier data. Starting earlier means more refits and a longer wait.'),
    { k: 'm_hist_mean', type: 'switch', label: NAMES.hist_mean, value: true, group: 'Models', help: 'The average past return. It is almost always positive, so it says "up" every day.' },
    { k: 'm_ridge', type: 'switch', label: NAMES.ridge, value: true, group: 'Models', help: 'Penalised linear regression on 25 scale-free features: lagged returns, momentum, volatility ratios and the VIX. The penalty is chosen on the validation years.' },
    { k: 'm_gbm', type: 'switch', label: NAMES.gbm, value: true, group: 'Models', help: 'Boosted decision trees on the same features. Tree size, learning rate and number of trees are chosen on the validation years.' },
    { k: 'm_arima', type: 'switch', label: NAMES.arima, value: false, group: 'Models', help: 'ARIMA(p, 0, q) on daily returns, with the order chosen once by AIC. Adds a few seconds.' },
  ],
  debounce: 400,
  presets: [
    { label: 'S&P 500 since 2015', values: { ticker: 'SPY', start_year: 2015, m_hist_mean: true, m_ridge: true, m_gbm: true, m_arima: false } },
    { label: 'Nvidia', values: { ticker: 'NVDA', start_year: 2015, m_hist_mean: true, m_ridge: true, m_gbm: true, m_arima: false } },
    { label: 'Last five years', values: { ticker: 'SPY', start_year: 2022, m_hist_mean: true, m_ridge: true, m_gbm: true, m_arima: false } },
  ],
  request: (s, sig) => {
    const models = picked(s, RET_MODELS);
    if (!models.length) return Promise.reject(new Error('Pick at least one return model.'));
    return call('/api/markets/returns', { ticker: s.ticker, models, start_year: s.start_year, end_year: END }, { signal: sig });
  },
  kpis: d => {
    const { kr, kh, kt, bt, bh, up } = retBest(d), mr = d.models[kr], mh = d.models[kh], sh = bt[kt].sharpe;
    return [
      { label: 'Best R² vs forecasting zero', value: r2pct(mr.r2_oos), unit: '%', tone: mr.r2_oos > 0 && mr.dm_p < 0.05 ? 'good' : mr.r2_oos > 0 ? 'warn' : 'bad',
        note: `${NAMES[kr]} · Diebold–Mariano p ${pval(mr.dm_p)}` },
      { label: 'Best direction hit rate', value: pct(mh.hit_rate, 1), tone: mh.hit_rate > up && mh.hit_p < 0.05 ? 'good' : mh.hit_rate > up ? 'warn' : 'bad',
        note: `${NAMES[kh]} · always up ${pct(up, 1)}, p ${pval(mh.hit_p)}` },
      { label: 'Best timing Sharpe ratio', value: fmt(sh, 2), tone: sh > bh.sharpe ? 'good' : 'bad', note: `${NAMES[kt]} · buy and hold ${fmt(bh.sharpe, 2)}` },
      { label: 'Buy and hold', value: pct(bh.cagr, 1), unit: 'a year', note: `worst drawdown ${pct(bh.max_dd, 0)}` },
    ];
  },
  illustration: {
    title: 'Walk-forward: refit every year, test on the next', height: 230,
    caption: 'Each row is one refit. A model never sees a day of its test year, and its settings are chosen on the two validation years before it.',
    draw(svg, d, s) {
      const y0 = DATA_START[s.ticker] || 1993, years = [];
      for (let y = s.start_year; y <= END; y++) years.push(y);
      const X = y => 64 + (y - y0) / (END + 1 - y0) * 526, top = 34, avail = 160;
      const rowH = Math.min(16, avail / years.length), bh = Math.max(3, rowH * 0.72);
      const every = Math.ceil(13 / rowH);
      [y0, 2000, 2005, 2010, 2015, 2020, END].filter((v, i, a) => v >= y0 && a.indexOf(v) === i && (v === y0 || v - y0 >= 4)).forEach(v => {
        text(svg, X(v), 22, String(v));
        S('line', { x1: X(v), x2: X(v), y1: 27, y2: top + years.length * rowH, class: 'ln-m', 'stroke-dasharray': '2 4' }, svg);
      });
      const hp = d?.hyperparams;
      years.forEach((Y, i) => {
        const y = top + i * rowH + (rowH - bh) / 2;
        S('rect', { x: X(y0), y, width: Math.max(0, X(Y - 2) - X(y0)), height: bh, rx: 2, class: 'fill1', opacity: 0.3 }, svg);
        S('rect', { x: X(Y - 2), y, width: X(Y) - X(Y - 2), height: bh, rx: 2, class: 'fill4', opacity: 0.6 }, svg);
        S('rect', { x: X(Y), y, width: X(Y + 1) - X(Y), height: bh, rx: 2, class: 'fill2', opacity: 0.9 }, svg);
        if (i % every === 0 || i === years.length - 1) text(svg, 56, y + bh / 2 + 4, String(Y), 'n-sub', 'end');
        if (hp && rowH >= 12) {
          const parts = [hp.ridge?.[Y] && `ridge α ${alphaLabel(hp.ridge[Y].alpha)}`, hp.gbm?.[Y] && `${hp.gbm[Y].iters} trees`].filter(Boolean);
          if (parts.length) text(svg, X(y0) + 6, y + bh / 2 + 3.5, parts.join(' · '), 'n-sub', 'start').setAttribute('style', 'font-size:9.5px');
        }
      });
      const ly = 216;
      [['fill1', 0.3, 'train: all earlier years'], ['fill4', 0.6, 'validation: 2 years'], ['fill2', 0.9, `test: 1 year${d ? `, ${int(d.n_test)} days in all` : ''}`]].forEach(([c, o, t], i) => {
        const x = 64 + i * 168;
        S('rect', { x, y: ly - 9, width: 14, height: 10, rx: 2, class: c, opacity: o }, svg);
        text(svg, x + 20, ly, t, 'n-sub', 'start');
      });
    },
  },
  charts: [
    { id: 'table', kind: 'html', title: 'Scores, out of sample', height: 220,
      sub: d => `${fdate(d.test_start)} to ${fdate(d.test_end)}, ${int(d.n_test)} trading days. Each strategy holds the stock when the forecast is positive and cash otherwise, paying ${d.cost_bps} bp per trade.`,
      caption: d => `Always predicting "up" is right ${pct(d.always_up_hit_rate)} of the time over these years. The highlighted row has the best R².`,
      draw(host, d) { host.innerHTML = scoreTable(d, Object.keys(d.models)); } },
    { id: 'equity', title: 'Growth of $1', height: 280, sub: 'Log scale, sampled weekly. Timing strategies are either fully in the stock or in cash.',
      legend: d => [['neutral', 'line', 'Buy and hold'], ...Object.keys(d.equity.curves).filter(k => k !== 'buy_hold' && k !== 'zero').map(k => [legendColor(k), 'line', NAMES[k]])],
      draw(host, d) { equity(host, d.equity, ['buy_hold', ...Object.keys(d.equity.curves).filter(k => k !== 'buy_hold' && k !== 'zero')]); } },
  ],
  insights: d => {
    const { kr, kh, kt, bt, bh, up } = retBest(d), m = d.models[kr], mh = d.models[kh];
    const out = [
      `The best model, ${NAMES[kr]}, explains <b>${r2pct(m.r2_oos)}%</b> of the variation in next-day returns. ` + (m.dm_p > 0.05 ? `Its gain over forecasting zero (Diebold–Mariano p ${pval(m.dm_p)}) is consistent with luck.` : `Its gain over forecasting zero is unlikely to be luck (p ${pval(m.dm_p)}), but it is tiny.`),
      `Always predicting "up" is right on <b>${pct(up)}</b> of days. ${NAMES[kh]} is right on ${pct(mh.hit_rate)}` + (mh.hit_rate > up ? ` (p ${pval(mh.hit_p)} against always up).` : mh.hit_rate < up ? ', worse than always up.' : ', no better.'),
      `Timing the market with these forecasts earns a Sharpe ratio of <b>${fmt(bt[kt].sharpe, 2)}</b> at best (${NAMES[kt]}), against <b>${fmt(bh.sharpe, 2)}</b> for buy and hold.` + (bt.hist_mean?.time_in_market > 0.999 ? ' The historical mean is positive every day, so it simply holds the stock.' : ''),
    ];
    const ra = d.hyperparams?.ridge ? Object.values(d.hyperparams.ridge).map(v => v.alpha) : [];
    if (ra.length) {
      const big = ra.filter(a => a === Math.max(...ra)).length;
      out[0] += ` Ridge picks its strongest penalty (α = ${int(Math.max(...ra))}) in ${big} of ${ra.length} years, shrinking its forecast towards a constant.`;
    }
    return out;
  },
  maths: { eq: 'target     rₜ₊₁ = log(Pₜ₊₁ / Pₜ)\nlearned    yₜ = rₜ₊₁ / σ₆₃,ₜ   (clipped at ±5)\nR² vs zero 1 − Σ(r − r̂)² / Σ r²\nDM test    dₜ = r²ₜ − (rₜ − r̂ₜ)²,  t-statistic of mean(d)\nstrategy   long if r̂ₜ > 0, else cash, 5 bp per trade',
    notes: ['Ridge and gradient boosting learn the volatility-scaled return y on standardised features, and their forecasts are scaled back by σ₆₃. The Diebold–Mariano variance uses Newey–West with 5 lags. The hit-rate p tests the model against always predicting up. This is a study of forecasting, not investment advice.'] },
};

/* ================================================================ Neural nets */
const TRAP_NAMES = { naive: 'Tomorrow = today', leaky: 'Tutorial LSTM (scaler fitted on all prices)', clean: 'Same LSTM, leak fixed' };

const neural = {
  id: 'neural', group: 'markets', nav: 'Neural nets', tag: 'LSTM · GRU', title: 'Neural networks on prices and returns',
  question: 'A popular LSTM tutorial forecasts stock prices with an R² near 0.97. Is that better than saying tomorrow\'s price equals today\'s?',
  params: [
    tickerParam,
    { k: 'trap_epochs', label: 'Epochs', min: 1, max: 30, step: 1, value: 8, group: 'The LSTM trap', help: 'Passes over the training prices. Each epoch trains two networks, leaky and clean.' },
    { k: 'window', label: 'Window', min: 20, max: 120, step: 5, value: 60, unit: 'days', group: 'The LSTM trap', help: 'Days of past prices the network sees for each forecast.' },
    { k: 'net', type: 'seg', label: 'Network', value: 'lstm', options: [['lstm', 'LSTM'], ['gru', 'GRU']], group: 'Walk-forward on returns',
      help: 'LSTM and GRU are recurrent networks with gates that decide what to remember. The GRU has fewer gates.' },
    { k: 'rnn_epochs', label: 'Epochs', min: 1, max: 30, step: 1, value: 6, group: 'Walk-forward on returns', help: 'The most passes per yearly refit. Early stopping on the validation years can end sooner.' },
    { k: 'start_year', label: 'Test years from', min: 2008, max: 2025, step: 1, value: 2015, group: 'Walk-forward on returns', help: 'The network is refit for every year from here to 2026.' },
  ],
  emptyText: 'Press Train to fit the networks on the server. The trap takes about 15 s.',
  actions: [
    { label: 'Train the LSTM trap', icon: 'play', primary: true, async run(ctx) {
      const s = ctx.state;
      const r = await ctx.job({ kind: 'lstm_trap', ticker: s.ticker, epochs: s.trap_epochs, window: s.window });
      ctx.setData({ ...(ctx.data || {}), trap: r, meta: { compute_ms: ctx.lastJob ? ctx.lastJob.elapsed_s * 1000 : null } });
    } },
    { label: s => `Train ${s.net.toUpperCase()} walk-forward`, icon: 'play', async run(ctx) {
      const s = ctx.state;
      const r = await ctx.job({ kind: 'rnn_returns', ticker: s.ticker, model: s.net, epochs: s.rnn_epochs, start_year: s.start_year, end_year: END });
      ctx.setData({ ...(ctx.data || {}), rnn: r, meta: { compute_ms: ctx.lastJob ? ctx.lastJob.elapsed_s * 1000 : null } });
    } },
  ],
  kpis: d => {
    const T = d.trap, R = d.rnn, out = [];
    if (T) {
      const sc = T.scores;
      out.push({ label: 'Tutorial LSTM, R² on price', value: fmt(sc.leaky.r2, 3), tone: sc.leaky.r2 < sc.naive.r2 ? 'bad' : 'good',
        note: sc.leaky.r2 > 0.9 ? `${T.ticker} test years · looks near perfect` : `${T.ticker} · undertrained, add epochs` });
      out.push({ label: 'Tomorrow = today, R²', value: fmt(sc.naive.r2, 3), tone: 'good', note: `RMSE $${fmt(sc.naive.rmse, 2)} against $${fmt(sc.leaky.rmse, 2)} for the LSTM` });
      if (!R) out.push(
        { label: 'Leak fixed, R²', value: fmt(sc.clean.r2, 3), note: 'scaler fitted on training prices only' },
        { label: 'Daily moves predicted', value: fmt(T.change_corr, 2), unit: 'correlation', tone: Math.abs(T.change_corr) < 0.1 ? 'bad' : 'warn', note: `direction right on ${pct(T.direction_hit, 0)} of days` });
    }
    if (R) {
      const m = R.models[R.model], b = R.backtest[R.model], bh = R.backtest.buy_hold, up = R.always_up_hit_rate;
      out.push({ label: `${NAMES[R.model]} R² vs forecasting zero`, value: r2pct(m.r2_oos), unit: '%', tone: m.r2_oos > 0 && m.dm_p < 0.05 ? 'good' : m.r2_oos > 0 ? 'warn' : 'bad',
        note: `${R.ticker} ${R.years[0]}–${R.years[R.years.length - 1]} · DM p ${pval(m.dm_p)}` });
      out.push({ label: 'Direction right', value: pct(m.hit_rate, 1), tone: m.hit_rate > up && m.hit_p < 0.05 ? 'good' : m.hit_rate > up ? 'warn' : 'bad', note: `always up ${pct(up, 1)}` });
      if (!T) out.push(
        { label: 'Timing Sharpe ratio', value: fmt(b.sharpe, 2), tone: b.sharpe > bh.sharpe ? 'good' : 'bad', note: `buy and hold ${fmt(bh.sharpe, 2)}` },
        { label: 'Growth a year', value: pct(b.cagr, 1), note: `buy and hold ${pct(bh.cagr, 1)} · in the market ${pct(b.time_in_market, 0)} of days` });
    }
    return out;
  },
  illustration: {
    title: 'The LSTM trap: a scaler that sees the future', height: 220,
    caption: 'Min–max scaling squeezes prices into 0 to 1. Fitted on all prices, the scaler already knows how high the test years will go.',
    draw(svg, d, s) {
      const T = d?.trap, ratio = T ? T.test_max / T.train_max : 2.6, k = Math.log(ratio) / 0.2;
      const xs = [], ys = [];
      for (let i = 0; i <= 120; i++) { const t = i / 120; xs.push(t); ys.push(Math.exp(k * t) * (1 + 0.05 * Math.sin(29 * t) + 0.03 * Math.sin(71 * t))); }
      const trMax = Math.max(...ys.filter((_, i) => xs[i] <= 0.8)), teMax = Math.max(...ys);
      const box = [24, 40, 316, 146], { X, Y } = spark(svg, xs, ys, box, 'st1', [0, teMax * 1.04]);
      const xs8 = X(96), money = v => '$' + int(v);
      S('line', { x1: xs8, x2: xs8, y1: 30, y2: 190, class: 'ln-m', 'stroke-dasharray': '3 3' }, svg);
      text(svg, (box[0] + xs8) / 2, 206, 'training 80%'); text(svg, (xs8 + box[0] + box[2]) / 2, 206, 'test 20%');
      [[teMax, 'st2', T ? `future max ${money(T.test_max)}` : 'future max'], [trMax, 'st3', T ? `training max ${money(T.train_max)}` : 'training max']].forEach(([v, c, t]) => {
        S('line', { x1: box[0], x2: box[0] + box[2], y1: Y(v), y2: Y(v), class: c, 'stroke-dasharray': '4 4', 'stroke-width': 1.5 }, svg);
        text(svg, box[0] + 4, Y(v) - 6, t, 'n-sub', 'start');
      });
      const sc = T?.scores, r2 = x => sc ? `R² ${fmt(x.r2, 3)}` : null;
      const a = node(svg, { x: 384, y: 4, w: 208, h: 62, hot: true, title: 'Scaler on all prices', value: r2(sc?.leaky), sub: 'sees the future max' });
      const b = node(svg, { x: 384, y: 78, w: 208, h: 62, title: 'Scaler on training prices', value: r2(sc?.clean), sub: 'test prices run off the scale' });
      node(svg, { x: 384, y: 152, w: 208, h: 62, title: 'Tomorrow = today', value: r2(sc?.naive), sub: 'the baseline to beat' });
      edge(svg, [box[0] + box[2] + 4, Y(teMax)], [a.x - 6, a.cy], { width: 2, cls: 'st2' });
      edge(svg, [box[0] + box[2] + 4, Y(trMax)], [b.x - 6, b.cy], { width: 2, cls: 'st3' });
      text(svg, box[0], 22, `${T?.ticker || s.ticker} price, schematic`, 'n-sub', 'start');
    },
  },
  chartLayout: 'two',
  charts: [
    { id: 'trap', title: 'Price forecasts on the test years', height: 300, show: d => !!d.trap,
      sub: d => `${d.trap.ticker} from ${fdate(d.trap.split_date)}: the last 20% of days. ${d.trap.epochs} epochs, ${d.trap.window}-day window.`,
      legend: [['s1', 'line', 'Actual'], ['neutral', 'line', 'Tomorrow = today'], ['s2', 'line', 'Leaky LSTM'], ['s3', 'line', 'Leak fixed']],
      draw(host, d) {
        const T = d.trap, yrs = T.dates.map(yearOf), [lo, hi] = extent(T.actual, T.leaky, T.clean, T.naive);
        const P = frame(host, { h: 300, x: [yrs[0], yrs[yrs.length - 1]], y: [Math.max(0, lo * 0.95), hi * 1.05], yl: `${T.ticker} price ($)`, xf: v => fmt(v, 0) });
        P.line(zip(yrs, T.naive), 'sn thin'); P.line(zip(yrs, T.clean), 's3 thin'); P.line(zip(yrs, T.leaky), 's2 thin'); P.line(zip(yrs, T.actual), 's1');
        hover(host, P, px => { const i = nearest(yrs, P.xInv(px)); return { cross: P.X(yrs[i]), pts: [[P.X(yrs[i]), P.Y(T.actual[i]), 'f1'], [P.X(yrs[i]), P.Y(T.leaky[i]), 'f2']],
          html: tip(fdate(T.dates[i]), [['Actual', '$' + fmt(T.actual[i])], ['Tomorrow = today', '$' + fmt(T.naive[i])], ['Leaky LSTM', '$' + fmt(T.leaky[i])], ['Leak fixed', '$' + fmt(T.clean[i])]]) }; });
      } },
    { id: 'echo', title: 'What the leaky LSTM learned', height: 240, show: d => !!d.trap,
      sub: 'Correlation of its predicted daily change with the actual change k days earlier. A peak at lag 1 means it repeats yesterday\'s move.',
      legend: [['s2', 'box', 'Strongest lag'], ['neutral', 'box', 'Other lags']],
      draw(host, d) {
        const xc = d.trap.xcorr, lags = Object.keys(xc).map(Number).sort((a, b) => a - b), v = lags.map(l => xc[l]);
        const [lo, hi] = extent(v), best = lags[v.indexOf(hi)];
        const P = frame(host, { h: 240, x: [-0.5, lags[lags.length - 1] + 0.5], y: [Math.min(0, lo) - 0.05, Math.max(0.1, hi) * 1.15], xt: lags, xl: 'Lag k (days)', yl: 'Correlation', yf: v => fmt(v, 1) });
        P.hline(0);
        lags.forEach((l, i) => {
          const x0 = P.X(l - 0.32), x1 = P.X(l + 0.32), ya = P.Y(Math.max(0, v[i])), yb = P.Y(Math.min(0, v[i]));
          E('rect', { x: x0, y: ya, width: x1 - x0, height: Math.max(1, yb - ya), rx: 3, class: l === best ? 'f2' : 'fn', opacity: l === best ? 1 : 0.6 }, P.body);
          P.text(P.X(l), (v[i] >= 0 ? ya - 6 : yb + 14), fmt(v[i], 2), { 'text-anchor': 'middle', style: 'font-family:var(--f-mono)' });
        });
        hover(host, P, px => { const i = nearest(lags, P.xInv(px)); return { html: tip(i === 0 ? 'Same day' : `Lag ${lags[i]} day${lags[i] > 1 ? 's' : ''}`, [['Correlation', fmt(v[i], 3)]]) }; });
      } },
    { id: 'trap_table', kind: 'html', title: 'Scores on the price level', height: 160, show: d => !!d.trap,
      sub: 'A forecast that copies yesterday\'s price is the benchmark for any price model.',
      caption: d => `The leaky LSTM calls the direction of the next move right on ${pct(d.trap.direction_hit)} of days.`,
      draw(host, d) {
        const sc = d.trap.scores;
        host.innerHTML = `<div class="tbl-wrap"><table><thead><tr><th>Model on price level</th><th class="num">R²</th><th class="num">RMSE ($)</th><th class="num">MAPE</th></tr></thead><tbody>`
          + ['naive', 'leaky', 'clean'].map(k => `<tr${k === 'naive' ? ' class="best"' : ''}><td>${TRAP_NAMES[k]}</td><td class="num">${fmt(sc[k].r2, 3)}</td><td class="num">${fmt(sc[k].rmse, 2)}</td><td class="num">${fmt(sc[k].mape_pct, 1)}%</td></tr>`).join('')
          + '</tbody></table></div>';
      } },
    { id: 'rnn_table', kind: 'html', title: 'Walk-forward on returns', height: 160, show: d => !!d.rnn,
      sub: d => `${NAMES[d.rnn.model]} on ${d.rnn.ticker}, test years ${d.rnn.years[0]}–${d.rnn.years[d.rnn.years.length - 1]}, at most ${d.rnn.epochs} epochs per refit, ${d.rnn.window}-day sequences. Long when the forecast is positive, 5 bp per trade.`,
      caption: d => `Always predicting "up" is right ${pct(d.rnn.always_up_hit_rate)} of the time over these years.`,
      draw(host, d) { host.innerHTML = scoreTable(d.rnn, ['zero', d.rnn.model]); } },
    { id: 'rnn_eq', title: 'Growth of $1, network against buy and hold', height: 280, show: d => !!d.rnn,
      sub: 'Log scale, sampled weekly.',
      legend: d => [['neutral', 'line', 'Buy and hold'], [legendColor(d.rnn.model), 'line', NAMES[d.rnn.model]]],
      draw(host, d) { equity(host, d.rnn.equity, ['buy_hold', d.rnn.model]); } },
  ],
  insights: d => {
    const T = d.trap, R = d.rnn, out = [];
    if (T) {
      const sc = T.scores, xc = T.xcorr;
      out.push(`The tutorial LSTM scores R² <b>${fmt(sc.leaky.r2, 3)}</b> on the price level. Saying tomorrow equals today scores <b>${fmt(sc.naive.r2, 3)}</b>. Prices move little from one day to the next, so any forecast near yesterday's price looks good.`);
      out.push(`Its predicted daily change correlates <b>${fmt(xc[1], 2)}</b> with yesterday's actual change and ${fmt(xc[0], 2)} with today's. It mostly repeats the last move, a day late.`);
      if (!R) out.push(`The scaler fitted on all prices knows the test years reach ${'$' + int(T.test_max)}, ${fmt(T.test_max / T.train_max, 1)}× the training maximum of ${'$' + int(T.train_max)}. Fitted on training prices only, the R² is ${fmt(sc.clean.r2, 3)}` + (sc.clean.r2 < sc.leaky.r2 ? ': the network now has to forecast prices it has never seen.' : '.'));
    }
    if (R) {
      const m = R.models[R.model], b = R.backtest[R.model], bh = R.backtest.buy_hold;
      out.push(`On returns, the walk-forward ${NAMES[R.model]} explains <b>${r2pct(m.r2_oos)}%</b> of the next day's move (Diebold–Mariano p ${pval(m.dm_p)}) and is right on ${pct(m.hit_rate)} of days, against ${pct(R.always_up_hit_rate)} for always up.`);
      if (!T) {
        out.push(`Trading on its forecasts gives a Sharpe ratio of <b>${fmt(b.sharpe, 2)}</b>, against <b>${fmt(bh.sharpe, 2)}</b> for buy and hold.`);
        const used = Object.values(R.epochs_used || {});
        if (used.length) out.push(`Early stopping kept ${fmt(used.reduce((a, v) => a + v, 0) / used.length, 1)} of the allowed ${R.epochs} epochs per refit on average. The validation loss stops improving quickly when there is little signal to learn.`);
      }
    }
    return out;
  },
  maths: { eq: 'min–max     x̃ = (x − min) / (max − min)\nleaky       min, max over all prices, train and test\nclean       min, max over training prices only\nbaseline    P̂ₜ₊₁ = Pₜ\nR²          1 − Σ(P − P̂)² / Σ(P − P̄)²',
    notes: ['The trap: an LSTM with 50 hidden units, Adam (learning rate 0.001), batches of 32 and a chronological 80/20 split. The walk-forward network reads 20-day sequences of the 25 return features, forecasts the volatility-scaled return, and is refit every test year with early stopping on the two validation years.'] },
};

export default [volatility, returnsView, neural];
