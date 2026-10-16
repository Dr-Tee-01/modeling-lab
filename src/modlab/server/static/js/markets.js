// Markets tab: walk-forward forecasting computed on request; neural nets train as background jobs.
import { $, $$, call, fdate, fmt, fmtMs, livePanel, pct, readouts, segmented } from './api.js';
import { E, extent, frame, hover, nearest, responsive, zip } from './plot.js';

const tip = (title, rows) => `<b>${title}</b>` + rows.map(([k, v]) => `<br>${k}: <span class="n">${v}</span>`).join('');
const yearOf = s => +s.slice(0, 4) + (+s.slice(5, 7) - 1) / 12 + (+s.slice(8, 10)) / 365;
const NAMES = { zero: 'Zero (random walk)', hist_mean: 'Historical mean', arima: 'ARIMA', ridge: 'Ridge', gbm: 'Gradient boosting', lstm: 'LSTM', gru: 'GRU', ensemble: 'Ensemble',
  hist21: '21-day history', ewma: 'EWMA', garch: 'GARCH(1,1)', har: 'HAR-RV', har_x: 'HAR-RV + VIX', buy_hold: 'Buy and hold' };
const SERIES = ['s1', 's2', 's3', 's4'];
const pval = p => p == null ? '—' : p < 0.001 ? p.toExponential(1) : fmt(p, 3);

const TEMPLATE = `
  <div class="study-h"><span class="tag">Pillar 3</span><h2>Markets: the direction is noise, the size is not</h2>
    <p>For each test year every model is refit on data strictly before that year, then forecasts each day one step ahead. All of it runs when you press a control.</p></div>
  <div class="controls">
    <div class="seg" role="group" aria-label="Ticker" data-seg="ticker">${['SPY', 'AAPL', 'MSFT', 'NVDA', 'JPM', 'XOM'].map((t, i) => `<button type="button" data-v="${t}" aria-pressed="${i === 0}">${t}</button>`).join('')}</div>
    <div class="ctl"><label for="mk-start">Test years from <output></output></label><input id="mk-start" type="range" data-k="start_year" min="2008" max="2025" step="1"></div>
  </div>

  <div class="panel" data-panel="vol">
    <header><h3>Volatility: forecasting tomorrow's variance</h3><span class="compute"></span>
      <p>Target: next-day log realised variance (Garman–Klass range plus the overnight gap). GARCH is refit by maximum likelihood each year on a 12-year window.</p></header>
    <div class="controls" data-models="vol">${['hist21', 'ewma', 'garch', 'har', 'har_x'].map(m => `<label class="chip"><input type="checkbox" value="${m}" checked> ${NAMES[m]}</label>`).join('')}</div>
    <div class="grid2">
      <div><h3>R² of log variance</h3><div class="plot-host" data-plot="vol-bars"></div></div>
      <div><h3>Value at Risk backtest (99%)</h3><div class="readout" data-out="var"></div><p class="cap">A good 99% VaR is breached on about 1% of days. Kupiec's test asks whether the breach rate is consistent with 1%.</p></div>
    </div>
    <div class="controls"><span class="cap">Forecast shown:</span><div class="seg" role="group" aria-label="Model to plot" data-seg="volshow"></div></div>
    <div class="legend"><span style="color:var(--neutral)"><i></i>Realised (log variance)</span><span style="color:var(--s2)"><i></i>Forecast</span></div>
    <div class="plot-host" data-plot="vol-series"></div>
  </div>

  <div class="panel" data-panel="ret">
    <header><h3>Returns: forecasting tomorrow's direction</h3><span class="compute"></span>
      <p>Target: next-day log return. Out-of-sample R² is measured against forecasting zero; the Diebold–Mariano test asks whether the gain is more than luck.</p></header>
    <div class="controls" data-models="ret">${['hist_mean', 'ridge', 'gbm', 'arima'].map(m => `<label class="chip"><input type="checkbox" value="${m}" ${m === 'arima' ? '' : 'checked'}> ${NAMES[m]}</label>`).join('')}</div>
    <div class="tbl-wrap"><table><thead><tr><th>Model</th><th>R² vs zero</th><th>Direction right</th><th>p vs always up</th><th>DM p vs zero</th><th>CAGR</th><th>Sharpe</th><th>Max drawdown</th></tr></thead><tbody data-out="ret-table"></tbody></table></div>
    <p class="cap" data-out="ret-note"></p>
    <div class="legend" data-out="eq-legend"></div>
    <div class="plot-host" data-plot="equity"></div>
  </div>

  <div class="panel" data-panel="jobs">
    <header><h3>Neural networks, trained on request</h3>
      <p>Training takes longer than a web request, so it runs in a background worker. The page starts a job and polls its progress.</p></header>
    <div class="grid2">
      <div class="card" style="gap:10px">
        <h3>The LSTM trap</h3>
        <p>The popular tutorial: an LSTM on price levels with the scaler fitted on all data. It looks near perfect. Is it better than "tomorrow equals today"?</p>
        <div class="controls"><div class="ctl"><label>Epochs <output></output></label><input type="range" data-job="trap-epochs" min="1" max="30" step="1" value="8"></div>
          <button class="btn primary" type="button" data-act="trap">Train</button></div>
        <div class="progress" hidden><i></i></div><p class="cap" data-out="trap-status"></p>
      </div>
      <div class="card" style="gap:10px">
        <h3>LSTM / GRU on returns, walk-forward</h3>
        <p>The honest version: a recurrent net on scale-free features, refit every test year with early stopping.</p>
        <div class="controls"><div class="seg" role="group" aria-label="Network" data-seg="rnn"><button type="button" data-v="lstm" aria-pressed="true">LSTM</button><button type="button" data-v="gru" aria-pressed="false">GRU</button></div>
          <div class="ctl"><label>Epochs <output></output></label><input type="range" data-job="rnn-epochs" min="1" max="30" step="1" value="6"></div>
          <button class="btn primary" type="button" data-act="rnn">Train</button></div>
        <div class="progress" hidden><i></i></div><p class="cap" data-out="rnn-status"></p>
      </div>
    </div>
    <div data-out="trap-result" hidden>
      <div class="tbl-wrap"><table><thead><tr><th>Model on price level</th><th>R²</th><th>RMSE ($)</th><th>MAPE</th></tr></thead><tbody data-out="trap-table"></tbody></table></div>
      <div class="legend"><span style="color:var(--s1)"><i></i>Actual</span><span style="color:var(--neutral)"><i></i>Tomorrow = today</span><span style="color:var(--s2)"><i></i>Leaky LSTM</span><span style="color:var(--s3)"><i></i>Leak fixed</span></div>
      <div class="plot-host" data-plot="trap"></div>
    </div>
    <div class="readout" data-out="rnn-result"></div>
  </div>`;

export function initMarkets(root) {
  root.innerHTML = TEMPLATE;
  const s = { ticker: 'SPY', start_year: 2015 };
  const end = 2026;
  const startCtl = $('[data-k="start_year"]', root);
  startCtl.value = s.start_year; startCtl.closest('.ctl').querySelector('output').textContent = s.start_year;
  const picked = kind => $$(`[data-models="${kind}"] input:checked`, root).map(i => i.value);

  /* ---------------- volatility */
  const pV = $('[data-panel="vol"]', root);
  let volShow = 'har_x';
  const vol = livePanel(pV, { delay: 200, request: sig => {
    const models = picked('vol'); if (!models.length) throw new Error('Pick at least one model.');
    const varModel = models.includes('har_x') ? 'har_x' : models[models.length - 1];
    return call('/api/markets/volatility', { ticker: s.ticker, models, start_year: s.start_year, end_year: end, var_alpha: 0.01, var_model: varModel }, { signal: sig });
  }, render: renderVol });
  const hBars = $('[data-plot="vol-bars"]', pV), hSer = $('[data-plot="vol-series"]', pV);
  const showSeg = $('[data-seg="volshow"]', pV);
  const drawVol = () => {
    const d = vol.data; if (!d) return;
    const rows = Object.entries(d.models).map(([k, v]) => ({ k, v: v.r2_log })).sort((a, b) => a.v - b.v);
    hbars(hBars, rows, Math.max(0.7, ...rows.map(r => r.v)), v => Math.round(v * 100) + '%', r => tip(NAMES[r.k], [['R² log variance', pct(d.models[r.k].r2_log)], ['QLIKE', fmt(d.models[r.k].qlike, 3)]]));
    const S = d.series, yrs = S.dates.map(yearOf), f = S.forecasts[volShow] || Object.values(S.forecasts)[0];
    const step = Math.max(1, Math.floor(yrs.length / 1500)), idx = yrs.map((_, i) => i).filter(i => i % step === 0);
    const [lo, hi] = extent(idx.map(i => S.realized[i]), idx.map(i => f[i]));
    const P = frame(hSer, { h: 280, x: [yrs[0], yrs[yrs.length - 1]], y: [lo, hi], yl: 'Log daily variance', xf: v => Math.round(v) });
    P.line(idx.map(i => [yrs[i], S.realized[i]]), 'sn thin').setAttribute('opacity', '.6');
    P.line(idx.map(i => [yrs[i], f[i]]), 's2');
    hover(hSer, P, px => { const i = nearest(yrs, P.xInv(px)); const vol = v => pct(Math.sqrt(Math.exp(v) * 252), 0); return { cross: P.X(yrs[i]), pts: [[P.X(yrs[i]), P.Y(f[i]), 'f2']], html: tip(fdate(S.dates[i]), [['Realised (annualised)', vol(S.realized[i])], ['Forecast (annualised)', vol(f[i])]]) }; });
  };
  responsive(hBars, drawVol); responsive(hSer, drawVol);
  function renderVol(d) {
    const names = Object.keys(d.series.forecasts);
    if (!names.includes(volShow)) volShow = names.includes('har_x') ? 'har_x' : names[0];
    showSeg.innerHTML = names.map(n => `<button type="button" data-v="${n}" aria-pressed="${n === volShow}">${NAMES[n] || n}</button>`).join('');
    segmented(showSeg, v => { volShow = v; drawVol(); });
    const V = d.var || {}, row = (label, x) => x ? [label, `${pct(x.rate, 2)} breached (${x.exceed} of ${x.n}), Kupiec p ${pval(x.kupiec_p)}`] : null;
    readouts($('[data-out="var"]', pV), [row(`${NAMES[V.model] || V.model}, normal tails`, V.normal), row('Same, filtered historical simulation', V.fhs)].filter(Boolean));
    drawVol();
  }

  /* ---------------- returns */
  const pR = $('[data-panel="ret"]', root);
  const ret = livePanel(pR, { delay: 200, request: sig => {
    const models = picked('ret'); if (!models.length) throw new Error('Pick at least one model.');
    return call('/api/markets/returns', { ticker: s.ticker, models, start_year: s.start_year, end_year: end }, { signal: sig });
  }, render: renderRet });
  const hEq = $('[data-plot="equity"]', pR);
  let eqNames = [];
  const drawEq = responsive(hEq, () => {
    const d = ret.data; if (!d || !d.equity) return;
    const E2 = d.equity, yrs = E2.dates.map(yearOf);
    eqNames = ['buy_hold', ...Object.keys(E2.curves).filter(k => k !== 'buy_hold' && k !== 'zero')];
    const [lo, hi] = extent(...eqNames.map(n => E2.curves[n]));
    const P = frame(hEq, { h: 280, x: [yrs[0], yrs[yrs.length - 1]], y: [Math.min(lo, 1) * 0.95, hi * 1.05], ylog: true, yt: [0.5, 1, 2, 3, 5, 10, 20, 50].filter(v => v >= lo * 0.9 && v <= hi * 1.1), yf: v => v + '×', yl: 'Growth of $1 (log), long or cash, 5 bp per trade', xf: v => Math.round(v) });
    eqNames.forEach((n, i) => P.line(zip(yrs, E2.curves[n]), n === 'buy_hold' ? 'sn' : SERIES[(i - 1) % 4]));
    hover(hEq, P, px => { const i = nearest(yrs, P.xInv(px)); return { cross: P.X(yrs[i]), html: tip(fdate(E2.dates[i]), eqNames.map(n => [NAMES[n] || n, fmt(E2.curves[n][i], 2) + '×'])) }; });
    $('[data-out="eq-legend"]', pR).innerHTML = eqNames.map((n, i) => `<span style="color:var(--${n === 'buy_hold' ? 'neutral' : SERIES[(i - 1) % 4].replace('s', 's')})"><i></i>${NAMES[n] || n}</span>`).join('');
  });
  function renderRet(d) {
    const bt = d.backtest || {};
    const rows = Object.entries(d.models).map(([k, m]) => {
      const b = bt[k] || {};
      return `<tr><td>${NAMES[k] || k}</td><td class="num">${fmt(m.r2_oos * 100, 2)}%</td><td class="num">${m.hit_rate == null ? '—' : pct(m.hit_rate)}</td><td class="num">${pval(m.hit_p)}</td><td class="num">${pval(m.dm_p)}</td><td class="num">${b.cagr == null ? '—' : pct(b.cagr)}</td><td class="num">${fmt(b.sharpe)}</td><td class="num">${b.max_dd == null ? '—' : pct(b.max_dd, 0)}</td></tr>`;
    });
    const bh = bt.buy_hold;
    if (bh) rows.push(`<tr><td>Buy and hold</td><td class="num">—</td><td class="num">—</td><td class="num">—</td><td class="num">—</td><td class="num">${pct(bh.cagr)}</td><td class="num">${fmt(bh.sharpe)}</td><td class="num">${pct(bh.max_dd, 0)}</td></tr>`);
    $('[data-out="ret-table"]', pR).innerHTML = rows.join('');
    $('[data-out="ret-note"]', pR).textContent = `Always predicting "up" is right ${pct(d.always_up_hit_rate)} of the time over these years.`;
    drawEq();
  }

  /* ---------------- jobs */
  const pJ = $('[data-panel="jobs"]', root);
  $$('[data-job]', pJ).forEach(r => { const o = r.closest('.ctl').querySelector('output'); o.textContent = r.value; r.addEventListener('input', () => o.textContent = r.value); });
  let rnnKind = 'lstm';
  segmented($('[data-seg="rnn"]', pJ), v => { rnnKind = v; });
  async function runJob(body, card, onDone) {
    const btn = $('.btn', card), bar = $('.progress', card), out = $('.cap[data-out]', card);
    btn.disabled = true; bar.hidden = false; bar.firstElementChild.style.width = '0%';
    try {
      let { job } = await call('/api/jobs', body, { method: 'POST' });
      while (job.status === 'queued' || job.status === 'running') {
        bar.firstElementChild.style.width = Math.round(job.progress * 100) + '%';
        out.textContent = `${job.message} · ${fmt(job.elapsed_s, 1)} s`;
        await new Promise(r => setTimeout(r, 700));
        ({ job } = await call(`/api/jobs/${job.id}`));
      }
      bar.firstElementChild.style.width = '100%';
      if (job.status === 'failed') { out.textContent = job.error; return; }
      out.textContent = `Trained on the server in ${fmtMs(job.elapsed_s * 1000)}.`;
      onDone(job.result);
    } catch (e) { out.textContent = e.message; } finally { btn.disabled = false; }
  }
  const [cTrap, cRnn] = $$('.card', pJ);
  const hTrap = $('[data-plot="trap"]', pJ);
  let trap = null;
  const drawTrap = responsive(hTrap, () => {
    if (!trap) return;
    const yrs = trap.dates.map(yearOf), [lo, hi] = extent(trap.actual, trap.leaky, trap.clean);
    const P = frame(hTrap, { h: 300, x: [yrs[0], yrs[yrs.length - 1]], y: [lo * 0.95, hi * 1.05], yl: `${trap.ticker || s.ticker} price ($)`, xf: v => fmt(v, 0) });
    P.line(zip(yrs, trap.naive), 'sn thin'); P.line(zip(yrs, trap.clean), 's3 thin'); P.line(zip(yrs, trap.leaky), 's2 thin'); P.line(zip(yrs, trap.actual), 's1');
    hover(hTrap, P, px => { const i = nearest(yrs, P.xInv(px)); return { cross: P.X(yrs[i]), html: tip(fdate(trap.dates[i]), [['Actual', '$' + fmt(trap.actual[i])], ['Tomorrow = today', '$' + fmt(trap.naive[i])], ['Leaky LSTM', '$' + fmt(trap.leaky[i])], ['Leak fixed', '$' + fmt(trap.clean[i])]]) }; });
  });
  $('[data-act="trap"]', pJ).addEventListener('click', () => runJob({ kind: 'lstm_trap', ticker: s.ticker, epochs: +$('[data-job="trap-epochs"]', pJ).value, window: 60 }, cTrap, r => {
    trap = r; $('[data-out="trap-result"]', pJ).hidden = false;
    const sc = r.scores, name = { naive: 'Tomorrow = today', leaky: 'Tutorial LSTM (scaler fitted on all data)', clean: 'Same LSTM, leak fixed' };
    $('[data-out="trap-table"]', pJ).innerHTML = ['naive', 'leaky', 'clean'].map(k => `<tr><td>${name[k]}</td><td class="num">${fmt(sc[k].r2, 3)}</td><td class="num">${fmt(sc[k].rmse, 1)}</td><td class="num">${fmt(sc[k].mape_pct, 1)}%</td></tr>`).join('')
      + `<tr><td colspan="4">The leaky LSTM calls the direction right on ${pct(r.direction_hit)} of days.</td></tr>`;
    drawTrap();
  }));
  $('[data-act="rnn"]', pJ).addEventListener('click', () => runJob({ kind: 'rnn_returns', ticker: s.ticker, model: rnnKind, epochs: +$('[data-job="rnn-epochs"]', pJ).value, start_year: s.start_year, end_year: end }, cRnn, r => {
    const m = r.scores || r.models?.[rnnKind] || r;
    readouts($('[data-out="rnn-result"]', pJ), [[`${NAMES[rnnKind]} R² vs zero`, fmt(m.r2_oos * 100, 2) + '%'], ['Direction right', pct(m.hit_rate)], ['DM p vs zero', pval(m.dm_p)]]);
  }));

  /* ---------------- shared controls */
  const refresh = () => { vol.refresh(); ret.refresh(); };
  segmented($('[data-seg="ticker"]', root), v => { s.ticker = v; refresh(); });
  startCtl.addEventListener('input', () => { s.start_year = +startCtl.value; startCtl.closest('.ctl').querySelector('output').textContent = s.start_year; refresh(); });
  $$('[data-models] input', root).forEach(i => i.addEventListener('change', () => (i.closest('[data-models]').dataset.models === 'vol' ? vol : ret).refresh()));
  vol.now(); ret.now();
}

function hbars(host, rows, max, f, tipf) {
  const rowH = 30, h = rows.length * rowH + 40, w = Math.max(260, host.clientWidth), lw = Math.min(140, w * 0.38);
  const P = frame(host, { h, x: [0, max], y: [0, rows.length], yt: [], m: { l: lw, r: 50, t: 6, b: 30 }, xf: f, xn: 4 });
  rows.forEach((r, i) => {
    const y = P.Y(i + 0.5), x0 = P.X(0), x1 = P.X(Math.max(r.v, 0));
    E('rect', { x: x0, y: y - 9, width: Math.max(2, x1 - x0), height: 18, rx: 4, class: 'f1' }, P.body);
    P.text(lw - 10, y + 4, NAMES[r.k] || r.k, { 'text-anchor': 'end' });
    P.text(x1 + 6, y + 4, f(r.v), { style: 'font-family:var(--f-mono)' });
  });
  hover(host, P, (px, py) => { const i = Math.floor((P.h - P.M.b - py) / rowH); return i >= 0 && i < rows.length ? { html: tipf(rows[i]) } : null; });
}
