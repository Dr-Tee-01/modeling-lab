// Tabs, server status and lazy initialisation of each view.
import { $, $$, call } from './api.js';
import { redrawAll } from './plot.js';

const views = ['overview', 'spread', 'shape', 'markets', 'crossover', 'api'];
const started = {};
const init = {
  spread: () => import('./spread.js').then(m => m.initSpread($('#v-spread'))),
  shape: () => import('./shape.js').then(m => m.initShape($('#v-shape'))),
  markets: () => import('./markets.js').then(m => m.initMarkets($('#markets-root'))),
  crossover: () => import('./crossover.js').then(m => m.initCrossover($('#v-crossover'))),
  api: () => initApi(),
};

function show(v, push) {
  if (!views.includes(v)) v = 'overview';
  views.forEach(k => { $('#v-' + k).hidden = k !== v; $('#t-' + k).setAttribute('aria-selected', String(k === v)); });
  if (push) { history.replaceState(null, '', '#' + v); window.scrollTo({ top: 0 }); }
  if (init[v] && !started[v]) {
    started[v] = true;
    // three.js loads with `defer`; wait for it before the Shape tab builds its viewers.
    const ready = v === 'shape' && !window.THREE ? new Promise(r => window.addEventListener('load', r, { once: true })) : Promise.resolve();
    ready.then(init[v]).catch(e => { started[v] = false; console.error(e); });
  }
  requestAnimationFrame(redrawAll);
}
$$('.tab').forEach(b => b.addEventListener('click', () => show(b.dataset.v, true)));
$$('[data-goto]').forEach(b => b.addEventListener('click', () => show(b.dataset.goto, true)));
window.addEventListener('hashchange', () => show(location.hash.slice(1)));
show(location.hash.slice(1) || 'overview');

async function ping() {
  const dot = $('#live-dot'), txt = $('#live-text');
  try {
    const t0 = performance.now(); await call('/api/health');
    dot.className = 'dot-live ok'; txt.textContent = `server live · ${Math.round(performance.now() - t0)} ms`;
  } catch (e) { dot.className = 'dot-live down'; txt.textContent = 'server unreachable'; }
}
ping(); setInterval(ping, 15000);

async function initApi() {
  const { routes } = await call('/api/routes');
  const list = $('#api-list'), out = $('#api-out');
  list.innerHTML = routes.map(r => `<div><span class="m">${r.methods.join('/')}</span><code>${r.path}</code>${r.summary ? ` <span class="cap">${r.summary}</span>` : ''}</div>`).join('');
  $$('div', list).forEach((row, i) => {
    const r = routes[i];
    if (!r.methods.includes('GET') || r.path.includes('<')) return;
    row.style.cursor = 'pointer';
    row.addEventListener('click', async () => {
      out.hidden = false; out.textContent = `GET ${r.path} …`;
      try { const j = await call(r.path); out.textContent = `GET ${r.path}  ·  ${j.meta ? j.meta.compute_ms + ' ms' : ''}\n` + JSON.stringify(j, (k, v) => Array.isArray(v) && v.length > 12 ? [...v.slice(0, 12), `… ${v.length - 12} more`] : v, 2).slice(0, 6000); }
      catch (e) { out.textContent = `GET ${r.path}\n${e.message}`; }
    });
  });
}
