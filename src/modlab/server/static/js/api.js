// Talking to the Flask API: query building, stale-request cancelling, compute badges.
export const $ = (s, root = document) => root.querySelector(s);
export const $$ = (s, root = document) => [...root.querySelectorAll(s)];

export class ApiError extends Error {
  constructor(message, code, status) { super(message); this.code = code; this.status = status; }
}

export async function call(path, params = {}, { method = 'GET', signal } = {}) {
  let url = path, init = { method, signal, headers: {} };
  if (method === 'GET') {
    const q = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== '') q.set(k, Array.isArray(v) ? v.join(',') : v);
    const s = q.toString(); if (s) url += '?' + s;
  } else {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(params);
  }
  let res;
  try { res = await fetch(url, init); }
  catch (e) { if (e.name === 'AbortError') throw e; throw new ApiError('The server is not reachable. Is Flask running?', 'network', 0); }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new ApiError(body.error?.message || `Request failed (${res.status})`, body.error?.code || 'error', res.status);
  return body;
}

/* A live panel: one request in flight at a time, the newest wins, results go to render().
   The badge in the panel header shows server compute time or the error. */
export function livePanel(panel, { request, render, delay = 120 }) {
  const badge = panel.querySelector('.compute');
  const errBox = document.createElement('div'); errBox.className = 'err-box'; errBox.hidden = true;
  panel.appendChild(errBox);
  let ctrl = null, timer = null, last = null;
  async function run() {
    ctrl?.abort(); ctrl = new AbortController();
    const mine = ctrl;
    if (badge) { badge.className = 'compute busy'; badge.textContent = 'computing…'; }
    const t0 = performance.now();
    try {
      const data = await request(mine.signal);
      if (mine.signal.aborted) return;
      last = data; errBox.hidden = true;
      render(data);
      if (badge) {
        const m = data.meta || {};
        badge.className = 'compute';
        badge.textContent = `server ${fmtMs(m.compute_ms)}${m.cached ? ' · reused cached result' : ''} · ${Math.round(performance.now() - t0)} ms round trip`;
        badge.title = m.cached ? 'The expensive part (an eigenbasis, a fitted model, a simulation) was computed by an earlier request with the same inputs and kept in memory.' : 'Computed from scratch for this request.';
      }
    } catch (e) {
      if (e.name === 'AbortError') return;
      if (badge) { badge.className = 'compute err'; badge.textContent = 'error'; }
      errBox.textContent = e.message; errBox.hidden = false;
    }
  }
  const api = {
    refresh() { clearTimeout(timer); timer = setTimeout(run, delay); },
    now() { clearTimeout(timer); return run(); },
    redraw() { if (last) render(last); },
    get data() { return last; },
  };
  return api;
}

export const fmtMs = ms => ms == null ? '—' : ms < 1 ? '<1 ms' : ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
export const fmt = (v, d = 2) => v == null || Number.isNaN(+v) ? '—' : Number(v).toFixed(d);
export const pct = (v, d = 1) => v == null ? '—' : (v * 100).toFixed(d) + '%';
export const int = v => v == null ? '—' : Math.round(v).toLocaleString('en-US');
const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
export const fdate = s => { if (!s) return '—'; const [y, m, d] = s.split('-'); return `${+d} ${MON[+m - 1]} ${y}`; };
export const mon = s => { const [y, m] = s.split('-'); return `${MON[+m - 1]} ${y}`; };

/* Bind a range/select/checkbox to state[key] with a formatted <output>. */
export function bind(root, state, key, onChange, format = v => v) {
  const el = root.querySelector(`[data-k="${key}"]`);
  if (!el) throw new Error('No control for ' + key);
  const out = el.closest('.ctl')?.querySelector('output');
  const read = () => el.type === 'checkbox' ? el.checked : el.tagName === 'SELECT' ? el.value : +el.value;
  const show = () => { if (out) out.textContent = format(state[key]); };
  if (el.type === 'checkbox') el.checked = !!state[key]; else el.value = state[key];
  show();
  el.addEventListener(el.type === 'range' ? 'input' : 'change', () => { state[key] = read(); show(); onChange(key); });
  return { set(v) { state[key] = v; if (el.type === 'checkbox') el.checked = !!v; else el.value = v; show(); } };
}

export function segmented(group, onPick) {
  const btns = $$('button', group);
  btns.forEach(b => b.addEventListener('click', () => { btns.forEach(x => x.setAttribute('aria-pressed', String(x === b))); onPick(b.dataset.v, b); }));
  return v => btns.forEach(x => x.setAttribute('aria-pressed', String(x.dataset.v === v)));
}

export const readouts = (el, items) => { el.innerHTML = items.map(([k, v]) => `<span>${k} <b>${v}</b></span>`).join(''); };
export const verdicts = (el, items) => { el.innerHTML = items.map(([q, a]) => `<div><span class="q">${q}</span><span class="a">${a}</span></div>`).join(''); };
