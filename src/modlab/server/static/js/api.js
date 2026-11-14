// Talking to the Flask API, plus number and date formatting shared by every view.

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

export const fmtMs = ms => ms == null ? '—' : ms < 1 ? '<1 ms' : ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`;
export const fmt = (v, d = 2) => v == null || Number.isNaN(+v) ? '—' : Number(v).toFixed(d);
export const pct = (v, d = 1) => v == null ? '—' : (v * 100).toFixed(d) + '%';
export const int = v => v == null ? '—' : Math.round(v).toLocaleString('en-US');
const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
export const fdate = s => { if (!s) return '—'; const [y, m, d] = s.split('-'); return `${+d} ${MON[+m - 1]} ${y}`; };
export const mon = s => { const [y, m] = s.split('-'); return `${MON[+m - 1]} ${y}`; };
