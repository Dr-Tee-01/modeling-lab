// Reference data from the server (GET /api/catalog): tickers, meshes, outlines, model and
// agent names, dataset facts. The browser keeps no data of its own; every module that
// needs a list or a label imports it from here. Loading waits (and retries) until the
// server answers, so nothing renders with stale or invented values.
import { call } from './api.js';

async function load() {
  for (let attempt = 0; ; attempt++) {
    try { return await call('/api/catalog'); }
    catch (e) {
      const view = typeof document !== 'undefined' && document.getElementById('view');
      if (view) {
        const box = document.createElement('div'), t = document.createElement('h2'), p = document.createElement('p');
        box.className = 'empty-state'; t.textContent = 'Waiting for the server…'; p.textContent = `${e.message} Retrying.`;
        box.append(t, p); view.replaceChildren(box);
      }
      if (typeof process !== 'undefined' && attempt > 2) throw e;          // Node harness: fail fast
      await new Promise(r => setTimeout(r, Math.min(5000, 1000 * (attempt + 1))));
    }
  }
}

export const catalog = await load();
