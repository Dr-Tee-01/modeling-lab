// Smoke-test view specs against a running server, without a browser.
//
//   node tools/check_views.mjs spread [shape markets crossover agents]   (server at $BASE, default http://localhost:8000)
//
// For every spec: checks the required fields, builds the default state, runs request()
// against the server, then runs kpis() and insights() on the response, and each preset's
// request. Charts and illustrations need a DOM and are checked in the browser instead.
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const BASE = process.env.BASE || 'http://localhost:8000';
const realFetch = globalThis.fetch;
globalThis.fetch = (url, init) => realFetch(url.startsWith('/') ? BASE + url : url, init);

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../src/modlab/server/static/js/views');
const groups = process.argv.slice(2).length ? process.argv.slice(2) : ['spread'];
let failed = 0;

for (const g of groups) {
  const specs = (await import(pathToFileURL(path.join(root, `${g}.js`)).href)).default;
  if (!Array.isArray(specs) || !specs.length) { console.log(`✗ ${g}: no specs exported`); failed++; continue; }
  for (const spec of specs) {
    const label = `${g}/${spec.id}`;
    try {
      for (const k of ['id', 'group', 'title', 'nav']) if (!spec[k]) throw new Error(`missing ${k}`);
      if (spec.group !== g) throw new Error(`group is "${spec.group}", expected "${g}"`);
      for (const p of spec.params || []) {
        if (!p.k || p.value === undefined) throw new Error(`param without k/value: ${JSON.stringify(p)}`);
        if ((p.type || 'range') === 'range' && !(p.min <= p.value && p.value <= p.max)) throw new Error(`param ${p.k}: default outside [min, max]`);
      }
      const state = { ...Object.fromEntries((spec.params || []).map(p => [p.k, p.value])), ...(spec.hidden || {}) };
      const ctx = { state, extra: {}, set() {}, refresh() {}, now() {}, toast() {}, setData() {} };
      const t0 = Date.now();
      if (spec.request) {
        const d = await spec.request(state, new AbortController().signal, ctx);
        const kpis = spec.kpis ? spec.kpis(d, state).filter(Boolean) : [];
        const ins = spec.insights ? spec.insights(d, state).filter(Boolean) : [];
        for (const pr of spec.presets || []) {
          const st = { ...state, ...pr.values };
          try {
            const dp = await spec.request(st, new AbortController().signal, ctx);
            spec.kpis?.(dp, st); spec.insights?.(dp, st);
          } catch (e) {
            if (e.code !== 'no_model') throw new Error(`preset "${pr.label}": ${e.message}`);
            console.log(`  ⚠ ${label} preset "${pr.label}" needs a trained model first (expected on a fresh server)`);
          }
        }
        console.log(`✓ ${label}  ${Date.now() - t0} ms  · ${kpis.length} tiles · ${ins.length} insights · ${(spec.presets || []).length} presets`);
      } else {
        console.log(`✓ ${label}  (no request; data comes from actions/jobs)`);
      }
    } catch (e) {
      failed++; console.log(`✗ ${label}: ${e.message}`);
    }
  }
}
process.exit(failed ? 1 : 0);
