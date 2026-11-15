// The workbench: renders any view spec into the same layout.
//
//   header      breadcrumb, title, question, compute badge, copy link, reset
//   top row     live illustration  |  result tiles (KPIs)
//   body        controls rail (presets, grouped fields, actions)  |  chart cards
//   foot        what this shows (insights)  |  the maths
//
// A spec is a plain object (see views/spread.js for the reference ones):
//   { id, group, title, nav, tag, question,
//     params: [{k, label, min, max, step, value, unit, scale, help, group, type: 'range'|'switch'|'seg', options, show(s)}],
//     presets: [{label, values}],  request(s, signal, ctx) -> Promise<data>,  debounce,
//     kpis(d, s) -> [{label, value, unit, note, tone}],  kpiCount,
//     illustration: {title, height, caption, draw(svg, d, s)},
//     charts: [{id, title, sub, legend, kind: 'svg'|'3d'|'html', height, draw(host|viewer, d, s, ctx), show(d, s)}], chartLayout: 'one'|'two',
//     actions: [{label (string or fn(state)), icon, primary, run(ctx, ui)}]   (ctx.job(...) sets ctx.lastJob.elapsed_s),  insights(d, s) -> [html],  maths: {eq, notes},
//     reset(ctx) (optional: clear view-local state on Reset), destroy(ctx) }
//   chart.sub / caption / legend may be functions of (d, s, ctx).
import { call, fmtMs } from '../api.js';
import { decorate, responsive } from '../plot.js';
import { viewer } from '../viewer3d.js';
import { h, helpButton, icon, seg, toast } from './dom.js';
import { canvas } from './illustrate.js';

const decimals = step => { const s = String(step); return s.includes('.') ? s.split('.')[1].length : 0; };

function coerce(p, raw) {
  if (p.type === 'switch') return raw === true || raw === 'true' || raw === '1';
  if (p.type === 'seg') return p.options.some(o => String(o[0]) === String(raw)) ? p.options.find(o => String(o[0]) === String(raw))[0] : p.value;
  const v = Number(raw);
  if (!Number.isFinite(v)) return p.value;
  return Math.min(p.max, Math.max(p.min, v));
}

export function mountWorkbench(root, spec, initial = {}, { onState, crumbs, accent } = {}) {
  const params = spec.params || [];
  const defaults = Object.fromEntries(params.map(p => [p.k, p.value]));
  const state = { ...defaults };
  for (const p of params) if (initial[p.k] != null) state[p.k] = coerce(p, initial[p.k]);
  Object.assign(state, spec.hidden ? { ...spec.hidden } : {});

  let data = null, ctrl = null, timer = null, alive = true;
  const lastKpi = new Map();
  const disposers = [];
  const ctx = { state, get data() { return data; }, extra: {}, call, toast, set, refresh, now, job, redraw: renderAll, root,
    setData(d) { data = d; errBox.hidden = true; badgeFor(d); renderAll(); } };

  /* ---------------- header */
  const badge = h('span.compute', { title: '' }, 'connecting…');
  const linkBtn = h('button.btn.sm', { type: 'button', onclick: copyLink, title: 'Copy a link to this exact setup' }, icon('link', ''), 'Copy link');
  const resetBtn = h('button.btn.sm.ghost', { type: 'button', onclick: reset, title: 'Back to the default parameters' }, icon('reset', ''), 'Reset');
  const explainBtn = h('a.btn.sm.explain-btn', { href: `#/${spec.group}/${spec.id}/explain`, title: 'A step-by-step illustrated explanation of this model' }, icon('book', ''), 'See explanation');
  const head = h('header.wb-head',
    h('div.crumbs', crumbs || [], spec.tag ? h('span.lab', spec.tag) : null),
    h('div', h('h1', spec.title), spec.question ? h('p.question', spec.question) : null),
    h('div.wb-head-actions', explainBtn, badge, params.length ? linkBtn : null, params.length ? resetBtn : null));

  /* ---------------- top: illustration + KPIs */
  const illHost = h('div', { 'data-k': spec.id });
  const ill = spec.illustration ? h('section.card.illus', { 'aria-label': 'Illustration' },
    h('div.illus-head', h('h3', spec.illustration.title || 'How it works')), illHost,
    spec.illustration.caption ? h('p.cap', spec.illustration.caption) : null) : null;
  const kpiBox = h('div.kpis', { 'aria-live': 'polite' });
  for (let i = 0; i < (spec.kpiCount || 4); i++) kpiBox.append(h('div.card.kpi.skeleton', h('span.k-label', ' '), h('span.k-value', '0000'), h('span.k-note', ' ')));
  const top = h('div.wb-top', ill, kpiBox);
  if (!ill) top.style.gridTemplateColumns = '1fr';

  /* ---------------- controls */
  const fields = {};
  const labelPainters = [];
  const controls = h('aside.card.controls', { 'aria-label': 'Controls' }, h('div.controls-head', h('h3', 'Controls'), h('span.c-note', 'Changes recompute on the server')));
  if (spec.presets?.length) {
    const box = h('div.presets');
    spec.presets.forEach(pr => {
      const b = h('button.preset', { type: 'button', 'aria-pressed': 'false', title: pr.note || '', onclick: () => { set(pr.values, { immediate: true }); } }, pr.label);
      b._values = pr.values; box.append(b);
    });
    controls.append(h('div.c-section', h('span.eyebrow', 'Scenarios'), box));
  }
  let section = null, lastGroup = Symbol();
  for (const p of params) {
    if (p.group !== lastGroup) { section = h('div.c-section', p.group ? h('span.eyebrow', p.group) : null); controls.append(section); lastGroup = p.group; }
    const f = field(p); fields[p.k] = f; section.append(f.el);
  }
  const actionBox = h('div.c-section');
  (spec.actions || []).forEach(a => {
    const lbl = h('span');
    const btn = h('button.btn' + (a.primary ? '.primary' : '') + '.block', { type: 'button' }, a.icon ? icon(a.icon, '') : null, lbl);
    const paintLabel = () => { lbl.textContent = typeof a.label === 'function' ? a.label(state) : a.label; };
    paintLabel(); labelPainters.push(paintLabel);
    const note = h('p.c-note', { hidden: true });
    const prog = h('div.job-progress', { hidden: true }, h('div.bar', h('i')), note);
    btn.addEventListener('click', async () => {
      btn.disabled = true; ctx._prog = prog; note.hidden = false;
      try { await a.run(ctx, { note: m => { note.hidden = false; note.textContent = m; } }); }
      catch (e) { note.hidden = false; note.textContent = e.message; }
      finally { btn.disabled = false; ctx._prog = null; }
    });
    actionBox.append(btn, prog);
  });
  if (spec.actions?.length) controls.append(actionBox);
  if (!params.length && !spec.actions?.length && !spec.presets?.length) controls.hidden = true;

  /* ---------------- charts */
  const errBox = h('div.err-box', { hidden: true, role: 'alert' });
  const chartsEl = h('div.charts' + (spec.chartLayout === 'two' ? '.two' : ''));
  const charts = (spec.charts || []).map(c => {
    const legend = h('div.legend');
    const sub = h('p.sub');
    const tools = h('div.chart-tools');
    const host = c.kind === '3d' ? h('div.viewer3d') : h('div.plot-host', h('div.skeleton-plot', { style: { height: (c.height || 280) + 'px' } }));
    const cap = h('p.cap', { hidden: !c.caption });
    const card = h('section.card.chart', { 'aria-label': c.title },
      h('div.chart-head', h('div', h('h3', c.title), sub), tools), legend, host, cap);
    chartsEl.append(card);
    const rec = { c, card, host, legend, sub, tools, cap, v3: null };
    if (c.tools) c.tools(tools, ctx);
    if (c.kind !== '3d' && c.kind !== 'html') {
      const run = responsive(host, () => { if (data && visible(rec)) safeDraw(rec); });
      disposers.push(() => run.dispose?.());
      rec.run = run;
    }
    return rec;
  });
  const body = h('div.wb-body', controls, h('div', { style: { display: 'grid', gap: '16px', minWidth: 0 } }, errBox, chartsEl));
  if (controls.hidden) body.style.gridTemplateColumns = '1fr';

  /* ---------------- foot */
  const insightsList = h('ul');
  const foot = h('div.wb-foot',
    spec.insights ? h('section.card.insights', h('h3', 'What this shows'), insightsList) : null,
    spec.maths ? h('section.card.maths', h('h3', 'The maths'),
      spec.maths.eq ? h('div.eq', spec.maths.eq) : null, ...(spec.maths.notes || []).map(n => h('p', { html: n }))) : null);
  if (!spec.insights || !spec.maths) foot.style.gridTemplateColumns = '1fr';

  const page = h('div.page.wb', head, top, body, (spec.insights || spec.maths) ? foot : null);
  if (accent) page.style.setProperty('--sec', `var(--${accent})`);
  root.replaceChildren(page);

  /* ---------------- fields */
  function field(p) {
    const label = h('label', { for: `f-${spec.id}-${p.k}` }, p.label);
    const helpB = p.help ? helpButton(p.help) : null;
    if (p.type === 'switch') {
      const input = h('input', { type: 'checkbox', id: `f-${spec.id}-${p.k}`, role: 'switch' });
      input.checked = !!state[p.k];
      input.addEventListener('change', () => set({ [p.k]: input.checked }));
      const el = h('div.field', h('div.field-top', label, helpB, h('span.switch', input, h('span'))));
      return { el, sync: () => { input.checked = !!state[p.k]; } };
    }
    if (p.type === 'seg') {
      const s = seg(p.options, state[p.k], v => set({ [p.k]: v }), { block: true, label: p.label });
      const el = h('div.field', p.label ? h('div.field-top', h('span', p.label), helpB) : null, s);
      return { el, sync: () => s.set(state[p.k]) };
    }
    const scale = p.scale || 1, dec = decimals(p.step) + (scale === 100 ? -2 : 0);
    const range = h('input', { type: 'range', id: `f-${spec.id}-${p.k}`, min: p.min, max: p.max, step: p.step, 'aria-label': p.label });
    const num = h('input', { type: 'number', min: p.min * scale, max: p.max * scale, step: p.step * scale, 'aria-label': `${p.label} value` });
    let shown = state[p.k], pulse = 0;
    const show = () => {
      if (state[p.k] !== shown) { shown = state[p.k]; el.classList.add('tick'); clearTimeout(pulse); pulse = setTimeout(() => el.classList.remove('tick'), 280); }
      range.value = state[p.k];
      num.value = (state[p.k] * scale).toFixed(Math.max(0, dec));
      range.style.setProperty('--fill', ((state[p.k] - p.min) / (p.max - p.min) * 100) + '%');
    };
    range.addEventListener('input', () => { state[p.k] = +range.value; show(); changed(); });
    num.addEventListener('change', () => { const v = coerce(p, +num.value / scale); set({ [p.k]: v }); });
    const el = h('div.field', h('div.field-top', label, helpB, h('span.field-val', num, p.unit ? h('span.unit', p.unit) : null)), range);
    show();
    return { el, sync: show, range };
  }

  function syncVisibility() {
    for (const p of params) if (p.show) fields[p.k].el.hidden = !p.show(state);
    controls.querySelectorAll('.c-section').forEach(sec => { const f = [...sec.querySelectorAll('.field')]; if (f.length) sec.hidden = f.every(x => x.hidden); });
    controls.querySelectorAll('.preset').forEach(b => b.setAttribute('aria-pressed', String(Object.entries(b._values).every(([k, v]) => String(state[k]) === String(v)))));
  }

  /* ---------------- state changes */
  function set(patch, { immediate = false, run = true } = {}) {
    Object.assign(state, patch);
    for (const k of Object.keys(patch)) fields[k]?.sync();
    changed({ immediate, run });
  }
  function changed({ immediate = false, run = true } = {}) {
    syncVisibility();
    labelPainters.forEach(f => f());
    drawIllustration();
    onState?.(publicState());
    if (run) immediate ? now() : refresh();
  }
  function publicState() { return Object.fromEntries(params.map(p => [p.k, state[p.k]])); }
  function refresh() { clearTimeout(timer); timer = setTimeout(now, spec.debounce ?? 120); }

  function badgeFor(d, t0) {
    const m = d?.meta || {};
    badge.className = 'compute';
    badge.textContent = m.compute_ms == null ? 'ready' : `server ${fmtMs(m.compute_ms)}${m.cached ? ' · cached' : ''}`;
    badge.title = m.compute_ms == null ? '' : `${m.cached ? 'Reused a result computed earlier with the same inputs. ' : 'Computed from scratch for this request. '}${t0 ? 'Round trip ' + Math.round(performance.now() - t0) + ' ms.' : ''}`;
  }
  function showEmpty() {
    badge.className = 'compute'; badge.textContent = 'ready';
    kpiBox.hidden = true;
    charts.forEach(r => { r.host.querySelector('.skeleton-plot')?.remove(); if (r.c.kind !== '3d') { r.host.classList.add('empty'); r.host.textContent = spec.emptyText || 'Run the model to see results here.'; } });
  }
  async function now() {
    if (!spec.request) { if (data) renderAll(); else showEmpty(); return; }
    clearTimeout(timer);
    ctrl?.abort(); ctrl = new AbortController();
    const mine = ctrl, t0 = performance.now();
    badge.className = 'compute busy'; badge.textContent = 'computing…';
    charts.forEach(r => r.card.classList.add('is-loading'));
    try {
      const d = await spec.request(state, mine.signal, ctx);
      if (mine.signal.aborted || !alive) return;
      data = d; errBox.hidden = true;
      badgeFor(d, t0);
      renderAll();
    } catch (e) {
      if (e.name === 'AbortError' || !alive) return;
      badge.className = 'compute err'; badge.textContent = 'error';
      errBox.textContent = e.message; errBox.hidden = false;
    } finally {
      if (ctrl === mine) charts.forEach(r => r.card.classList.remove('is-loading'));
    }
  }

  /* ---------------- rendering */
  function visible(rec) { return !rec.c.show || rec.c.show(data, state); }
  function safeDraw(rec) {
    try {
      rec.host.querySelector('.skeleton-plot')?.remove();
      if (rec.c.kind === '3d') { rec.v3 ||= viewer(rec.host); if (rec.v3) rec.c.draw(rec.v3, data, state, ctx); }
      else {
        rec.c.draw(rec.host, data, state, ctx);
        if (rec.c.kind !== 'html') { decorate(rec.host, { animate: !rec.drawn }); rec.drawn = true; }
      }
    } catch (e) { console.error(`[${spec.id}] chart ${rec.c.id}`, e); }
  }
  function renderAll() {
    if (!data) return;
    charts.forEach(r => { if (r.host.classList.contains('empty')) { r.host.classList.remove('empty'); r.host.textContent = ''; } });
    if (spec.kpis) {
      const items = spec.kpis(data, state).filter(Boolean);
      kpiBox.replaceChildren(...items.map(k => {
        const val = h('span.k-value', k.value ?? '—', k.unit ? h('small', k.unit) : null);
        const card = h('div.card.kpi' + (k.tone ? '.' + k.tone : ''),
          h('span.k-label', k.label, k.help ? helpButton(k.help) : null), val,
          k.note ? h('span.k-note', { html: k.note }) : null);
        const before = lastKpi.get(k.label), now = String(k.value ?? '—');
        if (before != null && before !== now) { card.classList.add('flash'); countUp(val.firstChild, before, now); }
        lastKpi.set(k.label, now);
        return card;
      }));
      kpiBox.hidden = !items.length;
    } else kpiBox.hidden = true;
    for (const rec of charts) {
      const vis = visible(rec);
      rec.card.hidden = !vis;
      if (!vis) continue;
      const leg = typeof rec.c.legend === 'function' ? rec.c.legend(data, state, ctx) : rec.c.legend;
      rec.legend.replaceChildren(...(leg || []).map(([color, kind, text]) => h('span', { style: { color: `var(--${color})` } }, h('i' + (kind && kind !== 'line' ? '.' + kind : '')), text)));
      rec.legend.hidden = !leg?.length;
      const sub = typeof rec.c.sub === 'function' ? rec.c.sub(data, state, ctx) : rec.c.sub;
      rec.sub.innerHTML = sub || ''; rec.sub.hidden = !sub;
      const cap = typeof rec.c.caption === 'function' ? rec.c.caption(data, state, ctx) : rec.c.caption;
      rec.cap.innerHTML = cap || ''; rec.cap.hidden = !cap;
      if (rec.run) rec.run(); else safeDraw(rec);
    }
    if (spec.insights) insightsList.replaceChildren(...spec.insights(data, state).filter(Boolean).map(t => h('li', h('span', { html: t }))));
    drawIllustration();
  }
  function drawIllustration() {
    if (!spec.illustration) return;
    try { const svg = canvas(illHost, spec.illustration.height || 220); spec.illustration.draw(svg, data, state, ctx); }
    catch (e) { console.error(`[${spec.id}] illustration`, e); }
  }

  /* Tween a tile's number from its previous value, keeping the server's formatting
     (decimals, thousands separators, prefix and suffix). Text that is not a number just swaps. */
  function countUp(node, from, to) {
    if (!node || node.nodeType !== 3 || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const re = /^([^\d\-−]*)([\-−]?[\d,]*\.?\d+)(.*)$/s, a = from.match(re), b = to.match(re);
    if (!a || !b || a[1] !== b[1] || a[3] !== b[3]) return;
    const num = x => parseFloat(x.replace(/,/g, '').replace('−', '-'));
    const v0 = num(a[2]), v1 = num(b[2]);
    if (!Number.isFinite(v0) || !Number.isFinite(v1) || v0 === v1) return;
    const dec = (b[2].split('.')[1] || '').length, comma = b[2].includes(','), minus = b[2].startsWith('−') ? '−' : '-';
    const fmt = v => { let s = Math.abs(v).toFixed(dec); if (comma) s = Number(s).toLocaleString('en-US', { minimumFractionDigits: dec, maximumFractionDigits: dec }); return (v < 0 ? minus : '') + s; };
    const t0 = performance.now(), D = 520;
    const step = t => { const f = Math.min(1, (t - t0) / D), e = 1 - Math.pow(1 - f, 3); node.textContent = b[1] + fmt(v0 + (v1 - v0) * e) + b[3]; if (f < 1 && node.isConnected) requestAnimationFrame(step); else node.textContent = to; };
    node.textContent = from; requestAnimationFrame(step);
  }

  /* ---------------- header actions */
  async function copyLink() {
    try { await navigator.clipboard.writeText(location.href); toast('Link copied. It opens this view with these exact settings.'); }
    catch (e) { toast('Copy the address bar to share this setup.'); }
  }
  function reset() { spec.reset?.(ctx); set({ ...defaults }, { immediate: true }); }

  /* ---------------- background jobs */
  async function job(body, onProgress) {
    const prog = ctx._prog, bar = prog?.querySelector('i'), note = prog?.querySelector('.c-note');
    if (prog) { prog.hidden = false; bar.style.width = '0%'; }
    let { job: j } = await call('/api/jobs', body, { method: 'POST' });
    while (alive && (j.status === 'queued' || j.status === 'running')) {
      if (bar) bar.style.width = Math.round(j.progress * 100) + '%';
      if (note) note.textContent = `${j.message} · ${j.elapsed_s.toFixed(1)} s`;
      onProgress?.(j);
      await new Promise(r => setTimeout(r, 600));
      ({ job: j } = await call(`/api/jobs/${j.id}`));
    }
    if (bar) bar.style.width = '100%';
    if (j.status === 'failed') throw new Error(j.error || 'The job failed.');
    if (note) note.textContent = `Finished on the server in ${fmtMs(j.elapsed_s * 1000)}.`;
    ctx.lastJob = j;                     // includes elapsed_s, the server-side duration
    return j.result;
  }

  syncVisibility();
  drawIllustration();
  spec.init?.(ctx);
  now();

  return {
    state,
    destroy() { alive = false; ctrl?.abort(); clearTimeout(timer); disposers.forEach(d => d()); charts.forEach(r => r.v3?.dispose?.()); spec.destroy?.(ctx); },
  };
}
