// Big Data Lab · warehouse topic: OLAP operations, SQL grouping operators, and the lattice of cuboids.
// Every number, member list, table row and lattice node comes from the server: dataset facts from
// GET /api/catalog → bigdata_warehouse, results from /api/bigdata/{olap,cube,lattice}. This file only lays
// them out (tables, lattice diagrams, bar and line charts).
import { call, fmt, int } from '../../api.js';
import { catalog } from '../../catalog.js';
import { E, frame, hover, nearest } from '../../plot.js';
import { h } from '../../ui/dom.js';
import { S, edge, node, text } from '../../ui/illustrate.js';

const C = catalog.bigdata_warehouse;
const CUBES = C.cubes;
const get = (path, q, sig) => call(path, q, { signal: sig });
const nf = v => v == null ? '' : Number.isInteger(v) ? int(v) : fmt(v, 2);
const short = (s, n = 11) => (s = String(s)).length > n ? s.slice(0, n - 1) + '…' : s;
const tip = (title, rows) => `<b>${title}</b>` + rows.map(([k, v]) => `<br>${k}: <span class="n">${v}</span>`).join('');
const sql = (code, open = false, label = 'SQL') => h('details.wh-sql', { open }, h('summary', label), h('pre.out', code));
const chip = (text, cls = '') => h('span.chip' + cls, text);

/* ------------------------------------------------------------------ shared drawing helpers */

// A cuboid as a table: scalar, list, cross-tab (rows × nested columns) or relation.
// o.pick = {mode, dims, selected}: header labels on those dims become buttons calling o.onPick(dim, member).
function tableEl(t, o = {}) {
  const sel = o.pick?.selected || {}, has = Object.keys(sel).length > 0;
  const on = (dim, m) => (sel[dim] || []).includes(m);
  const inside = (dims, keys) => has && dims.every((d, i) => !sel[d] || sel[d].includes(keys[i]));
  const label = (dim, m, extra = {}) => {
    const can = o.onPick && o.pick?.dims?.includes(dim);
    const content = can ? h('button.wh-pick', { type: 'button', 'aria-pressed': String(on(dim, m)), title: o.pick.mode === 'slice' ? `Slice at ${m}` : `Add or remove ${m} from the dice`, onclick: () => o.onPick(dim, m) }, m) : m;
    return h('th' + (on(dim, m) ? '.on' : ''), extra, content);
  };
  const cell = (v, inn) => h('td.num' + (v == null ? '.empty' : '') + (inn ? '.in' : ''), v == null ? '·' : nf(v));
  if (t.view === 'scalar') return h('div.wh-scalar', h('span.eyebrow', `SUM(${t.measure}) over everything`), h('b', nf(t.total)), h('span', t.unit));
  if (t.view === 'list') {
    const d = t.rows.dim;
    return h('div.tbl-wrap', h('table.wh-tab', h('thead', h('tr', h('th', t.rows.level), h('th.num', t.measure))),
      h('tbody', t.rows.labels.map((m, i) => h('tr', label(d, m), cell(t.values[i], inside([d], [m]))))),
      h('tfoot', h('tr', h('th', 'Total'), h('td.num.tot', nf(t.total))))));
  }
  if (t.view === 'relation') {
    const dims = t.dims || [];
    return h('div.tbl-wrap.wh-scroll', h('table.wh-tab', h('thead', h('tr', t.columns.map((c, i) => h('th' + (i === t.columns.length - 1 ? '.num' : ''), c)))),
      h('tbody', t.rows.map(r => h('tr' + (inside(dims, r) ? '.in' : ''), r.slice(0, -1).map((v, i) => h('td', v)), h('td.num', nf(r[r.length - 1]))))),
      h('tfoot', h('tr', h('th', { colspan: t.columns.length - 1 }, `Total (${t.rows.length} rows)`), h('td.num.tot', nf(t.total))))));
  }
  // cross-tab
  const heads = t.cols, rd = t.rows.dim, dims = [rd, ...heads.map(x => x.dim)];
  const thead = h('thead');
  heads.forEach((hd, i) => {
    const span = heads.slice(i + 1).reduce((a, x) => a * x.labels.length, 1);
    const rep = heads.slice(0, i).reduce((a, x) => a * x.labels.length, 1);
    const tr = h('tr');
    if (i === 0) tr.append(h('th.wh-corner', { rowspan: heads.length }, `${t.rows.level} ╲ ${heads.map(x => x.level).join(' × ')}`));
    for (let r = 0; r < rep; r++) hd.labels.forEach(m => tr.append(label(hd.dim, m, { colspan: span, class: 'c' })));
    if (i === 0) tr.append(h('th.num', { rowspan: heads.length }, 'Total'));
    thead.append(tr);
  });
  const tbody = h('tbody', t.rows.labels.map((m, r) => h('tr', label(rd, m), t.cells[r].map((v, j) => cell(v, inside(dims, [m, ...t.col_keys[j]]))), h('td.num.tot', nf(t.row_totals[r])))));
  const tfoot = h('tfoot', h('tr', h('th', 'Total'), t.col_totals.map(v => h('td.num.tot', nf(v))), h('td.num.tot.grand', nf(t.total))));
  return h('div.tbl-wrap.wh-scroll', h('table.wh-tab.cross', thead, tbody, tfoot));
}

// A lattice diagram: nodes carry x, y in [0, 1] (apex at the top); opts style and label them.
function drawLattice(host, nodes, edges, o = {}) {
  const layers = {};
  nodes.forEach(n => { (layers[n.y] ||= []).push(n); });
  const nl = Object.keys(layers).length, widest = Math.max(...Object.values(layers).map(l => l.length));
  const W = 600, H = o.height || Math.max(170, 40 + nl * (o.rowH || 54));
  const nw = Math.min(o.maxW || 116, W / widest - 6), nh = o.nodeH || 34;
  const svg = S('svg', { class: 'wh-lat', viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': o.label || 'Lattice of cuboids' });
  const pos = {};
  nodes.forEach(n => { pos[n.id] = [n.x * W, 20 + nh / 2 + n.y * (H - 40 - nh)]; });
  const ge = S('g', {}, svg), gn = S('g', {}, svg);
  edges.forEach(([a, b]) => { const [x1, y1] = pos[a], [x2, y2] = pos[b]; S('line', { x1, y1, x2, y2, class: 'wh-edge ' + (o.edgeClass?.(a, b) || '') }, ge); });
  nodes.forEach(n => {
    const [cx, cy] = pos[n.id];
    const g = S('g', { class: 'wh-node ' + (o.nodeClass?.(n) || ''), transform: `translate(${cx - nw / 2},${cy - nh / 2})` }, gn);
    S('rect', { width: nw, height: nh, rx: 8 }, g);
    const t1 = S('text', { x: nw / 2, y: nh / 2 - (o.sub ? 2 : -4), 'text-anchor': 'middle', class: 'l1' }, g); t1.textContent = o.label1 ? o.label1(n) : n.label;
    if (o.sub) { const t2 = S('text', { x: nw / 2, y: nh / 2 + 11, 'text-anchor': 'middle', class: 'l2' }, g); t2.textContent = o.sub(n); }
    const ti = S('title', {}, g); ti.textContent = o.title ? o.title(n) : n.label;
    if (o.onClick) { g.style.cursor = 'pointer'; g.setAttribute('tabindex', '0'); g.setAttribute('role', 'button'); g.addEventListener('click', () => o.onClick(n)); g.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); o.onClick(n); } }); }
  });
  host.replaceChildren(svg);
  return svg;
}

function barChart(host, labels, values, o = {}) {
  const n = labels.length, top = Math.max(1, ...values.filter(v => v != null));
  const P = frame(host, { h: o.h || 240, x: [-0.6, n - 0.4], y: [0, top * 1.15], xt: labels.map((_, i) => i), xf: i => short(labels[i], o.cut || 11), yl: o.yl, xl: o.xl });
  values.forEach((v, i) => {
    if (v == null) return;
    const w = Math.min(0.38, 0.38), x0 = P.X(i - w), x1 = P.X(i + w);
    E('rect', { x: x0, y: P.Y(v), width: Math.max(1, x1 - x0), height: Math.max(0, P.Y(0) - P.Y(v)), rx: 3, class: 'wh-bar ' + (o.cls?.(i) || o.color || 's1') }, P.body);
    if (n <= 16) P.text((x0 + x1) / 2, P.Y(v) - 5, nf(v), { 'text-anchor': 'middle' });
  });
  hover(host, P, px => { const i = Math.round(P.xInv(px)); if (i < 0 || i >= n) return null; return { cross: P.X(i), html: tip(labels[i], [[o.yl || 'value', nf(values[i])]]) }; });
  return P;
}

/* ================================================================ 1 · OLAP operations */
const DSKEY = { hk: 'dim_hk', enrol: 'dim_en', fact30: 'dim_f30', lab: 'dim_lab' };
const dimOpts = k => CUBES[k].dims.map(d => [d.key, d.label]);
const dimParam = (k, dflt, label) => ({ k: DSKEY[k], type: 'seg', label: 'Dimension', value: dflt, options: dimOpts(k), group: 'Operation',
  show: s => s.ds === k && !['window', 'dice'].includes(s.op),
  help: `Which dimension of the ${CUBES[k].label.toLowerCase()} cube the operation acts on. Roll-up/drill-down climb or descend its hierarchy, slice fixes one of its members, pivot puts it on the rows.` });
const olapDim = s => s[DSKEY[s.ds]];

function pickHandler(ctx, d) {
  return (dim, m) => {
    const s = ctx.state;
    if (d.pick.mode === 'slice') { ctx.set({ [DSKEY[s.ds]]: dim, member: m }); return; }
    const sel = Object.fromEntries(Object.entries(d.pick.selected).map(([k, v]) => [k, [...v]]));
    const cur = sel[dim] || [];
    sel[dim] = cur.includes(m) ? cur.filter(x => x !== m) : [...cur, m];
    if (!sel[dim].length) delete sel[dim];
    const order = d.members;
    const str = Object.entries(sel).map(([k, v]) => `${k}:${order[k].filter(x => v.includes(x)).join('|')}`).join(';');
    ctx.set({ dice: str || '*' });
  };
}

const olap = {
  id: 'olap', group: 'bigdata', nav: 'OLAP operations', tag: 'DW', title: 'OLAP operations on a data cube',
  question: 'How do roll-up, drill-down, slice, dice and pivot turn one cube of numbers into the table you need, and what SQL does each one run?',
  params: [
    { k: 'op', type: 'seg', label: 'Operation', value: 'rollup', group: 'Operation', options: Object.entries(C.ops),
      help: 'Roll-up and drill-down move along a concept hierarchy (or add/remove whole dimensions); slice fixes one member; dice keeps a sub-cube; pivot rotates the axes; window computes running aggregates, RANK and LAG on the monthly table.' },
    { k: 'ds', type: 'seg', label: 'Cube', value: 'hk', group: 'Data', show: s => s.op !== 'window',
      options: Object.values(CUBES).map(c => [c.key, { hk: 'H&K sales', enrol: 'Enrolments', fact30: 'Fact table', lab: 'Lab supplies' }[c.key]]),
      help: `H&K sales: ${CUBES.hk.source}. Enrolments: ${CUBES.enrol.source}. Fact table: ${CUBES.fact30.source}. Lab supplies: ${CUBES.lab.source}.` },
    dimParam('hk', 'location'), dimParam('enrol', 'degree'), dimParam('fact30', 'item'), dimParam('lab', 'item'),
    { k: 'path', type: 'seg', label: 'How far', value: 'dim', group: 'Operation', show: s => ['rollup', 'drill'].includes(s.op) && s.op !== 'window',
      options: [['dim', 'Along this dimension'], ['apex', 'All the way']],
      help: 'Along this dimension: climb (or descend) only its hierarchy, other dimensions stay at the finest level. All the way: go between the base cuboid and the apex, dimension by dimension (the slide 22 and 27–29 sequences).' },
    { k: 'wkind', type: 'seg', label: 'Window', value: 'sliding', group: 'Window', show: s => s.op === 'window', options: [['sliding', 'Sliding'], ['tumbling', 'Tumbling']],
      help: 'Tumbling (fixed) windows do not overlap: each month belongs to one window. A sliding window moves one row at a time: ROWS BETWEEN n−1 PRECEDING AND CURRENT ROW.' },
    { k: 'wsize', label: 'Window size', min: 2, max: 4, step: 1, value: 3, unit: 'months', group: 'Window', show: s => s.op === 'window', help: 'Months in each window frame.' },
    { k: 'wagg', type: 'seg', label: 'Aggregate', value: 'sum', group: 'Window', show: s => s.op === 'window', options: [['sum', 'SUM'], ['avg', 'AVG'], ['max', 'MAX']], help: 'The aggregate computed over each frame.' },
    { k: 'witem', type: 'seg', label: 'Item', value: C.window.items[0], group: 'Window', show: s => s.op === 'window', options: C.window.items.map(i => [i, i]), help: 'The item whose frames, LAG and rank are tabulated.' },
  ],
  hidden: { member: '', dice: '' },
  presets: [
    { label: 'Slide 17 is the Vancouver slice', values: { op: 'slice', ds: 'hk', dim_hk: 'location', member: 'Vancouver' } },
    { label: 'Slide 30: BIT slice', values: { op: 'slice', ds: 'enrol', dim_en: 'degree', member: 'BIT' } },
    { label: 'Slide 31: dice', values: { op: 'dice', ds: 'enrol', dice: 'degree:BCom|BIT;country:Australia|India|Malaysia' } },
    { label: 'Slides 27–29: drill-down', values: { op: 'drill', ds: 'fact30', dim_f30: 'item', path: 'apex' } },
    { label: 'Cities → countries', values: { op: 'rollup', ds: 'hk', dim_hk: 'location', path: 'dim' } },
    { label: 'Slide 32: pivot', values: { op: 'pivot', ds: 'lab', dim_lab: 'item' } },
    { label: '3-month moving sum', values: { op: 'window', wkind: 'sliding', wsize: 3, wagg: 'sum' } },
  ],
  request: (s, sig) => s.op === 'window'
    ? get('/api/bigdata/olap', { op: 'window', wkind: s.wkind, wsize: s.wsize, wagg: s.wagg, witem: s.witem }, sig)
    : get('/api/bigdata/olap', { op: s.op, ds: s.ds, dim: olapDim(s), path: s.path, member: s.member, dice: s.dice }, sig),
  kpis: (d, s) => {
    if (d.op === 'window') {
      const w = d.window;
      return [
        { label: 'Frame', value: w.kind === 'sliding' ? `${w.size} rows` : `${w.windows.length} windows`, note: w.kind === 'sliding' ? `ROWS ${w.size - 1} PRECEDING · ${w.partial} short frames at the start` : `${w.size} months each, last one ${w.windows[w.windows.length - 1].months}` },
        { label: `Best ${w.agg.toUpperCase()} for ${w.item}`, value: nf(w.best.value), note: `frame ending ${w.best.month}` },
        { label: 'Biggest rise (LAG)', value: '+' + nf(w.biggest_rise.change), note: `${w.biggest_rise.month} vs the month before`, tone: 'good' },
        { label: 'Biggest fall (LAG)', value: nf(w.biggest_fall.change), note: `${w.biggest_fall.month} vs the month before`, tone: 'warn' },
      ];
    }
    const last = d.steps[d.steps.length - 1], ok = d.checks.filter(c => c.ok).length;
    return [
      { label: 'Lands on cuboid', value: last.cuboid.split(' | ')[0], note: last.group_by.length ? 'GROUP BY ' + last.group_by.join(', ') : 'no GROUP BY: the apex' },
      { label: 'Cells before → after', value: `${d.before_cells} → ${d.after_cells}`, note: `${d.steps.length} tables, one per step` },
      { label: 'Total after', value: nf(d.after_total), unit: d.ds.unit, note: d.preserved ? 'same total as before: nothing lost' : d.share != null ? `${(100 * d.share).toFixed(1)}% of the cube's total` : '', tone: d.preserved ? 'good' : undefined },
      d.checks.length ? { label: 'Checked against the slides', value: `${ok}/${d.checks.length}`, note: d.checks.map(c => c.slide).filter((x, i, a) => a.indexOf(x) === i).join(', '), tone: ok === d.checks.length ? 'good' : 'bad' }
        : { label: 'Lattice', value: `${d.lattice.count} cuboids`, note: `∏(Lᵢ + 1) over ${d.ds.dims.map(x => x.levels.length).join(' · ')} levels` },
    ];
  },
  illustration: {
    title: 'Where each dimension sits', height: 230,
    caption: 'Each column is a concept hierarchy (ALL at the top). The highlighted box is the level after the operation; the arrow is the move.',
    draw(svg, d, s) {
      if (s.op === 'window') {
        const months = C.window.months, w = d?.window, size = s.wsize, bw = 600 / months.length;
        text(svg, 300, 22, s.wkind === 'sliding' ? `Sliding frame: ROWS BETWEEN ${size - 1} PRECEDING AND CURRENT ROW` : `Tumbling windows of ${size} months`, 'n-title');
        months.forEach((m, i) => { S('rect', { x: i * bw + 4, y: 70, width: bw - 8, height: 40, rx: 6, class: 'node' }, svg); text(svg, i * bw + bw / 2, 95, m, 'n-sub'); });
        if (s.wkind === 'tumbling') {
          for (let a = 0, k = 0; a < months.length; a += size, k++) {
            const b = Math.min(months.length - 1, a + size - 1);
            S('rect', { x: a * bw + 1, y: 62, width: (b - a + 1) * bw - 2, height: 56, rx: 9, class: 'wh-frame' + (k % 2 ? ' alt' : '') }, svg);
            text(svg, (a + b + 1) * bw / 2, 140, w ? `W${k + 1}: ${nf(w.windows[k]?.values[s.witem])}` : `W${k + 1}`, 'n-sub');
          }
        } else {
          const t = w ? months.indexOf(w.best.month) : size - 1, a = Math.max(0, t - size + 1);
          S('rect', { x: a * bw + 1, y: 62, width: (t - a + 1) * bw - 2, height: 56, rx: 9, class: 'wh-frame' }, svg);
          text(svg, (a + t + 1) * bw / 2, 140, w ? `${w.agg.toUpperCase()} = ${nf(w.best.value)} (best frame for ${w.item})` : 'frame', 'n-sub');
        }
        text(svg, 300, 185, 'RANK() ranks items inside each month; LAG() reads the previous month of the same item.', 'n-sub');
        return;
      }
      const cube = CUBES[s.ds], dims = cube.dims, colW = 600 / dims.length;
      const ok = d && d.steps && d.ds.key === s.ds;
      const first = ok ? d.steps[0].levels : null, last = ok ? d.steps[d.steps.length - 1].levels : null;
      dims.forEach((dm, i) => {
        const levels = ['ALL', ...[...dm.levels].reverse()], cx = colW * (i + 0.5), bw = Math.min(130, colW - 30);
        text(svg, cx, 18, dm.label, 'n-title');
        const ys = {};
        levels.forEach((lv, j) => { const y = 30 + j * 42; ys[lv] = y + 15; node(svg, { x: cx - bw / 2, y, w: bw, h: 30, title: lv, hot: last && last[dm.key] === lv }); });
        if (first && last && first[dm.key] !== last[dm.key]) edge(svg, [cx + bw / 2 + 2, ys[first[dm.key]]], [cx + bw / 2 + 2, ys[last[dm.key]]], { curve: [26, 0], width: 2.5, flow: true });
        const sel = ok && d.pick.mode ? d.pick.selected[dm.key] : null;
        if (sel) text(svg, cx, 30 + levels.length * 42 + 6, `${d.pick.mode === 'slice' ? '=' : '∈'} ${short(sel.join(', '), 26)}`, 'edge-lbl');
      });
    },
  },
  chartLayout: 'one',
  charts: [
    { id: 'steps', kind: 'html', title: 'Step by step', show: d => d.op !== 'window',
      sub: (d, s) => d.pick?.mode === 'slice' ? 'Click a member in the first table to slice there.' : d.pick?.mode === 'dice' ? 'Click members in the first table to add them to the dice or remove them.' : 'Each table is a cuboid; open SQL to see the query that produces it from the fact table.',
      draw(host, d, s, ctx) {
        const onPick = d.pick?.mode ? pickHandler(ctx, d) : null, n = d.steps.length;
        const items = [];
        d.steps.forEach((st, i) => {
          items.push(h('section.wh-step' + (i === n - 1 ? '.final' : ''),
            h('header', h('span.wh-n', String(i + 1)), h('b', st.title), chip(st.cuboid, '.wh-cub'), h('span.wh-meta', `${st.cells} cell${st.cells === 1 ? '' : 's'} · total ${nf(st.total)}`)),
            st.table ? tableEl(st.table, i === 0 && onPick ? { pick: d.pick, onPick } : i === 0 ? {} : {}) : null,
            sql(st.sql, i === n - 1)));
        });
        const checks = d.checks.length ? h('div.wh-checks', h('span.eyebrow', 'Expected output'), d.checks.map(c => h('div.wh-check' + (c.ok ? '.ok' : '.bad'), h('b', c.ok ? '✓' : '✗'), ` ${c.label} `, h('span.wh-meta', `(${c.slide})`)))) : null;
        const notes = d.notes.length ? h('div.wh-notes', d.notes.map(t => h('p', t))) : null;
        host.replaceChildren(h('div.wh-steps', items, checks, notes));
      } },
    { id: 'lattice', kind: 'html', title: 'Where it lands in the lattice', show: d => d.op !== 'window',
      sub: d => `All ${d.lattice.count} cuboids of this cube (every level combination). The path follows the steps; the last node is the result.`,
      legend: [['accent', 'box', 'visited'], ['s2', 'box', 'result']],
      draw(host, d) {
        const vis = new Set(d.lattice.visited), path = new Set(d.lattice.visited.slice(1).map((v, i) => d.lattice.visited[i] + '>' + v));
        drawLattice(host, d.lattice.nodes, d.lattice.edges, {
          label: 'Lattice of cuboids for this cube', sub: n => `${n.cells} cell${n.cells === 1 ? '' : 's'}`, title: n => `${n.label}: GROUP BY ${n.title} → ${n.cells} cells`,
          nodeClass: n => (n.id === d.lattice.final ? 'final ' : '') + (vis.has(n.id) ? 'on' : ''),
          edgeClass: (a, b) => path.has(a + '>' + b) || path.has(b + '>' + a) ? 'on' : '', maxW: 96,
        });
      } },
    { id: 'bars', title: 'The result, totalled by row', show: d => d.op !== 'window', sub: d => d.bars.title,
      draw(host, d) { barChart(host, d.bars.labels, d.bars.values, { yl: d.ds.unit }); } },
    { id: 'wlines', title: 'Monthly amounts and the window aggregate', show: d => d.op === 'window',
      sub: d => `${d.window.agg.toUpperCase()} over ${d.window.kind} frames of ${d.window.size} months for ${d.window.item} (dashed), with every item's monthly amount.`,
      legend: d => [...d.window.items.map((it, i) => [`s${i + 1}`, 'line', it]), ['accent', 'dash', `${d.window.agg.toUpperCase()} frame, ${d.window.item}`]],
      draw(host, d) {
        const w = d.window, xs = w.months.map((_, i) => i), all = [...Object.values(w.amount).flat(), ...w.aggs[w.item]];
        const P = frame(host, { h: 300, x: [0, xs.length - 1], y: [0, Math.max(...all) * 1.08], xt: xs, xf: i => w.months[i], yl: 'amount' });
        if (w.kind === 'tumbling') w.windows.forEach((win, k) => { const a = w.months.indexOf(win.from), b = w.months.indexOf(win.to); if (k % 2) E('rect', { x: P.X(a - 0.5), y: P.M.t, width: P.X(b + 0.5) - P.X(a - 0.5), height: P.h - P.M.t - P.M.b, class: 'wh-band' }, P.body); });
        w.items.forEach((it, i) => P.line(xs.map(x => [x, w.amount[it][x]]), `s${i + 1}` + (it === w.item ? '' : ' thin')));
        P.line(xs.map(x => [x, w.aggs[w.item][x]]), 'wh-acc dashed');
        hover(host, P, px => { const i = nearest(xs, P.xInv(px)); return { cross: P.X(i), pts: [[P.X(i), P.Y(w.aggs[w.item][i]), 'f1']], html: tip(w.months[i], [...w.items.map(it => [it, nf(w.amount[it][i])]), [`${w.agg.toUpperCase()} (${w.item})`, nf(w.aggs[w.item][i])]]) }; });
      } },
    { id: 'wtable', kind: 'html', title: 'Frames, LAG and RANK row by row', show: d => d.op === 'window',
      sub: d => `${d.window.item}: each row's frame, the aggregate over it, the previous month (LAG) and the item's rank among the ${d.window.items.length} items that month.`,
      draw(host, d) {
        const w = d.window;
        const t = h('div.tbl-wrap', h('table.wh-tab', h('thead', h('tr', ['month', 'amount', 'frame', `${w.agg}`, 'LAG', 'change', 'RANK'].map((c, i) => h('th' + (i && i !== 2 ? '.num' : ''), c)))),
          h('tbody', w.rows.map(r => h('tr' + (r.month === w.best.month ? '.best' : ''), h('td', r.month), h('td.num', nf(r.amount)), h('td', `${r.frame} (${r.frame_rows})`), h('td.num', nf(r.agg)),
            h('td.num' + (r.lag == null ? '.empty' : ''), r.lag == null ? 'NULL' : nf(r.lag)), h('td.num' + (r.change > 0 ? '.up' : r.change < 0 ? '.down' : ''), r.change == null ? 'NULL' : (r.change > 0 ? '+' : '') + nf(r.change)), h('td.num', String(r.rank)))))));
        const rank = h('div.tbl-wrap', h('table.wh-tab.cross', h('thead', h('tr', h('th', 'RANK in month'), w.months.map(m => h('th.num', m)))),
          h('tbody', w.items.map((it, i) => h('tr' + (it === w.item ? '.in' : ''), h('th', it), w.rank[i].map(r => h('td.num' + (r === 1 ? '.in' : ''), String(r))))))));
        const wins = w.windows.length ? h('div.tbl-wrap', h('table.wh-tab', h('thead', h('tr', h('th', 'window'), h('th', 'months'), w.items.map(it => h('th.num', it)))),
          h('tbody', w.windows.map(x => h('tr', h('td', `W${x.id}`), h('td', `${x.from}–${x.to} (${x.months})`), w.items.map(it => h('td.num', nf(x.values[it])))))))) : null;
        host.replaceChildren(h('div.wh-steps', t, wins, rank, sql(w.sql.window, true, 'SQL: the window'), sql(w.sql.lag, false, 'SQL: LAG'), sql(w.sql.rank, false, 'SQL: RANK')));
      } },
  ],
  insights: (d, s) => {
    if (d.op === 'window') {
      const w = d.window;
      return [
        w.kind === 'sliding' ? `A sliding frame of ${w.size} months gives one value per month; the first ${w.partial} frame${w.partial === 1 ? ' is' : 's are'} shorter because there are not ${w.size - 1} earlier rows yet.` : `Tumbling windows split the ${w.months.length} months into ${w.windows.length} blocks; each month is counted once.`,
        `${w.item}'s biggest month-on-month change from LAG is <b>${w.biggest_rise.change > 0 ? '+' : ''}${nf(w.biggest_rise.change)}</b> in ${w.biggest_rise.month}, and its biggest drop is <b>${nf(w.biggest_fall.change)}</b> in ${w.biggest_fall.month}.`,
        'Window functions keep every row (unlike GROUP BY): the aggregate is attached to each row, which is why they suit running totals and rankings.',
      ];
    }
    const last = d.steps[d.steps.length - 1];
    return [
      ...d.notes.map(n => n),
      d.preserved ? `Every step keeps the grand total at <b>${nf(d.after_total)}</b>: roll-up and drill-down only regroup the same numbers.` : d.op === 'slice' ? `The slice keeps ${d.after_cells} of ${d.before_cells} cells, <b>${(100 * d.share).toFixed(1)}%</b> of the total. It reads part of the base cuboid; nothing new is computed.` : d.op === 'dice' ? `The dice keeps ${d.after_cells} of ${d.before_cells} cells (${(100 * d.share).toFixed(1)}% of the total) and, unlike a slice, keeps every dimension.` : '',
      `The result is cuboid <b>${last.cuboid}</b>: ${last.group_by.length ? 'GROUP BY ' + last.group_by.join(', ') : 'the apex, no GROUP BY'}.`,
      ...d.checks.filter(c => c.ok).slice(0, 2).map(c => `Matches ${c.slide}: ${c.label.toLowerCase()}.`),
    ].filter(Boolean);
  },
  maths: { eq: 'roll-up   C(i′,j,k)[p,j,k] = Σ_{c : parent(c) = p} C(i,j,k)[c,j,k]\nslice     σ_{i = x} C(i,j,k)  →  a table over (j, k)\ndice      σ_{i ∈ A ∧ j ∈ B} C(i,j,k)\npivot     rows ↔ columns (same cells)\nwindow    AGG(m) OVER (PARTITION BY item ORDER BY month ROWS n−1 PRECEDING)',
    notes: ['C(i,j,k) is the base cuboid; a prime marks a coarser level of the same dimension (i′ = province, i″ = country) and 0 means ALL, as on DW slide 22.', 'Roll-up and drill-down never change the grand total; slice and dice select, pivot only re-arranges.'] },
};

/* ================================================================ 2 · SQL grouping operators */
const ORDER_LABEL = o => o.split(',').join(', ');
const cube = {
  id: 'cube', group: 'bigdata', nav: 'Cube in SQL', tag: 'Cube SQL', title: 'GROUP BY, ROLLUP, CUBE and GROUPING SETS',
  question: 'Which grouping sets does each SQL operator produce, how do you build CUBE from plain GROUP BYs, and why does NULL need GROUPING()?',
  params: [
    { k: 'op', type: 'seg', label: 'Operator', value: 'cube', group: 'Query', options: [['groupby', 'GROUP BY'], ['rollup', 'ROLLUP'], ['cube', 'CUBE'], ['gsets', 'GROUPING SETS']],
      help: 'GROUP BY: one grouping set. ROLLUP(a,b,c): the n+1 prefixes. CUBE(a,b,c): all 2ⁿ subsets. GROUPING SETS (beyond the slides): exactly the sets you list.' },
    { k: 'fact', type: 'seg', label: 'Fact table', value: 'fact9', group: 'Query', options: Object.entries(C.facts).map(([k, f]) => [k, `${f.label}`]),
      help: `9 rows: ${C.facts.fact9.source} (total ${C.facts.fact9.total}). 30 rows: ${C.facts.fact30.source} (total ${C.facts.fact30.total}).` },
    { k: 'order', type: 'seg', label: 'Dimension order', value: 'store,item,cust', group: 'Query', options: C.orders.map(o => [o, ORDER_LABEL(o)]),
      help: 'The order matters for ROLLUP (it keeps prefixes) and for the rotations used to build CUBE from ROLLUPs. CUBE and GROUP BY ignore it.' },
    { k: 'ndims', type: 'seg', label: 'Dimensions used', value: 3, group: 'Query', options: [[3, 'All three'], [2, 'First two']], help: 'Group by all three dimensions, or only the first two of the order.' },
    { k: 'gs', type: 'seg', label: 'Grouping sets', value: 'pairs', group: 'Query', options: Object.entries(C.gsets), show: s => s.op === 'gsets', help: 'GROUPING SETS lists the sets explicitly; these are three common choices.' },
    { k: 'nul', type: 'switch', label: 'Add a sale with an unknown customer (real NULL)', value: false, group: 'NULL pitfall',
      help: 'Adds one fact row whose customer is NULL in the data. In the output it looks exactly like the ALL marker; GROUPING() tells them apart.' },
    { k: 'showall', type: 'switch', label: 'Label rolled-up columns ALL (uses GROUPING)', value: false, group: 'NULL pitfall',
      help: "CASE WHEN GROUPING(col) = 1 THEN 'ALL' ELSE col END: the fix for the ambiguity, applied to the result table." },
    { k: 'schema', type: 'seg', label: 'Schema', value: 'star', group: 'Schema', options: [['star', 'Star'], ['snowflake', 'Snowflake']], help: 'The Sales fact table with denormalised dimensions (star), or with the dimensions normalised into chains of tables (snowflake).' },
  ],
  presets: [
    { label: 'Slide 13: CUBE on 9 rows', values: { op: 'cube', fact: 'fact9', order: 'store,item,cust', ndims: 3, nul: false } },
    { label: 'Slide 24: ROLLUP(item, store, cust)', values: { op: 'rollup', fact: 'fact9', order: 'item,store,cust', ndims: 3, nul: false } },
    { label: 'Real NULL vs ALL', values: { op: 'cube', fact: 'fact9', nul: true, showall: false } },
    { label: 'GROUPING SETS (beyond the slides)', values: { op: 'gsets', gs: 'pairs', ndims: 3 } },
    { label: 'CUBE on the 30-row table', values: { op: 'cube', fact: 'fact30', order: 'item,store,cust', ndims: 3, nul: false } },
  ],
  request: (s, sig) => get('/api/bigdata/cube', { fact: s.fact, op: s.op, order: s.order.split(',').slice(0, s.ndims).join(','), gs: s.gs, null: s.nul }, sig),
  kpis: (d, s) => {
    const exp = d.fact.key === 'fact9' && !d.with_null && d.order.length === 3 ? (d.op === 'cube' ? 33 : d.op === 'rollup' ? 19 : null) : null;
    const c = d.constructions;
    return [
      { label: 'Result rows', value: String(d.row_count), note: exp ? `slide ${d.op === 'cube' ? 13 : 24} shows ${exp}` : `from ${d.fact.rows} fact rows`, tone: exp ? (exp === d.row_count ? 'good' : 'bad') : undefined },
      { label: 'Grouping sets', value: String(d.sets.length), note: d.sets.map(x => x.label).join(' ') },
      { label: 'Grand total (apex)', value: d.apex == null ? '—' : nf(d.apex), note: d.apex == null ? 'this operator has no () set' : `SUM(${d.fact.measure}) over ${d.fact.rows} rows` },
      { label: 'CUBE = ∪ rotated ROLLUPs', value: c.rot_equal ? '✓' : '✗', tone: c.rot_equal ? 'good' : 'bad', note: `UNION ${c.rot_union} rows; UNION ALL ${c.rot_union_all} (duplicates)` },
    ];
  },
  illustration: {
    title: 'From fact rows to grouping sets', height: 200,
    caption: 'The operator expands into grouping sets; each set is an ordinary GROUP BY, and the results are stacked with NULL in the columns that were aggregated away.',
    draw(svg, d, s) {
      const f = node(svg, { x: 8, y: 66, w: 130, h: 70, title: 'Sales', value: d ? `${d.fact.rows} rows` : '…', sub: d ? `total ${nf(d.fact.total)}` : '' });
      const g = node(svg, { x: 232, y: 56, w: 140, h: 90, title: { groupby: 'GROUP BY', rollup: 'ROLLUP', cube: 'CUBE', gsets: 'GROUPING SETS' }[s.op], value: d ? `${d.sets.length} set${d.sets.length === 1 ? '' : 's'}` : '…', sub: s.order.split(',').slice(0, s.ndims).join(', '), hot: true });
      const r = node(svg, { x: 462, y: 66, w: 130, h: 70, title: 'Result', value: d ? `${d.row_count} rows` : '…', sub: d ? `${d.null_rows} real NULL row${d.null_rows === 1 ? '' : 's'}` : '' });
      edge(svg, [f.r + 4, f.cy], [g.x - 6, g.cy], { label: 'scan once', width: 3, flow: true });
      edge(svg, [g.r + 4, g.cy], [r.x - 6, r.cy], { label: 'stack with NULL', width: 3, flow: true });
      if (d) text(svg, 300, 182, short(d.sets.map(x => x.label).join('  '), 80), 'n-sub');
    },
  },
  chartLayout: 'one',
  charts: [
    { id: 'sets', kind: 'html', title: 'Grouping sets on the lattice', sub: d => `The ${2 ** d.order.length} possible GROUP BYs over (${d.order.join(', ')}); highlighted are the ${d.sets.length} this operator runs. Each node shows its row count.`,
      legend: [['accent', 'box', 'produced by the operator']],
      draw(host, d) {
        const lat = h('div.wh-latbox');
        drawLattice(lat, d.lattice.nodes, d.lattice.edges, { label: 'Grouping-set lattice', sub: n => `${n.rows} row${n.rows === 1 ? '' : 's'}`, nodeClass: n => n.on ? 'on' : 'off', maxW: 130 });
        const list = h('div.tbl-wrap', h('table.wh-tab', h('thead', h('tr', h('th', 'grouping set'), d.order.map(x => h('th.num', `GROUPING(${x})`)), h('th.num', 'rows'), h('th.num', 'sum'))),
          h('tbody', d.sets.map(x => h('tr', h('td', h('code', x.label)), x.bits.map(b => h('td.num', String(b))), h('td.num', String(x.rows)), h('td.num', nf(x.total))))),
          h('tfoot', h('tr', h('th', 'Total rows'), d.order.map(() => h('td')), h('td.num.tot', String(d.row_count)), h('td')))));
        host.replaceChildren(h('div.wh-two', lat, list), sql(d.sql.native, true, 'SQL (PostgreSQL)'));
      } },
    { id: 'result', kind: 'html', title: 'The result, row by row', sub: (d, s) => s.showall ? "Rolled-up columns show ALL (GROUPING() = 1); a real NULL from the data stays NULL." : 'NULL in grey = the ALL marker (GROUPING() = 1). NULL in red = a real NULL from the data (GROUPING() = 0).',
      draw(host, d, s) {
        const val = (v, bit, real) => v != null ? v : bit ? h('span.wh-all', { title: 'ALL: this column was aggregated away (GROUPING = 1)' }, s.showall ? 'ALL' : 'NULL') : h('span.wh-realnull', { title: 'A real NULL in the data (GROUPING = 0)' }, 'NULL');
        const rows = d.rows.map(r => h('tr' + (r.set % 2 ? '.alt' : '') + (r.real_null.some(Boolean) ? '.warnrow' : ''), r.vals.map((v, j) => h('td', val(v, r.grouping[j]))), r.grouping.map(b => h('td.num.bits', String(b))), h('td.num', String(r.gid)), h('td.num', nf(r.amount))));
        host.replaceChildren(h('div', h('div.tbl-wrap.wh-scroll', h('table.wh-tab', h('thead', h('tr', d.order.map(x => h('th', x)), d.order.map(x => h('th.num', `g_${x}`)), h('th.num', 'GROUPING_ID'), h('th.num', `sum(${d.fact.measure})`))), h('tbody', rows))),
          d.ambiguous.length ? h('div.wh-notes.warn', h('p', h('b', `${d.ambiguous.length} pairs of rows look identical without GROUPING():`)), d.ambiguous.map(a => h('p', `(${a.vals.map(v => v ?? 'NULL').join(', ')}): ${nf(a.data_row)} is the unknown customer's own group (bits ${a.grouping_data.join('')}), ${nf(a.subtotal)} is the subtotal (bits ${a.grouping_subtotal.join('')}).`))) : null));
      } },
    { id: 'rows', title: 'Rows per grouping set', sub: d => `${d.row_count} rows in all. A set with fewer dimensions has fewer, larger groups.`,
      draw(host, d) { barChart(host, d.sets.map(x => x.label), d.sets.map(x => x.rows), { yl: 'rows', cut: 16, cls: i => d.sets[i].set.length === d.order.length ? 's1' : d.sets[i].set.length ? 's3' : 's2' }); } },
    { id: 'build', kind: 'html', title: 'Three ways to build CUBE', sub: d => `On (${d.order.join(', ')}): the operator, the 2ⁿ UNION of GROUP BYs (SQL deck slide 12), and the union of ${d.order.length} rotated ROLLUPs (slide 25).`,
      draw(host, d) {
        const c = d.constructions;
        const card = (title, rows, ok, note, code) => h('div.wh-card', h('b', title), h('span.wh-big', `${rows} rows`), h('span' + (ok ? '.wh-ok' : '.wh-bad'), ok ? '✓ same rows as CUBE' : '✗ differs from CUBE'), h('p', note), code);
        const cov = h('div.tbl-wrap', h('table.wh-tab', h('thead', h('tr', h('th', 'dimensions n'), h('th.num', '2ⁿ sets'), h('th.num', 'covered by n rotated ROLLUPs'), h('th', 'missing'))),
          h('tbody', c.coverage.map(r => h('tr' + (r.missing.length ? '' : '.in'), h('td', String(r.n)), h('td.num', String(r.all)), h('td.num', String(r.covered)), h('td', r.missing.length ? short(r.missing.join(' '), 40) : '—'))))));
        host.replaceChildren(h('div.wh-cards',
          card('GROUP BY CUBE(…)', c.cube_rows, true, 'One scan; the engine generates every subset.', sql(d.sql.cube, false)),
          card(`UNION ALL of ${2 ** d.order.length} GROUP BYs`, c.union_rows, c.union_equal, 'Each SELECT names NULL AS col for the columns it drops. UNION ALL keeps every row; UNION would also try to remove duplicates.', sql(d.sql.union)),
          card(`UNION of ${c.rotations.length} rotated ROLLUPs`, c.rot_union, c.rot_equal, `Covers ${c.rot_sets} distinct sets. With UNION ALL the apex and base rows repeat: ${c.rot_union_all} rows.`, sql(d.sql.rollups))),
          h('p.wh-meta', 'Why the rotation trick is limited to three dimensions:'), cov);
      } },
    { id: 'bugs', kind: 'html', title: 'The deck’s SQL, corrected', sub: 'Run on the materialised 9-row cube (33 rows). Row counts are computed on the server.',
      draw(host, d) {
        host.replaceChildren(h('div.tbl-wrap.wh-scroll', h('table.wh-tab.wh-bugs', h('thead', h('tr', h('th', 'slide'), h('th', 'as on the slide'), h('th.num', 'rows'), h('th', 'corrected'), h('th.num', 'rows'))),
          h('tbody', d.bugs.map(b => h('tr', h('td', h('b', b.slide), h('br'), b.query), h('td', h('pre.wh-code', b.deck_sql), h('p.wh-problem', b.problem)), h('td.num' + (b.deck_rows === 'error' ? '.bad' : ''), String(b.deck_rows)), h('td', h('pre.wh-code', b.fixed_sql)), h('td.num', String(b.fixed_rows))))))));
      } },
    { id: 'schema', kind: 'html', title: 'Star versus snowflake', sub: (d, s) => s.schema === 'star' ? 'SQL deck slide 5: one fact table, each dimension one join away.' : 'The same dimensions normalised (as the deck’s slide 4 does for Han & Kamber): fewer repeated values, longer join paths.',
      draw(host, d, s) {
        const sc = d.schema[s.schema];
        const P = s.schema === 'star'
          ? { Sales: [230, 95], Store: [20, 20], Item: [440, 20], Customer: [440, 185] }
          : { Sales: [230, 95], Store: [20, 120], City: [20, 230], County: [230, 245], Item: [440, 20], Category: [440, 130], Customer: [440, 240] };
        const W = 600, H = s.schema === 'star' ? 290 : 350, bw = 140, rh = 17;
        const svg = S('svg', { class: 'wh-schema', viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `${s.schema} schema` });
        const box = {};
        Object.entries(sc.tables).forEach(([t, cols]) => { const [x, y] = P[t]; box[t] = { x, y, w: bw, h: 24 + cols.length * rh }; });
        sc.edges.forEach(([a, b, k]) => { const A = box[a], B = box[b]; const l = S('line', { x1: A.x + bw / 2, y1: A.y + A.h / 2, x2: B.x + bw / 2, y2: B.y + B.h / 2, class: 'wh-edge on' }, svg); const t = S('text', { x: (A.x + B.x + bw) / 2, y: (A.y + A.h / 2 + B.y + B.h / 2) / 2 - 4, class: 'wh-key', 'text-anchor': 'middle' }, svg); t.textContent = k; void l; });
        Object.entries(sc.tables).forEach(([t, cols]) => {
          const b = box[t], g = S('g', { class: 'wh-table' + (t === 'Sales' ? ' fact' : '') }, svg);
          S('rect', { x: b.x, y: b.y, width: bw, height: b.h, rx: 8 }, g);
          const tt = S('text', { x: b.x + 10, y: b.y + 17, class: 'tn' }, g); tt.textContent = t;
          cols.forEach((c, i) => { const ct = S('text', { x: b.x + 10, y: b.y + 36 + i * rh, class: 'cn' }, g); ct.textContent = c; });
        });
        const joins = h('div.tbl-wrap', h('table.wh-tab', h('thead', h('tr', h('th', 'GROUP BY'), h('th.num', 'joins in a star'), h('th.num', 'joins in a snowflake'))),
          h('tbody', d.schema.attrs.map(a => h('tr', h('td', a), h('td.num', String(d.schema.star.joins[a])), h('td.num', String(d.schema.snowflake.joins[a])))))));
        const wrap = h('div.wh-latbox'); wrap.append(svg);
        host.replaceChildren(h('div.wh-two', wrap, joins));
      } },
  ],
  insights: (d, s) => [
    `${{ groupby: 'GROUP BY', rollup: 'ROLLUP', cube: 'CUBE', gsets: 'GROUPING SETS' }[d.op]} on (${d.order.join(', ')}) runs <b>${d.sets.length}</b> grouping set${d.sets.length === 1 ? '' : 's'} and returns <b>${d.row_count}</b> rows from ${d.fact.rows} fact rows.`,
    `CUBE equals the UNION ALL of all ${2 ** d.order.length} GROUP BYs (${d.constructions.union_rows} rows) and the UNION of the rotated ROLLUPs (${d.constructions.rot_union}); with UNION ALL the rotations give ${d.constructions.rot_union_all} rows because the apex and base sets repeat.`,
    d.with_null ? `With a real NULL customer, <b>${d.ambiguous.length}</b> pairs of rows print identically; only GROUPING() (or GROUPING_ID) separates the unknown customer from the subtotal.` : 'Turn on the real-NULL switch to see why a WHERE custid IS NULL test is not a safe way to find subtotals.',
    `In a snowflake, grouping by state needs ${d.schema.snowflake.joins.state} joins instead of ${d.schema.star.joins.state}.`,
  ],
  maths: { eq: 'ROLLUP(a,b,c)        = GROUPING SETS ((a,b,c), (a,b), (a), ())          n+1 sets\nCUBE(a,b,c)          = GROUPING SETS (all 2ⁿ subsets)                    2ⁿ sets\nGROUPING(x)          = 1 if x was aggregated away in this row, else 0\nGROUPING_ID(a,b,c)   = 4·g_a + 2·g_b + g_c',
    notes: ['CUBE(a,b,c) = ROLLUP(a,b,c) ∪ ROLLUP(b,c,a) ∪ ROLLUP(c,a,b) only for three dimensions: for four, the pairs (a,c) and (b,d) are never a prefix.'] },
};

/* ================================================================ 3 · lattice, hierarchies, HRU */
const LC = C.lattice;
const lattice = {
  id: 'lattice', group: 'bigdata', nav: 'Cube lattice', tag: 'Lattice', title: 'The lattice of cuboids and what to materialise',
  question: 'How many cuboids does a cube have, how are they computed layer by layer, which one answers a query, and which few are worth storing?',
  params: [
    { k: 'ds', type: 'seg', label: 'Data', value: 'hk', group: 'Cube', options: [['hk', 'H&K sales'], ['fact30', 'Fact table']],
      help: `H&K sales: ${LC.hk.dims.join(', ')} (${LC.hk.fact_rows} rows after the synthetic split). Fact table: ${LC.fact30.dims.join(', ')} (${LC.fact30.fact_rows} rows). ${LC.hk.synthetic.join(' and ')} are a documented, seeded extension (seed ${C.seed}).` },
    { k: 'n', label: 'Dimensions', min: 2, max: 5, step: 1, value: 4, group: 'Cube', help: 'Number of dimensions in the cube: 2ⁿ cuboids. Dimensions 4 and 5 are synthetic (supplier, channel).' },
    { k: 'mode', type: 'seg', label: 'Materialised', value: 'all', group: 'Materialisation', options: [['all', 'Full cube (Kylin)'], ['hru', 'HRU: k views']],
      help: 'Full cube: every cuboid is precomputed, as Kylin does (Lattice slides 21–22). HRU: only the base cuboid and the k views chosen greedily by benefit (an extension beyond the slides).' },
    { k: 'k', label: 'Views to materialise (HRU)', min: 0, max: 10, step: 1, value: 3, group: 'Materialisation', help: 'Rounds of the HRU greedy algorithm: each round adds the view with the largest benefit under the linear cost model.' },
    { k: 'where', type: 'switch', label: 'Add a WHERE on the first dimension', value: true, group: 'Query',
      help: 'Lattice slide 21: a filter on time means the cuboid must still contain time, so the query cannot use the smaller (item, location) cuboid. Click a node in the lattice to choose the GROUP BY.' },
  ],
  hidden: { q: 'item,location' },
  presets: [
    { label: 'Kylin slide 21 (WHERE time)', values: { ds: 'hk', n: 4, mode: 'all', where: true, q: 'item,location' } },
    { label: 'Kylin slide 22 (no WHERE)', values: { ds: 'hk', n: 4, mode: 'all', where: false, q: 'item,location' } },
    { label: 'HRU: 3 views', values: { ds: 'hk', n: 4, mode: 'hru', k: 3 } },
    { label: 'Five dimensions', values: { ds: 'hk', n: 5, mode: 'hru', k: 5 } },
    { label: 'FACT30 lattice', values: { ds: 'fact30', n: 3, mode: 'hru', k: 2, where: false, q: 'item,store' } },
  ],
  request: (s, sig) => get('/api/bigdata/lattice', { ds: s.ds, n: s.n, k: s.k, mode: s.mode, where: s.where, q: s.q }, sig),
  kpis: d => {
    const H = d.hru, q = d.query, red = H.totals.length > 1 ? 1 - H.totals[H.totals.length - 1] / H.totals[0] : 0;
    return [
      { label: 'Cuboids', value: String(d.cuboids), note: `2^${d.n}; with concept hierarchies ∏(Lᵢ + 1) = ${d.hierarchy.count}` },
      { label: 'Base cuboid', value: int(d.base_rows), unit: 'rows', note: `from ${int(d.fact_rows)} fact rows` },
      { label: 'Query reads', value: int(q.route.cost), unit: 'rows', note: `cuboid (${q.route.chosen.join(', ') || 'apex'}) · ${fmt(q.speedup, 1)}× fewer than the base`, tone: q.speedup > 1 ? 'good' : 'warn' },
      { label: `HRU, ${H.k} view${H.k === 1 ? '' : 's'}`, value: `−${(100 * red).toFixed(0)}%`, note: `total query cost ${int(H.totals[0])} → ${int(H.totals[H.totals.length - 1])} rows`, tone: red > 0 ? 'good' : undefined },
    ];
  },
  illustration: {
    title: 'Layer-by-layer cubing (MR layers)', height: 200,
    caption: 'MR1 reads the fact file and builds the base cuboid; every later layer drops one dimension and reads each cuboid’s smallest parent from the layer below.',
    draw(svg, d, s) {
      const n = d ? d.n : s.n, L = n + 1, bw = Math.min(96, (560 - 70) / L - 10), gap = (600 - 70 - L * bw) / (L + 1);
      const f = node(svg, { x: 4, y: 70, w: 64, h: 60, title: 'Fact', value: d ? int(d.fact_rows) : '…' });
      let prev = f;
      for (let i = 0; i < L; i++) {
        const dims = n - i, rows = d ? d.layers.filter(x => x.dims === dims).reduce((a, x) => a + x.rows, 0) : null;
        const cnt = d ? d.layers.filter(x => x.dims === dims).length : null;
        const b = node(svg, { x: 70 + gap + i * (bw + gap), y: 60, w: bw, h: 80, title: `MR${i + 1}`, value: rows == null ? '…' : int(rows), sub: `${dims}-D ×${cnt ?? ''}`, hot: i === 0 });
        edge(svg, [prev.r + 3, prev.cy], [b.x - 4, b.cy], { width: 2, flow: true });
        prev = b;
      }
      if (d) text(svg, 300, 178, `rows read: ${int(d.layer_reads)} layer by layer vs ${int(d.naive_reads)} if every cuboid rescans the fact file`, 'n-sub');
    },
  },
  chartLayout: 'one',
  charts: [
    { id: 'lattice', kind: 'html', title: 'The lattice (click a node to query it)',
      sub: d => `${d.cuboids} cuboids over ${d.dims.join(', ')}${d.synthetic.length ? ` (${d.synthetic.join(', ')} synthetic)` : ''}. Each node shows its distinct groups. ${d.n > 3 ? 'Letters: ' + d.dims.map(x => `${d.short[x]} = ${x}`).join(', ') + '.' : ''}`,
      legend: d => [['accent', 'box', d.query.mode === 'all' ? 'materialised (all)' : 'materialised (base + HRU)'], ['s2', 'box', 'answers the query'], ['s4', 'box', 'cuboid the query needs']],
      draw(host, d, s, ctx) {
        drawLattice(host, d.lattice.nodes, d.lattice.edges, {
          label: 'Lattice of cuboids', sub: n => `${int(n.rows)} rows`, maxW: d.n > 4 ? 60 : d.n === 4 ? 110 : 170, rowH: d.n > 4 ? 48 : 56,
          title: n => `GROUP BY ${n.dims.join(', ') || '(nothing)'}: ${n.rows} rows${n.materialised ? ' · materialised' : ''}`,
          nodeClass: n => [n.materialised && 'on', n.id === d.lattice.route && 'final', n.id === d.lattice.need && 'need'].filter(Boolean).join(' '),
          onClick: n => ctx.set({ q: n.dims.join(',') || 'all' }),
        });
      } },
    { id: 'router', kind: 'html', title: 'Routing the query', sub: d => `The query needs a cuboid containing {${d.query.needs.join(', ') || '∅'}}; it is answered by the smallest materialised ancestor.`,
      draw(host, d) {
        const q = d.query, r = q.route;
        const cand = h('div.tbl-wrap', h('table.wh-tab', h('thead', h('tr', h('th', 'materialised ancestor'), h('th.num', 'rows to scan'))),
          h('tbody', r.candidates.slice(0, 12).map((c, i) => h('tr' + (i === 0 ? '.best' : ''), h('td', `(${c.dims.join(', ')})`), h('td.num', int(c.rows)))))));
        const alt = q.mode === 'all' ? h('p.wh-meta', `With only the base and the ${d.hru.k} HRU view${d.hru.k === 1 ? '' : 's'}, it would read (${q.route_hru.chosen.join(', ')}): ${int(q.route_hru.cost)} rows.`) : null;
        host.replaceChildren(h('div.wh-two', h('div', sql(q.sql, true, 'The query'), h('p', `Answered from (${r.chosen.join(', ') || 'apex'}): `, h('b', `${int(r.cost)} rows`), ` instead of ${int(q.base_cost)} in the base cuboid.`), alt), cand));
      } },
    { id: 'layers', kind: 'html', title: 'Computing each cuboid from its smallest parent', sub: d => `Layer t builds the ${d.n}−t+1 dimensional cuboids. Map key = the child’s dimensions; value = SUM of the measure.`,
      draw(host, d) {
        host.replaceChildren(h('div.tbl-wrap.wh-scroll', h('table.wh-tab', h('thead', h('tr', h('th', 'layer'), h('th', 'cuboid'), h('th', 'map key'), h('th', 'read from'), h('th.num', 'rows read'), h('th.num', 'rows out'))),
          h('tbody', d.layers.map(x => h('tr' + (x.layer % 2 ? '' : '.alt'), h('td', `MR${x.layer}`), h('td', x.label), h('td', h('code', x.key)), h('td', x.parent), h('td.num', int(x.parent_rows)), h('td.num', int(x.rows))))),
          h('tfoot', h('tr', h('th', { colspan: 4 }, `Total rows read (vs ${int(d.base_reads)} from the base, ${int(d.naive_reads)} from the fact file)`), h('td.num.tot', int(d.layer_reads)), h('td'))))));
      } },
    { id: 'hru', kind: 'html', title: 'HRU greedy selection: benefit each round', show: d => d.hru.k > 0,
      sub: 'Extension beyond the slides (Harinarayan, Rajaraman, Ullman 1996). Benefit of v = Σ over views w ≤ v of max(0, cost(w) − |v|). The highlighted cell is the pick.',
      draw(host, d) {
        const views = d.sizes.filter(v => v.dims.length < d.n).sort((a, b) => a.rows - b.rows || a.dims.length - b.dims.length);
        const key = v => v.dims.join(',');
        const picked = d.hru.rounds.map(r => r.pick.join(','));
        host.replaceChildren(h('div.tbl-wrap.wh-scroll', h('table.wh-tab.cross', h('thead', h('tr', h('th', 'view'), h('th.num', 'rows'), d.hru.rounds.map((_, i) => h('th.num', `round ${i + 1}`)))),
          h('tbody', views.map(v => h('tr', h('th', `(${v.dims.join(', ') || 'apex'})`), h('td.num', int(v.rows)), d.hru.rounds.map((r, i) => {
            const k = key(v), b = r.benefits[k];
            return b == null ? h('td.num.empty', '—') : h('td.num' + (picked[i] === k ? '.in' : ''), int(b));
          })))))));
      } },
    { id: 'cost', title: 'Total query cost against views materialised', sub: d => `Sum over all ${d.cuboids} cuboids of the rows read to answer each, as HRU adds views. Dashed: everything materialised (${int(d.hru.all_cost)}); top: only the base (${int(d.hru.none_cost)}).`,
      legend: [['s1', 'line', 'HRU greedy'], ['s2', 'dash', 'full cube']],
      draw(host, d) {
        const c = d.hru.curve, xs = c.map((_, i) => i);
        const P = frame(host, { h: 260, x: [0, Math.max(1, xs.length - 1)], y: [0, d.hru.none_cost * 1.05], xl: 'views materialised besides the base (k)', yl: 'total rows read', xt: xs.filter(x => x % 1 === 0) });
        P.line(xs.map(x => [x, d.hru.all_cost]), 's2 dashed');
        P.line(xs.map(x => [x, c[x]]), 's1');
        xs.forEach(x => P.dot(x, c[x], x === d.hru.k ? 'f2' : 'f1', x === d.hru.k ? 6 : 3.5));
        hover(host, P, px => { const i = nearest(xs, P.xInv(px)); return { cross: P.X(i), pts: [[P.X(i), P.Y(c[i]), 'f1']], html: tip(`k = ${i}`, [['total cost', int(c[i])], ['vs full cube', int(d.hru.all_cost)]]) }; });
      } },
    { id: 'hier', title: 'Concept hierarchies multiply the cuboids', sub: d => `${d.hierarchy.levels.map(l => `${l.dim}: ${l.L} level${l.L === 1 ? '' : 's'}`).join(' · ')} → ∏(Lᵢ + 1) = ${d.hierarchy.count} cuboids (2^${d.n} = ${d.hierarchy.flat} without hierarchies), grouped by how many levels were rolled up.`,
      draw(host, d) { barChart(host, d.hierarchy.by_depth.map(x => `${x.depth}`), d.hierarchy.by_depth.map(x => x.cuboids), { yl: 'cuboids', xl: 'levels rolled up from the base', color: 's3' }); } },
  ],
  insights: d => {
    const H = d.hru, q = d.query;
    return [
      `The query ${q.filter ? `with WHERE ${q.filter} = '${q.filter_value}' ` : ''}needs (${q.needs.join(', ') || 'nothing'}) and reads <b>${int(q.route.cost)}</b> rows from (${q.route.chosen.join(', ') || 'apex'}) instead of ${int(q.base_cost)}${q.filter ? ': the filter column must survive in the cuboid, exactly the point of Lattice slides 21–22' : ''}.`,
      `Building every cuboid from its smallest parent reads <b>${int(d.layer_reads)}</b> rows in all, against ${int(d.naive_reads)} if each GROUP BY rescanned the fact file.`,
      H.k ? `HRU's first pick (${H.chosen[0].join(', ')}) has benefit ${int(H.rounds[0].benefit)}; benefits only shrink after that (${H.rounds.map(r => int(r.benefit)).join(' ≥ ')}), so the first few views buy most of the speed-up.` : 'Move k above 0 to let HRU choose views to materialise.',
      `With concept hierarchies the same ${d.n} dimensions give ${d.hierarchy.count} cuboids, not ${d.hierarchy.flat}: ∏(Lᵢ + 1).`,
    ];
  },
  maths: { eq: 'cuboids              2ⁿ, or ∏ᵢ (Lᵢ + 1) with Lᵢ hierarchy levels\nquery cost(w | S)    = min { |v| : v ∈ S, w ⊆ v }    (linear cost model)\nbenefit(v | S)       = Σ_{w ⊆ v} max(0, cost(w | S) − |v|)\nHRU                  S ← S ∪ { argmax_v benefit(v | S) },  k times',
    notes: ['|v| is the number of rows (distinct groups) in cuboid v, computed from the data on the server. The base cuboid is always materialised.', 'HRU is guaranteed to reach at least (1 − 1/e) ≈ 63% of the best possible benefit for the same k.'] },
};

export default [olap, cube, lattice];
