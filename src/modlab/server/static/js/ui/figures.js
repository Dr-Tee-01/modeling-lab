// Renderers for explainer figures. The server sends a figure as data (see
// modlab/explain/registry.py for the kinds); these functions only draw it.
import { E, frame, hover, nearest, niceTicks } from '../plot.js';
import { canvas, edge, node, text } from './illustrate.js';

const SERIES = new Set(['s1', 's2', 's3', 's4']);
const col = c => `var(--${c || 's1'})`;
const finite = v => v != null && Number.isFinite(v);
export const fmtNum = v => {
  if (!finite(v)) return '—';
  const a = Math.abs(v);
  if (a >= 1e4) return Math.round(v).toLocaleString('en-US');
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  if (a === 0) return '0';
  return v.toPrecision(2);
};
const tip = (title, rows) => `<b>${title}</b>` + rows.map(([k, v, c]) => `<br>${c ? `<i class="sw" style="background:${col(c)}"></i>` : ''}${k}: <span class="n">${v}</span>`).join('');
const SUP = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
const logTicks = (lo, hi) => {
  const e0 = Math.floor(Math.log10(lo)), e1 = Math.ceil(Math.log10(hi)), span = e1 - e0;
  const mult = span <= 2 ? [1, 2, 5] : [1], every = Math.max(1, Math.ceil(span / 6)), out = [];
  for (let e = e0; e <= e1; e++) if ((e - e0) % every === 0) for (const m of mult) { const v = m * 10 ** e; if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v); }
  return out.length >= 2 ? out : [lo, hi];
};
const logFmt = v => { if (v >= 1e-3 && v < 1e5) return String(+v.toPrecision(3)); const e = Math.round(Math.log10(v)); return Math.abs(v / 10 ** e - 1) < 1e-6 ? '10' + String(e).replace(/./g, c => SUP[c] ?? c) : v.toExponential(0); };
const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const dateFmt = v => { const y = Math.floor(v + 1e-9), m = Math.min(11, Math.floor((v - y) * 12 + 1e-6)); return `${MON[m]} ${y}`; };
const dateTick = v => Math.abs(v - Math.round(v)) < 1e-6 ? String(Math.round(v)) : dateFmt(v);
const range = (vals, log, pad = 0.05) => {
  const v = vals.filter(x => finite(x) && (!log || x > 0));
  if (!v.length) return [0, 1];
  let lo = Math.min(...v), hi = Math.max(...v);
  if (log) return [lo / 1.3, hi * 1.3];
  if (lo === hi) { lo -= 1; hi += 1; }
  const p = (hi - lo) * pad; return [lo - (lo >= 0 && lo - p < 0 ? lo : p), hi + p];
};
let uid = 0;
function gradient(svg, color, top = 0.3) {
  const defs = svg.querySelector('defs') || E('defs', {}, svg), id = `fg${++uid}`;
  const g = E('linearGradient', { id, x1: 0, y1: 0, x2: 0, y2: 1 }, defs);
  E('stop', { offset: 0, style: `stop-color:${col(color)};stop-opacity:${top}` }, g);
  E('stop', { offset: 1, style: `stop-color:${col(color)};stop-opacity:0` }, g);
  return `url(#${id})`;
}
function styleLine(p, s) {
  p.style.stroke = col(s.color);
  if (!s.thin) p.style.filter = `drop-shadow(0 0 5px color-mix(in srgb, ${col(s.color)} 55%, transparent))`;
  if (s.dash) p.classList.add('dashed');
  if (s.thin) p.classList.add('thin');
  return p;
}
const axisOpts = (f, xr, yr) => ({
  h: f.height || 300, x: xr, y: yr, xlog: f.xlog, ylog: f.ylog, xl: f.xlabel, yl: f.ylabel,
  xt: f.xlog ? logTicks(...xr) : undefined, yt: f.ylog ? logTicks(...yr) : undefined,
  xf: f.xdate ? dateTick : f.xlog ? logFmt : Math.abs(xr[1]) >= 1e4 ? fmtNum : undefined, yf: f.ylog ? logFmt : Math.abs(yr[1]) >= 1e4 ? fmtNum : undefined,
});

function hlines(P, f) {
  for (const l of f.hlines || []) {
    const y = P.Y(l.y); if (!finite(y)) continue;
    const ln = E('line', { x1: P.M.l, x2: P.w - P.M.r, y1: y, y2: y, class: 'fig-marker' }, P.body);
    if (l.color) ln.style.stroke = col(l.color);
    if (l.label) P.text(P.w - P.M.r - 4, y - 5, l.label, { 'text-anchor': 'end', class: 'dlabel fig-marker-lbl' });
  }
}
const xname = f => f.xdate ? dateFmt : fmtNum;

/* ------------------------------------------------------------------ lines */
function lines(host, f) {
  const xs = f.x, ys = f.series.flatMap(s => s.y).concat((f.points || []).map(p => p.y));
  const xr = range(xs, f.xlog, 0); let yr = range(ys, f.ylog);
  if (f.ymin != null) yr[0] = f.ymin; if (f.ymax != null) yr[1] = f.ymax;
  const P = frame(host, axisOpts(f, [Math.min(...xs), Math.max(...xs)], yr));
  for (const b of f.bands || []) {
    const x0 = P.X(b.x0), x1 = P.X(b.x1);
    E('rect', { x: x0, width: Math.max(0, x1 - x0), y: P.M.t, height: P.h - P.M.t - P.M.b, class: 'fig-band' }, P.body);
    if (b.label) P.text((x0 + x1) / 2, P.M.t + 14, b.label, { 'text-anchor': 'middle', class: 'dlabel fig-band-lbl' });
  }
  const base = P.Y(f.ylog ? yr[0] : Math.max(yr[0], 0));
  f.series.forEach((s, i) => {
    const pts = xs.map((x, j) => [x, s.y[j]]);
    if (s.area) {
      const d = pts.filter(p => finite(p[1])).map((p, j) => `${j ? 'L' : 'M'}${P.X(p[0]).toFixed(1)},${P.Y(p[1]).toFixed(1)}`).join('');
      const first = pts.find(p => finite(p[1])), last = [...pts].reverse().find(p => finite(p[1]));
      if (first) E('path', { d: `${d}L${P.X(last[0]).toFixed(1)},${base}L${P.X(first[0]).toFixed(1)},${base}Z`, fill: gradient(P.svg, s.color, i ? 0.22 : 0.32), class: 'glow-area draw' }, P.body);
    }
    styleLine(P.line(pts, ''), s);
  });
  hlines(P, f);
  for (const m of f.markers || []) {
    const x = P.X(m.x); if (!finite(x)) continue;
    E('line', { x1: x, x2: x, y1: P.M.t, y2: P.h - P.M.b, class: 'fig-marker' }, P.body);
    if (m.label) P.text(x + 5, P.M.t + 12, m.label, { class: 'dlabel fig-marker-lbl' });
  }
  for (const p of f.points || []) {
    const c = P.dot(p.x, p.y, '', p.r || 4.5); c.style.fill = col(p.color);
    if (p.label) P.text(P.X(p.x) + 8, P.Y(p.y) - 8, p.label);
  }
  const shown = f.series.filter(s => s.hover !== false);
  hover(host, P, px => {
    const i = nearest(xs, P.xInv(px)), x = xs[i];
    return { cross: P.X(x), pts: shown.filter(s => finite(s.y[i])).map(s => [P.X(x), P.Y(s.y[i]), '']),
      html: tip(f.xdate ? dateFmt(x) : `${f.xlabel || 'x'} ${fmtNum(x)}`, shown.map(s => [s.name, fmtNum(s.y[i]), s.color])) };
  });
  host.querySelectorAll('.tip').forEach(t => t.classList.add('fig-tip'));
}

/* ------------------------------------------------------------------ bars */
function bars(host, f) {
  const n = f.labels.length, k = f.series.length, vals = f.series.flatMap(s => s.values);
  const yr = range(vals.concat([0]), false);
  const P = frame(host, { h: f.height || 280, x: [-0.6, n - 0.4], y: [Math.min(0, yr[0]), yr[1]], xt: f.labels.map((_, i) => i), xf: i => f.labels[i] ?? '', yl: f.ylabel, xl: f.xlabel });
  const bw = Math.min(46, (P.X(1) - P.X(0)) * 0.78 / k);
  f.series.forEach((s, j) => s.values.forEach((v, i) => {
    if (!finite(v)) return;
    const x = P.X(i) - (k * bw) / 2 + j * bw, y0 = P.Y(0), y1 = P.Y(v);
    const r = E('rect', { x: x + 1, width: bw - 2, y: Math.min(y0, y1), height: Math.max(1, Math.abs(y0 - y1)), rx: 4, class: 'fig-bar' }, P.body);
    r.style.fill = gradient(P.svg, s.color, 0.95); r.style.stroke = col(s.color);
  }));
  hlines(P, f);
  hover(host, P, px => { const i = Math.max(0, Math.min(n - 1, Math.round(P.xInv(px)))); return { cross: P.X(i), html: tip(f.labels[i], f.series.map(s => [s.name, fmtNum(s.values[i]), s.color])) }; });
}

/* ------------------------------------------------------------------ scatter and paths */
function equalFrame(host, f, xs, ys) {
  let xr = range(xs, false, 0.08), yr = range(ys, false, 0.08);
  if (f.equal) {
    const w = Math.max(260, host.clientWidth) - 68, hgt = Math.max(220, Math.min(480, w * (yr[1] - yr[0]) / (xr[1] - xr[0]))) - 40;
    const sx = (xr[1] - xr[0]) / w, sy = (yr[1] - yr[0]) / hgt, s = Math.max(sx, sy);
    const cx = (xr[0] + xr[1]) / 2, cy = (yr[0] + yr[1]) / 2;
    xr = [cx - s * w / 2, cx + s * w / 2]; yr = [cy - s * hgt / 2, cy + s * hgt / 2];
    f = { ...f, height: hgt + 40 + (f.xlabel ? 14 : 0) };
  }
  const P = frame(host, { ...axisOpts(f, xr, yr), m: f.axes === false ? { l: 8, r: 8, t: 8, b: 8 } : undefined });
  if (f.axes === false) P.svg.classList.add('noaxes');
  return P;
}
function scatter(host, f) {
  const xs = f.points.map(p => p.x).concat((f.lines || []).flatMap(l => l.x)), ys = f.points.map(p => p.y).concat((f.lines || []).flatMap(l => l.y));
  const P = equalFrame(host, f, xs, ys);
  hlines(P, f);
  for (const l of f.lines || []) styleLine(P.line(l.x.map((x, i) => [x, l.y[i]]), ''), l);
  for (const p of f.points) { const c = P.dot(p.x, p.y, '', p.r || 4); c.style.fill = col(p.color); if (p.label) P.text(P.X(p.x) + 7, P.Y(p.y) - 7, p.label); }
  hover(host, P, (px, py) => {
    let best = null, bd = 400;
    f.points.forEach(p => { const d = (P.X(p.x) - px) ** 2 + (P.Y(p.y) - py) ** 2; if (d < bd) { bd = d; best = p; } });
    return best && { pts: [[P.X(best.x), P.Y(best.y), '']], html: tip(best.name || best.label || 'Point', [[f.xlabel || 'x', xname(f)(best.x)], [f.ylabel || 'y', fmtNum(best.y)]]) };
  });
}
function paths(host, f) {
  const all = f.paths.flatMap(p => p.points).concat((f.dots || []).map(d => [d.x, d.y]));
  const P = equalFrame(host, { equal: true, ...f }, all.map(p => p[0]), all.map(p => p[1]));
  for (const p of f.paths) {
    const d = p.points.map((q, i) => `${i ? 'L' : 'M'}${P.X(q[0]).toFixed(1)},${P.Y(q[1]).toFixed(1)}`).join('') + (p.closed ? 'Z' : '');
    if (p.fill) E('path', { d, style: `fill:color-mix(in srgb, ${col(p.color)} 16%, transparent)` }, P.body);
    const ln = E('path', { d, class: 'ln' + (p.dash ? ' dashed' : ''), style: `stroke:${col(p.color)};stroke-width:${p.width || 2};fill:none` }, P.body);
    if (!p.thin && (p.width || 2) >= 2) ln.style.filter = `drop-shadow(0 0 4px color-mix(in srgb, ${col(p.color)} 50%, transparent))`;
  }
  for (const d of f.dots || []) { const c = P.dot(d.x, d.y, '', d.r || 4.5); c.style.fill = col(d.color); if (d.label) P.text(P.X(d.x) + 8, P.Y(d.y) - 8, d.label); }
}

/* ------------------------------------------------------------------ flow diagrams */
function clip(a, b) {             // point where the segment from a's centre to b's centre leaves box a
  const dx = b.cx - a.cx, dy = b.cy - a.cy;
  const t = Math.min(dx ? (a.w / 2 + 5) / Math.abs(dx) : Infinity, dy ? (a.h / 2 + 5) / Math.abs(dy) : Infinity);
  return [a.cx + dx * t, a.cy + dy * t];
}
function flow(host, f) {
  host.dataset.k = host.dataset.k || `fig${++uid}`;
  const svg = canvas(host, f.height || 220), boxes = {};
  for (const n of f.nodes) {
    const fc = n.fill_color && SERIES.has(n.fill_color) ? 'fill' + n.fill_color.slice(1) : 'fill1';
    boxes[n.id] = node(svg, { x: n.x, y: n.y, w: n.w || 130, h: n.h || 64, title: n.title, sub: n.sub, value: n.value, hot: n.hot, fill: n.fill, fillCls: fc });
  }
  for (const e of f.edges) {
    const a = boxes[e.from], b = boxes[e.to]; if (!a || !b) continue;
    const g = edge(svg, clip(a, b), clip(b, a), { label: e.label, below: e.below, width: e.width ?? 2.5, flow: e.flow, curve: e.curve, cls: SERIES.has(e.color) ? 'st' + e.color.slice(1) : '' });
    if (e.color && !SERIES.has(e.color)) g.querySelector('path').style.stroke = col(e.color);
  }
  for (const t of f.texts || []) text(svg, t.x, t.y, t.text, t.cls || 'n-sub', t.anchor || 'middle');
}

/* ------------------------------------------------------------------ grids (heatmaps, boards) */
function grid(host, f) {
  host.querySelectorAll('.fig-gridwrap, canvas').forEach(c => c.remove());
  const axes = f.xlabel || f.ylabel || f.xticks || f.yticks;
  const padL = axes ? 46 : 0, padB = axes ? 38 : 0;
  const cv = document.createElement('canvas'); cv.className = 'fig-grid';
  const W = Math.max(200, host.clientWidth - padL), cell = Math.max(2, Math.floor(Math.min(W / f.cols, (f.height || 360) / f.rows)));
  const dpr = Math.min(2, devicePixelRatio || 1), gw = f.cols * cell, gh = f.rows * cell;
  cv.width = gw * dpr; cv.height = gh * dpr; cv.style.width = gw + 'px'; cv.style.height = gh + 'px';
  const c = cv.getContext('2d'); c.scale(dpr, dpr);
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  c.fillStyle = css('--sunk'); c.fillRect(0, 0, gw, gh);
  const colors = (f.colors || ['neutral', 's1', 's2', 's3', 's4']).map(t => css('--' + t));
  const pos = css('--' + (f.color || 'accent')), neg = css('--s1'), hot = css('--bad');
  f.cells.forEach((v, i) => {
    if (v == null) return;
    const x = (i % f.cols) * cell, y = Math.floor(i / f.cols) * cell;
    if (f.palette === 'cat') { if (!v) return; c.globalAlpha = 1; c.fillStyle = colors[v % colors.length]; }
    else if (f.palette === 'div') { c.globalAlpha = Math.min(1, Math.abs(v)); c.fillStyle = v < 0 ? neg : hot; }
    else { c.globalAlpha = Math.max(0, Math.min(1, v)); c.fillStyle = pos; }
    c.fillRect(x + (cell > 6 ? 0.5 : 0), y + (cell > 6 ? 0.5 : 0), cell - (cell > 6 ? 1 : 0), cell - (cell > 6 ? 1 : 0));
  });
  c.globalAlpha = 1;
  if (!axes) { host.append(cv); return; }
  // Axis titles and tick labels around the grid: ticks are [index, text] at cell centres.
  const wrap = document.createElement('div'); wrap.className = 'fig-gridwrap';
  wrap.style.width = gw + padL + 'px'; wrap.style.height = gh + padB + 'px';
  cv.style.left = padL + 'px'; wrap.append(cv);
  const lab = (text, cls, style) => { const d = document.createElement('span'); d.className = cls; d.textContent = text; Object.assign(d.style, style); wrap.append(d); };
  for (const [i, t] of f.xticks || []) lab(t, 'fig-gtick x', { left: padL + (i + 0.5) * cell + 'px', top: gh + 4 + 'px' });
  for (const [i, t] of f.yticks || []) lab(t, 'fig-gtick y', { left: padL - 6 + 'px', top: (i + 0.5) * cell + 'px' });
  if (f.xlabel) lab(f.xlabel, 'fig-glabel x', { left: padL + gw / 2 + 'px', top: gh + 20 + 'px' });
  if (f.ylabel) lab(f.ylabel, 'fig-glabel y', { left: '0px', top: gh / 2 + 'px' });
  host.append(wrap);
}

/* ------------------------------------------------------------------ stats */
function stats(host, f) {
  host.replaceChildren(...f.items.map(it => {
    const d = document.createElement('div'); d.className = 'fig-stat' + (it.tone ? ' ' + it.tone : '');
    const l = document.createElement('span'); l.className = 'k-label'; l.textContent = it.label;
    const v = document.createElement('b'); v.textContent = it.value;
    d.append(l, v);
    if (it.note) { const n = document.createElement('span'); n.className = 'k-note'; n.textContent = it.note; d.append(n); }
    return d;
  }));
}

const KINDS = { lines, bars, scatter, paths, flow, grid, stats };
export const svgKind = k => !['flow', 'grid', 'stats'].includes(k);
export function drawFigure(host, f) {
  const fn = KINDS[f.kind];
  if (!fn) { host.textContent = `Unknown figure kind "${f.kind}".`; return; }
  fn(host, f);
}

/* Legend entries for a figure: [{name, color, dash}] */
export function legendOf(f) {
  if (Array.isArray(f.legend)) return f.legend;
  if (f.kind === 'lines') return f.series.filter(s => s.legend !== false).map(s => ({ name: s.name, color: s.color, dash: s.dash }));
  if (f.kind === 'bars') return f.series.length > 1 ? f.series.map(s => ({ name: s.name, color: s.color, box: true })) : [];
  if (f.kind === 'paths') return f.paths.filter(p => p.name).map(p => ({ name: p.name, color: p.color, dash: p.dash }));
  if (f.kind === 'scatter') return (f.lines || []).filter(l => l.name).map(l => ({ name: l.name, color: l.color, dash: l.dash }));
  return [];
}
export { niceTicks };
