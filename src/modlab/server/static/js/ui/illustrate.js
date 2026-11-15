// Live schematic diagrams. Every illustration is drawn in a fixed 600-wide viewBox and
// scales to its card, so all views share one visual language: boxes (nodes), arrows
// (edges) whose thickness can carry a rate, and small labels.

const NS = 'http://www.w3.org/2000/svg';
export function S(tag, attrs = {}, parent) {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) if (attrs[k] != null) n.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(n);
  return n;
}
const T = (parent, x, y, text, cls, attrs = {}) => { const t = S('text', Object.assign({ x, y, class: cls }, attrs), parent); t.textContent = text; return t; };

const calm = () => typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;

/* Shared paint servers: gradient fills for boxes, a soft glow, gradient text, an arrow head.
   Every diagram SVG carries its own copy (same ids, same content), so any one of them resolves. */
export function paints(svg, key) {
  const defs = S('defs', {}, svg);
  const grad = (id, stops, attrs = { x1: 0, y1: 0, x2: 0, y2: 1 }) => {
    const g = S('linearGradient', { id, ...attrs }, defs);
    stops.forEach(([o, c]) => S('stop', { offset: o, style: `stop-color:${c}` }, g));
  };
  grad('ill-node', [[0, 'color-mix(in srgb,var(--surface-2) 92%,var(--accent))'], [1, 'var(--sunk)']]);
  grad('ill-hot', [[0, 'color-mix(in srgb,var(--accent) 34%,var(--surface))'], [1, 'color-mix(in srgb,var(--v1) 26%,var(--surface))']], { x1: 0, y1: 0, x2: 1, y2: 1 });
  grad('ill-text', [[0, 'var(--accent-2)'], [1, 'var(--v1)']], { x1: 0, y1: 0, x2: 1, y2: 0 });
  const f = S('filter', { id: 'ill-glow', x: '-30%', y: '-30%', width: '160%', height: '160%' }, defs);
  S('feGaussianBlur', { in: 'SourceGraphic', stdDeviation: 3.2, result: 'b' }, f);
  const mg = S('feMerge', {}, f); S('feMergeNode', { in: 'b' }, mg); S('feMergeNode', { in: 'SourceGraphic' }, mg);
  const m = S('marker', { id: 'ah-' + key, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 12, markerHeight: 12, markerUnits: 'userSpaceOnUse', orient: 'auto-start-reverse' }, defs);
  S('path', { d: 'M0 0 L10 5 L0 10 z', style: 'fill:color-mix(in srgb,var(--accent-2) 60%,var(--ink-2))' }, m);
  svg.dataset.marker = 'ah-' + key;
  return defs;
}

/* canvas(host, height): clears host and returns an <svg> with the shared paints. */
export function canvas(host, height = 220) {
  host.innerHTML = '';
  const svg = S('svg', { class: 'ill', viewBox: `0 0 600 ${height}`, role: 'img' }, host);
  paints(svg, host.dataset.k);
  return svg;
}

/* node(svg, {x, y, w, h, title, sub, value, hot, fill}): a rounded box with up to three lines.
   fill: 0..1 draws a level inside the box (a tank), with fillCls choosing the colour. */
export function node(svg, o) {
  const g = S('g', {}, svg);
  const { x, y, w = 130, h = 64 } = o;
  S('rect', { x, y, width: w, height: h, rx: 10, class: 'node' + (o.hot ? ' hot' : '') }, g);
  if (o.fill != null) {
    const f = Math.max(0, Math.min(1, o.fill));
    S('rect', { x: x + 1, y: y + h - 1 - (h - 2) * f, width: w - 2, height: (h - 2) * f, rx: 9, class: o.fillCls || 'fill1', opacity: 0.22 }, g);
  }
  const lines = [o.title && ['n-title', o.title], o.value != null && ['n-val', o.value], o.sub && ['n-sub', o.sub]].filter(Boolean);
  const gap = 18, y0 = y + h / 2 - (lines.length - 1) * gap / 2 + 5;
  lines.forEach(([cls, text], i) => T(g, x + w / 2, y0 + i * gap, text, cls, { 'text-anchor': 'middle' }));
  return { x, y, w, h, cx: x + w / 2, cy: y + h / 2, r: x + w, b: y + h };
}

/* edge(svg, [x1,y1], [x2,y2], {label, width, cls, curve, flow, labelAt}) */
export function edge(svg, a, b, o = {}) {
  const g = S('g', {}, svg);
  const width = Math.max(1, o.width ?? 2);
  const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
  const d = o.curve ? `M${a[0]},${a[1]} Q${mx + (o.curve[0] || 0)},${my + (o.curve[1] || 0)} ${b[0]},${b[1]}` : `M${a[0]},${a[1]} L${b[0]},${b[1]}`;
  S('path', { d, class: 'edge ' + (o.cls || ''), 'stroke-width': width, 'marker-end': o.arrow === false ? null : `url(#${svg.dataset.marker})` }, g);
  if (o.flow) S('path', { d, class: 'edge flow', 'stroke-width': Math.max(1, width - 1), stroke: 'var(--surface)', opacity: 0.7 }, g);
  // Particles travel along busy edges: more and faster when the edge is thicker (a higher rate).
  if ((o.flow || width >= 3) && o.particles !== false && !calm()) {
    const n = width >= 7 ? 3 : width >= 4 ? 2 : 1, dur = Math.max(0.9, 2.6 - width * 0.16);
    for (let i = 0; i < n; i++) {
      const c = S('circle', { r: Math.min(3.4, 1.6 + width * 0.18), class: 'particle', opacity: 0.9 }, g);
      S('animateMotion', { dur: dur + 's', repeatCount: 'indefinite', begin: (-i * dur / n).toFixed(2) + 's', path: d, keyPoints: '0;1', keyTimes: '0;1', calcMode: 'linear' }, c);
    }
  }
  if (o.label) {
    const [lx, ly] = o.labelAt || [mx + (o.curve ? (o.curve[0] || 0) / 2 : 0), my + (o.curve ? (o.curve[1] || 0) / 2 : 0) - 12];
    T(g, lx, ly, o.label, 'edge-lbl', { 'text-anchor': 'middle' });
  }
  if (o.below) T(g, mx, my + 24, o.below, 'n-sub', { 'text-anchor': 'middle' });
  return g;
}

export const text = (svg, x, y, s, cls = 'n-sub', anchor = 'middle') => T(svg, x, y, s, cls, { 'text-anchor': anchor });

/* Map a rate to an arrow width, so faster flows look heavier. */
export const widthFor = (v, max) => 1.5 + 9 * Math.max(0, Math.min(1, v / (max || 1)));

/* A small inline sparkline inside an illustration. */
export function spark(svg, xs, ys, box, cls = 'st1', yr) {
  const [x0, y0, w, hh] = box;
  const lo = yr ? yr[0] : Math.min(...ys), hi = yr ? yr[1] : Math.max(...ys);
  const X = i => x0 + (xs[i] - xs[0]) / (xs[xs.length - 1] - xs[0] || 1) * w;
  const Y = v => y0 + hh - (v - lo) / (hi - lo || 1) * hh;
  S('path', { d: ys.map((v, i) => (i ? 'L' : 'M') + X(i).toFixed(1) + ',' + Y(v).toFixed(1)).join(''), fill: 'none', class: cls, 'stroke-width': 2, 'stroke-linejoin': 'round' }, svg);
  return { X, Y };
}
