// A small SVG plotting kit: scales, axes, lines, areas, hover tooltips. No dependencies.
const NS = 'http://www.w3.org/2000/svg';

export function E(tag, attrs, parent) {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(n);
  return n;
}

export function niceTicks(a, b, n = 5) {
  if (!(b > a)) return [a];
  const span = b - a, mag = Math.pow(10, Math.floor(Math.log10(span / n)));
  const step = [1, 2, 2.5, 5, 10].map(s => s * mag).find(s => span / s <= n) || mag * 10;
  const out = []; for (let v = Math.ceil(a / step - 1e-9) * step; v <= b + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

export const extent = (...arrs) => {
  let lo = Infinity, hi = -Infinity;
  for (const a of arrs) for (const v of a) if (v != null && Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
  return [lo, hi];
};

/* frame(host, opts): draws axes and grid at the host's real pixel width. */
export function frame(host, o) {
  host.querySelectorAll('svg.plot, .tip').forEach(n => n.remove());
  const w = Math.max(260, host.clientWidth), h = o.h || 300;
  const M = Object.assign({ l: 52, r: 16, t: o.yl ? 30 : 12, b: o.xl ? 40 : 26 }, o.m || {});
  const svg = E('svg', { class: 'plot', width: w, height: h, viewBox: `0 0 ${w} ${h}`, role: 'img', 'aria-label': o.label || '' }, host);
  svg.dataset.base = h - M.b;
  const tx = o.xlog ? Math.log10 : v => v, ty = o.ylog ? Math.log10 : v => v;
  const X = v => M.l + (tx(v) - tx(o.x[0])) / (tx(o.x[1]) - tx(o.x[0])) * (w - M.l - M.r);
  const Y = v => h - M.b - (ty(v) - ty(o.y[0])) / (ty(o.y[1]) - ty(o.y[0])) * (h - M.t - M.b);
  const xInv = px => { const f = (px - M.l) / (w - M.l - M.r); return o.xlog ? Math.pow(10, tx(o.x[0]) + f * (tx(o.x[1]) - tx(o.x[0]))) : o.x[0] + f * (o.x[1] - o.x[0]); };
  const g = E('g', { class: 'grid' }, svg), ax = E('g', { class: 'axis' }, svg);
  const yt = o.yt || niceTicks(o.y[0], o.y[1], o.yn || 5);
  const xt = o.xt || niceTicks(o.x[0], o.x[1], o.xn || Math.max(3, Math.floor(w / 110)));
  yt.forEach(v => { E('line', { x1: M.l, x2: w - M.r, y1: Y(v), y2: Y(v) }, g); E('text', { x: M.l - 8, y: Y(v) + 4, 'text-anchor': 'end' }, ax).textContent = (o.yf || String)(v); });
  xt.forEach(v => { E('text', { x: X(v), y: h - M.b + 16, 'text-anchor': 'middle' }, ax).textContent = (o.xf || String)(v); });
  E('line', { x1: M.l, x2: w - M.r, y1: h - M.b, y2: h - M.b, stroke: 'var(--line-2)' }, svg);
  if (o.xl) E('text', { class: 'lbl', x: (M.l + w - M.r) / 2, y: h - 4, 'text-anchor': 'middle' }, ax).textContent = o.xl;
  if (o.yl) E('text', { class: 'lbl', x: 4, y: 13, 'text-anchor': 'start' }, ax).textContent = o.yl;
  const clip = 'c' + Math.random().toString(36).slice(2);
  const defs = E('defs', {}, svg); const cp = E('clipPath', { id: clip }, defs);
  E('rect', { x: M.l, y: M.t - 2, width: w - M.l - M.r, height: h - M.t - M.b + 4 }, cp);
  const body = E('g', { 'clip-path': `url(#${clip})` }, svg), over = E('g', {}, svg);
  const path = pts => { let d = '', pen = false; for (const p of pts) { if (p[1] == null || !Number.isFinite(p[1])) { pen = false; continue; } d += (pen ? 'L' : 'M') + X(p[0]).toFixed(1) + ',' + Y(p[1]).toFixed(1); pen = true; } return d; };
  return {
    svg, body, over, X, Y, xInv, w, h, M,
    line(pts, cls) { return E('path', { d: path(pts), class: 'ln ' + cls }, body); },
    area(top, bot, cls) { const d = path(top) + bot.slice().reverse().map(p => 'L' + X(p[0]).toFixed(1) + ',' + Y(p[1]).toFixed(1)).join('') + 'Z'; return E('path', { d, class: cls }, body); },
    hline(v, cls = 'ref') { return E('line', { x1: M.l, x2: w - M.r, y1: Y(v), y2: Y(v), class: cls }, body); },
    vline(v, cls = 'ref') { return E('line', { x1: X(v), x2: X(v), y1: M.t, y2: h - M.b, class: cls }, body); },
    band(y0, y1, cls = 'zone') { return E('rect', { x: M.l, width: w - M.l - M.r, y: Y(y1), height: Math.max(0, Y(y0) - Y(y1)), class: cls }, body); },
    dot(x, y, cls, r = 4.5) { return E('circle', { cx: X(x), cy: Y(y), r, class: cls + ' ring' }, body); },
    bars(edges, counts, cls, base) { const b0 = base ?? o.y[0]; counts.forEach((c, i) => { if (!c) return; const x0 = X(edges[i]) + 1, x1 = X(edges[i + 1]) - 1; E('rect', { x: x0, y: Y(c), width: Math.max(1, x1 - x0), height: Math.max(0, Y(b0) - Y(c)), rx: 1.5, class: cls }, body); }); },
    text(x, y, s, attrs = {}) { const t = E('text', Object.assign({ x, y, class: 'dlabel' }, attrs), over); t.textContent = s; return t; },
  };
}

/* decorate(host, {animate}): after a chart is drawn, give its main series a glow fill that
   fades to the axis and, on the first draw, animate the lines in. Only charts with one or two
   main lines get fills, so busy charts stay readable. Purely visual. */
const SERIES = ['s1', 's2', 's3', 's4'];
export function decorate(host, { animate = false } = {}) {
  const calm = matchMedia('(prefers-reduced-motion: reduce)').matches;
  host.querySelectorAll('svg.plot').forEach(svg => {
    const base = +svg.dataset.base;
    const main = [...svg.querySelectorAll('path.ln')].filter(p => !/\b(thin|dashed|sn)\b/.test(p.getAttribute('class')) && SERIES.some(c => p.classList.contains(c)));
    if (main.length && main.length <= 2 && Number.isFinite(base)) {
      const defs = svg.querySelector('defs') || E('defs', {}, svg);
      main.forEach((p, i) => {
        const d = p.getAttribute('d') || '';
        if ((d.match(/M/g) || []).length !== 1) return;
        const pts = d.slice(1).split('L').map(xy => xy.split(',').map(Number));
        if (pts.length < 3) return;
        const col = SERIES.find(c => p.classList.contains(c)), id = `ga-${col}-${Math.random().toString(36).slice(2, 8)}`;
        const g = E('linearGradient', { id, x1: 0, y1: 0, x2: 0, y2: 1 }, defs);
        E('stop', { offset: 0, style: `stop-color:var(--${col});stop-opacity:${i ? 0.16 : 0.26}` }, g);
        E('stop', { offset: 1, style: `stop-color:var(--${col});stop-opacity:0` }, g);
        const area = E('path', { d: `${d}L${pts[pts.length - 1][0]},${base}L${pts[0][0]},${base}Z`, fill: `url(#${id})`, class: 'glow-area' + (animate && !calm ? ' draw' : '') }, p.parentNode);
        p.parentNode.insertBefore(area, p.parentNode.firstChild);
      });
    }
    if (animate && !calm) svg.querySelectorAll('path.ln').forEach(p => { if (!p.classList.contains('dashed')) { p.setAttribute('pathLength', 1); p.classList.add('draw'); } });
  });
}

/* hover(host, P, fn): fn(px, py) -> {html, cross (px), pts: [[px, py, cls]]} or null */
export function hover(host, P, fn, { onClick } = {}) {
  const tip = document.createElement('div'); tip.className = 'tip'; tip.hidden = true; host.appendChild(tip);
  const layer = E('g', {}, P.svg);
  const hit = E('rect', { class: 'hit', x: P.M.l, y: P.M.t, width: P.w - P.M.l - P.M.r, height: P.h - P.M.t - P.M.b }, P.svg);
  const at = ev => { const r = P.svg.getBoundingClientRect(); return [ev.clientX - r.left, ev.clientY - r.top]; };
  const move = ev => {
    const [px, py] = at(ev), res = fn(px, py);
    layer.innerHTML = '';
    if (!res) { tip.hidden = true; return; }
    if (res.cross != null) E('line', { class: 'xh', x1: res.cross, x2: res.cross, y1: P.M.t, y2: P.h - P.M.b }, layer);
    (res.pts || []).forEach(([x, y, cls]) => Number.isFinite(y) && E('circle', { cx: x, cy: y, r: 4.5, class: 'ring ' + cls }, layer));
    tip.innerHTML = res.html; tip.hidden = false;
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    let lx = px + 14; if (lx + tw > P.w) lx = px - tw - 14;
    let ly = py - th - 10; if (ly < 0) ly = py + 16;
    tip.style.left = Math.max(0, lx) + 'px'; tip.style.top = ly + 'px';
  };
  hit.addEventListener('pointermove', move);
  hit.addEventListener('pointerdown', move);
  hit.addEventListener('pointerleave', () => { tip.hidden = true; layer.innerHTML = ''; });
  if (onClick) { hit.style.cursor = 'pointer'; hit.addEventListener('click', ev => onClick(...at(ev))); }
  return hit;
}

/* Index of the x value nearest to the pointer, for crosshair tooltips. */
export function nearest(xs, v) {
  let lo = 0, hi = xs.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] < v) lo = m; else hi = m; }
  return Math.abs(xs[lo] - v) <= Math.abs(xs[hi] - v) ? lo : hi;
}

/* Re-run draw(host) whenever the host is resized to a new width (and is visible). */
const registry = new Set();
export function responsive(host, draw) {
  let lastW = 0;
  const run = () => { if (host.clientWidth > 0) { lastW = host.clientWidth; draw(host); } };
  const ro = new ResizeObserver(() => { if (host.clientWidth > 0 && host.clientWidth !== lastW) run(); });
  ro.observe(host);
  registry.add(run);
  run.dispose = () => { ro.disconnect(); registry.delete(run); };
  return run;
}
export const redrawAll = () => registry.forEach(r => r());

export const zip = (xs, ys) => xs.map((x, i) => [x, ys[i]]);
