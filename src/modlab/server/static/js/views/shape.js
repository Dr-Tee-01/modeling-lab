// Pillar 2 · Shape: the five ShapeLab modules (IT575 labs 1–5) as workbench specs.
import { call, fmt, int, pct } from '../api.js';
import { E, extent, frame, hover, nearest, zip } from '../plot.js';
import { S, edge, node, spark, text } from '../ui/illustrate.js';
import { categorical, diverging } from '../viewer3d.js';

const tip = (title, rows) => `<b>${title}</b>` + rows.map(([k, v]) => `<br>${k}: <span class="n">${v}</span>`).join('');

/* ---------------------------------------------------------------- shared helpers */

// Module-level cache for large or fixed payloads (meshes, the jaw model, curvature fields), so
// switching views or toggling a display option never refetches them. No abort signal is passed:
// a newer request may be waiting on the same promise.
const store = new Map();
function cached(key, fn) {
  const hit = store.has(key);
  if (!hit) store.set(key, fn().catch(e => { store.delete(key); throw e; }));
  return [store.get(key), hit];
}
const meshData = name => cached('mesh:' + name, () => call(`/api/shape/mesh/${name}`))[0];

const MESH = { cat4: { label: 'Cat', v: 3400 }, bunny: { label: 'Bunny', v: 14290 }, cactus: { label: 'Cactus', v: 5261 }, cylinder: { label: 'Cylinder', v: 4802 } };
const BASIS = { cat4: 120, bunny: 100 };
const OUTLINES = { bird1: { label: 'Bird 1', n: 843 }, bird2: { label: 'Bird 2', n: 793 } };
const NEG = 'rgb(41,120,214)', MID = 'rgb(150,155,165)', POS = 'rgb(227,74,71)';

/* Hand a mesh to the 3D viewer: faces (and a refit) only when the mesh itself changed. */
function put(v, ctx, slot, key, vertices, faces, colors) {
  const fresh = ctx.extra[slot] !== key;
  v.set(vertices, fresh ? faces : null, { colors, refit: fresh });
  ctx.extra[slot] = key;
}

const rgbOf = (arr, i) => `rgb(${Math.round(arr[i * 3] * 255)},${Math.round(arr[i * 3 + 1] * 255)},${Math.round(arr[i * 3 + 2] * 255)})`;
const partColours = n => { const c = categorical([...Array(n).keys()]); return [...Array(n).keys()].map(i => rgbOf(c, i)); };

/* Equal-aspect 2D frame for outlines and jawlines (image coordinates: y grows downward). */
function equalFrame(host, pts, h) {
  const [x0, x1] = extent(pts.map(p => p[0])), [y0, y1] = extent(pts.map(p => p[1]));
  const w = Math.max(260, host.clientWidth), pw = w - 32, ph = h - 32;
  const k = Math.min(pw / (x1 - x0 || 1), ph / (y1 - y0 || 1)) * 0.95, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  return frame(host, { h, x: [cx - pw / 2 / k, cx + pw / 2 / k], y: [cy + ph / 2 / k, cy - ph / 2 / k], xt: [], yt: [], m: { l: 16, r: 16, t: 16, b: 16 } });
}
function nearestPt(P, pts, px, py) {
  let best = 0, bd = Infinity;
  pts.forEach((p, i) => { const q = (P.X(p[0]) - px) ** 2 + (P.Y(p[1]) - py) ** 2; if (q < bd) { bd = q; best = i; } });
  return best;
}

/* Illustration helpers: a titled panel with room for a drawing, and a small equal-aspect shape. */
function panel(svg, { x, y, w, h, title, lines = [], hot }) {
  S('rect', { x, y, width: w, height: h, rx: 10, class: 'node' + (hot ? ' hot' : '') }, svg);
  text(svg, x + w / 2, y + 20, title, 'n-title');
  lines.forEach((l, i) => text(svg, x + w / 2, y + 38 + i * 15, l, 'n-sub'));
  const top = y + 30 + lines.length * 15;
  return { x, y, w, h, cx: x + w / 2, cy: y + h / 2, r: x + w, b: y + h, inner: [x + 10, top, w - 20, y + h - top - 10] };
}
function mini(svg, sets, box) {
  const all = sets.flatMap(s => s.pts);
  const [x0, x1] = extent(all.map(p => p[0])), [y0, y1] = extent(all.map(p => p[1]));
  const [bx, by, bw, bh] = box, k = Math.min(bw / (x1 - x0 || 1), bh / (y1 - y0 || 1));
  const ox = bx + (bw - (x1 - x0) * k) / 2, oy = by + (bh - (y1 - y0) * k) / 2;
  const X = v => (ox + (v - x0) * k).toFixed(1), Y = v => (oy + (v - y0) * k).toFixed(1);
  for (const s of sets) {
    if (s.dots) { s.pts.forEach(p => S('circle', { cx: X(p[0]), cy: Y(p[1]), r: s.r || 1.8, class: s.cls }, svg)); continue; }
    S('path', { d: 'M' + s.pts.map(p => X(p[0]) + ',' + Y(p[1])).join('L') + (s.closed ? 'Z' : ''), fill: 'none', class: s.cls, style: s.style,
      'stroke-width': s.width || 1.6, 'stroke-dasharray': s.dash, opacity: s.opacity, 'stroke-linejoin': 'round' }, svg);
  }
}

/* An error-against-count curve with a 10% line, a marker at the current choice and click-to-set. */
function errorCurve(host, P, xs, ys, cur, onPick, rows) {
  P.hline(0.1); P.text(P.w - P.M.r - 4, P.Y(0.1) - 5, '10% error', { 'text-anchor': 'end' });
  P.line(zip(xs, ys), 's1');
  const i0 = xs.indexOf(cur);
  if (i0 >= 0) { P.vline(cur, 'ln s2 thin'); P.dot(cur, ys[i0], 'f2', 5); }
  hover(host, P, px => { const i = nearest(xs, P.xInv(px)); return { cross: P.X(xs[i]), pts: [[P.X(xs[i]), P.Y(ys[i]), 'f1']], html: rows(i) + '<br><span class="n">click to choose</span>' }; },
    { onClick: px => onPick(xs[nearest(xs, P.xInv(px))]) });
}

/* ================================================================ L1: Fourier descriptors */
const fourier = {
  id: 'fourier', group: 'shape', nav: 'Fourier descriptors', tag: 'IT575 L1', title: 'Fourier descriptors: an outline from a handful of numbers',
  question: 'A bird outline has about 800 points. How few Fourier coefficients still draw a recognisable bird?',
  params: [
    { k: 'outline', type: 'seg', label: 'Outline', value: 'bird1', options: [['bird1', 'Bird 1'], ['bird2', 'Bird 2']], group: 'Outline',
      help: 'Two bird silhouettes, each traced as a closed curve: Bird 1 has 843 points, Bird 2 has 793.' },
    { k: 'keep', label: 'Coefficients kept K', min: 1, max: 200, step: 1, value: 6, group: 'Reconstruction',
      help: 'How many Fourier coefficients to keep, lowest frequencies first. The rest are set to zero before the inverse FFT.' },
  ],
  presets: [
    { label: '6 coefficients (10% error)', values: { outline: 'bird1', keep: 6 } },
    { label: 'Rough sketch (3)', values: { outline: 'bird1', keep: 3 } },
    { label: 'Fine detail (60)', values: { outline: 'bird1', keep: 60 } },
    { label: 'Bird 2 at 10%', values: { outline: 'bird2', keep: 5 } },
  ],
  debounce: 60,
  request: (s, sig) => call('/api/shape/fourier', { outline: s.outline, keep: s.keep }, { signal: sig }),
  kpis: d => [
    { label: 'Coefficients kept', value: d.keep, unit: `of ${d.n_points}`, note: `${pct(d.share, 2)} of the spectrum` },
    { label: 'Relative error', value: pct(d.error, 1), tone: d.error <= 0.1 ? 'good' : d.error <= 0.25 ? 'warn' : 'bad', note: '‖b − b̂‖ / ‖b‖ over all points' },
    { label: 'Needed for 10% error', value: d.coefficients_for_10pct, unit: 'coefficients', note: `${pct(d.coefficients_for_10pct / d.n_points, 1)} of ${d.n_points}` },
    { label: 'Compression', value: fmt(d.n_points / d.keep, 0) + '×', note: `${d.n_points} points stored as ${d.keep} complex numbers` },
  ],
  illustration: {
    title: 'From outline to a few frequencies and back', height: 200,
    caption: 'Bars are the size of each coefficient on a log scale, lowest frequency first. Orange bars are kept; the error updates as you move K.',
    draw(svg, d, s) {
      const same = d && d.outline === s.outline, N = same ? d.n_points : OUTLINES[s.outline].n;
      const ec = same ? d.error_curve : null, err = ec ? ec[Math.min(s.keep, ec.length - 1)] : null;
      const a = panel(svg, { x: 6, y: 12, w: 110, h: 176, title: 'Outline', lines: [`${N} points`] });
      if (same) mini(svg, [{ pts: d.original, cls: 'st1', closed: true }], a.inner);
      const z = node(svg, { x: 136, y: 66, w: 88, h: 68, title: 'x + iy', sub: 'then FFT' });
      edge(svg, [a.r + 3, z.cy], [z.x - 5, z.cy], { width: 2 });
      const sp = panel(svg, { x: 250, y: 12, w: 166, h: 176, title: 'Keep K lowest', lines: [`K = ${s.keep} of ${N}`], hot: true });
      edge(svg, [z.r + 3, z.cy], [sp.x - 5, z.cy], { width: 2 });
      const nb = 30, [bx, by, bw, bh] = sp.inner, cw = bw / nb;
      const amp = i => ec ? Math.sqrt(Math.max(0, ec[i] ** 2 - ec[i + 1] ** 2)) : 1 / (i + 1);
      for (let i = 0; i < nb; i++) {
        const f = Math.max(0.04, Math.min(1, (Math.log10(Math.max(amp(i), 1e-4)) + 4) / 4)), kept = i < s.keep;
        S('rect', { x: bx + i * cw + 0.6, y: by + bh * (1 - f), width: Math.max(1, cw - 1.2), height: bh * f, rx: 1, class: kept ? 'fill2' : 'fill1', opacity: kept ? 0.9 : 0.25 }, svg);
      }
      if (s.keep < nb) S('line', { x1: bx + s.keep * cw, x2: bx + s.keep * cw, y1: by - 2, y2: by + bh, class: 'ln-m', 'stroke-dasharray': '3 3' }, svg);
      const rc = panel(svg, { x: 444, y: 12, w: 150, h: 176, title: 'Reconstruction', lines: ['inverse FFT', err != null ? `error ${pct(err, 1)}` : 'error …'] });
      edge(svg, [sp.r + 3, z.cy], [rc.x - 5, z.cy], { width: 2 });
      if (same) mini(svg, [{ pts: d.original, style: 'stroke:var(--neutral)', closed: true, width: 1, opacity: 0.6 },
        ...(d.keep === s.keep ? [{ pts: d.reconstruction, cls: 'st2', closed: true, width: 2 }] : [])], rc.inner);
    },
  },
  chartLayout: 'two',
  charts: [
    { id: 'outline', title: 'Original and reconstruction', height: 300,
      sub: d => `${d.keep} of ${d.n_points} coefficients, ${pct(d.error, 1)} error. Hover to compare points.`,
      legend: [['s1', 'line', 'Original'], ['s2', 'line', 'Reconstruction']],
      draw(host, d) {
        const P = equalFrame(host, d.original.concat(d.reconstruction), 300);
        P.line(d.original.concat([d.original[0]]), 's1 thin');
        P.line(d.reconstruction.concat([d.reconstruction[0]]), 's2');
        hover(host, P, (px, py) => {
          const i = nearestPt(P, d.original, px, py), o = d.original[i], r = d.reconstruction[i];
          return { pts: [[P.X(o[0]), P.Y(o[1]), 'f1'], [P.X(r[0]), P.Y(r[1]), 'f2']],
            html: tip(`Point ${i + 1} of ${d.n_points}`, [['Original', `(${fmt(o[0], 1)}, ${fmt(o[1], 1)})`], ['Rebuilt', `(${fmt(r[0], 1)}, ${fmt(r[1], 1)})`], ['Gap', fmt(Math.hypot(o[0] - r[0], o[1] - r[1]), 1) + ' px']]) };
        });
      } },
    { id: 'error', title: 'Error against coefficients kept', height: 300,
      sub: 'One FFT gives the error of every K (Parseval). Click the curve to choose K.',
      legend: [['s1', 'line', 'Relative error'], ['s2', 'line', 'Current K']],
      draw(host, d, s, ctx) {
        const xs = d.error_curve.map((_, i) => i).slice(1), ys = d.error_curve.slice(1), N = xs[xs.length - 1];
        const P = frame(host, { h: 300, x: [1, N], xlog: true, y: [0, Math.min(1, Math.max(...ys) * 1.05)], xt: [1, 2, 5, 10, 20, 50, 100, 200].filter(v => v <= N),
          yf: v => Math.round(v * 100) + '%', xl: 'Coefficients kept K (log scale)', yl: 'Relative error' });
        errorCurve(host, P, xs, ys, d.keep, k => ctx.set({ keep: Math.max(1, k) }, { immediate: true }),
          i => tip(`K = ${xs[i]}`, [['Error', pct(ys[i], 2)], ['Share kept', pct(xs[i] / d.n_points, 2)]]));
      } },
  ],
  insights: d => {
    const ec = d.error_curve, K = d.keep, K2 = Math.min(2 * K, ec.length - 1), c10 = d.coefficients_for_10pct;
    return [
      `${OUTLINES[d.outline].label} needs <b>${c10} coefficients</b> for 10% error: ${pct(c10 / d.n_points, 1)} of its ${d.n_points}. The overall shape lives in the lowest frequencies.`,
      K2 > K ? `Doubling K from ${K} to ${K2} takes the error from <b>${pct(ec[K], 1)}</b> to <b>${pct(ec[K2], 1)}</b>. Sharp corners need many high frequencies, so each extra coefficient buys less.`
        : `At K = ${K} the error is <b>${pct(ec[K], 2)}</b>. What is left is the jagged pixel edge of the traced outline.`,
      'K = 1 keeps only the average point, the centre of the outline. The next two coefficients fit an ellipse; detail comes after that.',
    ];
  },
  maths: { eq: 'zₙ = xₙ + i yₙ,   n = 0 … N−1\nZₖ = Σₙ zₙ e^(−2πi kn/N)\nẑₙ = (1/N) Σ_{k kept} Zₖ e^(2πi kn/N)\n‖z − ẑ‖² = (1/N) Σ_{k dropped} |Zₖ|²   (Parseval)',
    notes: ['Coefficients are kept in order of |frequency|: 0, +1, −1, +2, −2, … K counts them one by one.', 'The relative error divides by ‖z‖, the size of the original outline.'] },
};

/* ================================================================ L2: statistical shape model */
const jawModel = () => cached('jaw-model', () => call('/api/shape/jaw-model', { n_modes: 6 }))[0];
const jawShape = (m, b) => m.mean.map((p, j) => [0, 1].map(c => p[c] + b.reduce((acc, bi, i) => acc + bi * m.sd[i] * m.modes[i][j][c], 0)));
const sd = v => (v > 0 ? '+' : v < 0 ? '−' : '') + fmt(Math.abs(v), 1);

const jaw = {
  id: 'jaw', group: 'shape', nav: 'Shape model', tag: 'IT575 L2', title: 'A statistical shape model of 300 jawlines',
  question: 'Every jaw is 41 landmarks, 82 numbers. How many numbers does it take to describe how jaws really vary?',
  params: [
    { k: 'b1', label: 'Mode 1 b₁', min: -3, max: 3, step: 0.1, value: 0, unit: 'sd', group: 'Move along each mode', help: 'The largest pattern of variation across the 300 jaws: 54.9% of it.' },
    { k: 'b2', label: 'Mode 2 b₂', min: -3, max: 3, step: 0.1, value: 0, unit: 'sd', group: 'Move along each mode', help: 'The second pattern, independent of mode 1: 19.6% of the variation.' },
    { k: 'b3', label: 'Mode 3 b₃', min: -3, max: 3, step: 0.1, value: 0, unit: 'sd', group: 'Move along each mode', help: 'The third pattern: 16.0% of the variation. Modes 4 and up add little.' },
  ],
  presets: [
    { label: 'Mean jaw', values: { b1: 0, b2: 0, b3: 0 } },
    { label: 'Mode 1 at +2 sd', values: { b1: 2, b2: 0, b3: 0 } },
    { label: 'Mode 2 at +2 sd', values: { b1: 0, b2: 2, b3: 0 } },
    { label: 'Mode 3 at −2 sd', values: { b1: 0, b2: 0, b3: -2 } },
  ],
  debounce: 40,
  request: async (s, sig) => {
    const model = jawModel();
    const smp = await call('/api/shape/jaw-sample', { b1: s.b1, b2: s.b2, b3: s.b3 }, { signal: sig });
    return { ...smp, model: await model };
  },
  kpis: d => {
    const m = d.model, b = d.b.slice(0, 3), dist = Math.hypot(...b);
    const [x0, x1] = extent(m.mean.map(p => p[0]));
    const shifts = d.shape.map((p, j) => Math.hypot(p[0] - m.mean[j][0], p[1] - m.mean[j][1])), j = shifts.indexOf(Math.max(...shifts));
    return [
      { label: 'Training jaws', value: int(m.n_shapes), note: `${m.n_landmarks} landmarks each` },
      { label: 'Variance in 3 modes', value: pct(m.cumulative[2], 1), tone: 'good', note: `mode 1 alone ${pct(m.variance_explained[0], 1)}` },
      { label: 'Distance from the mean', value: fmt(dist, 2), unit: 'sd', tone: dist <= 2 ? 'good' : dist <= 3 ? 'warn' : 'bad', note: '√(b₁² + b₂² + b₃²)' },
      { label: 'Largest landmark shift', value: pct(shifts[j] / (x1 - x0), 1), note: `of the jaw width, at landmark ${j + 1}` },
    ];
  },
  illustration: {
    title: 'From 300 drawings to three numbers', height: 210,
    caption: 'The model shape on the right is rebuilt in the browser as you drag, from the mean and the mode vectors the server sent.',
    draw(svg, d, s) {
      const m = d?.model, b = [s.b1, s.b2, s.b3];
      const p1 = panel(svg, { x: 6, y: 10, w: 132, h: 190, title: 'Landmarks', lines: ['300 jaws', '41 points each'] });
      const p2 = panel(svg, { x: 160, y: 10, w: 132, h: 190, title: 'Procrustes', lines: ['remove position,', 'size and rotation'] });
      const p3 = panel(svg, { x: 314, y: 10, w: 132, h: 190, title: 'PCA', lines: ['modes of', 'variation'] });
      const p4 = panel(svg, { x: 468, y: 10, w: 126, h: 190, title: 'Model shape', lines: [`b = ${sd(s.b1)}, ${sd(s.b2)},`, `${sd(s.b3)} sd`], hot: true });
      [[p1, p2], [p2, p3], [p3, p4]].forEach(([a, c]) => edge(svg, [a.r + 3, a.y + 40], [c.x - 5, a.y + 40], { width: 2 }));
      if (!m) return;
      mini(svg, [{ pts: m.mean, cls: 'fill1', dots: true, r: 2 }], p1.inner);
      mini(svg, [...m.aligned_sample.map(j => ({ pts: j, style: 'stroke:var(--neutral)', width: 1, opacity: 0.45 })), { pts: m.mean, style: 'stroke:var(--ink-2)', width: 1.5, dash: '3 3' }], p2.inner);
      const [bx, by, bw, bh] = p3.inner, v = m.variance_explained, cw = bw / v.length;
      v.forEach((x, i) => S('rect', { x: bx + i * cw + 3, y: by + bh - 8 - (bh - 22) * x, width: cw - 6, height: (bh - 22) * x, rx: 2, class: i < 3 ? 'fill2' : 'fill1', opacity: i < 3 ? 0.9 : 0.35 }, svg));
      text(svg, bx + bw / 2, by + 10, `3 modes: ${pct(m.cumulative[2], 1)}`, 'n-sub');
      const shape = jawShape(m, b);
      mini(svg, [{ pts: m.mean, style: 'stroke:var(--neutral)', width: 1.2, dash: '3 3' }, { pts: shape, cls: 'st2', width: 2.2 }], p4.inner);
    },
  },
  chartLayout: 'two',
  charts: [
    { id: 'shape', title: 'Model jaw against the training set', height: 300,
      sub: d => `24 of the ${d.model.n_shapes} aligned jaws in grey, the mean dashed. Hover a landmark to see how far it moved.`,
      legend: [['neutral', 'line', 'Aligned jaws'], ['neutral', 'dash', 'Mean'], ['s2', 'line', 'Model shape']],
      draw(host, d) {
        const m = d.model, P = equalFrame(host, m.aligned_sample.flat().concat(d.shape), 300);
        m.aligned_sample.forEach(j => P.line(j, 'sn thin').setAttribute('opacity', '.3'));
        P.line(m.mean, 'sn dashed'); P.line(d.shape, 's2');
        d.shape.forEach(p => P.dot(p[0], p[1], 'f2', 2.5));
        hover(host, P, (px, py) => {
          const j = nearestPt(P, d.shape, px, py), p = d.shape[j], q = m.mean[j];
          return { pts: [[P.X(p[0]), P.Y(p[1]), 'f2'], [P.X(q[0]), P.Y(q[1]), 'fn']], html: tip(`Landmark ${j + 1}`, [['Moved from the mean', fmt(Math.hypot(p[0] - q[0], p[1] - q[1]), 4)], ['Units', 'unit-size jaw']]) };
        });
      } },
    { id: 'variance', title: 'Variance explained by each mode', height: 300,
      sub: 'Bars: one mode each. Line: running total. The sliders move along modes 1 to 3.',
      legend: [['s1', 'box', 'One mode'], ['s2', 'line', 'Running total']],
      draw(host, d) {
        const m = d.model, v = m.variance_explained, idx = v.map((_, i) => i + 1);
        const P = frame(host, { h: 300, x: [0.5, v.length + 0.5], y: [0, 1], xt: idx, yf: x => Math.round(x * 100) + '%', xl: 'Mode', yl: 'Variance explained' });
        idx.forEach((i, k) => P.bars([i - 0.32, i + 0.32], [v[k]], k < 3 ? 'f1' : 'fn', 0));
        P.line(zip(idx, m.cumulative), 's2'); idx.forEach(i => P.dot(i, m.cumulative[i - 1], 'f2', 3.5));
        hover(host, P, px => { const k = Math.max(0, Math.min(v.length - 1, Math.round(P.xInv(px)) - 1));
          return { cross: P.X(k + 1), pts: [[P.X(k + 1), P.Y(m.cumulative[k]), 'f2']], html: tip(`Mode ${k + 1}`, [['Explains', pct(v[k], 1)], ['Running total', pct(m.cumulative[k], 1)], ['1 sd', fmt(m.sd[k], 4)]]) }; });
      } },
  ],
  insights: d => {
    const m = d.model, dist = Math.hypot(...d.b.slice(0, 3));
    return [
      `Three modes explain <b>${pct(m.cumulative[2], 1)}</b> of the variation in ${m.n_shapes} jawlines, so ${m.n_landmarks * 2} coordinates per jaw come down to 3 numbers.`,
      dist > 3 ? `This jaw is <b>${fmt(dist, 1)} sd</b> from the mean, further out than almost any real jaw, so it can look implausible.`
        : 'Any setting within ±3 sd gives a jaw-like shape: the model can only move along directions the 300 jaws actually varied in.',
      'Procrustes alignment comes first. Without it, PCA would spend its first modes on where the jaw was drawn and how big it was.',
    ];
  },
  maths: { eq: 'x = x̄ + Σᵢ bᵢ √λᵢ φᵢ        bᵢ in standard deviations\nC = (1/n) Σⱼ (xⱼ − x̄)(xⱼ − x̄)ᵀ = Φ Λ Φᵀ\nshare of mode i = λᵢ / Σ λ',
    notes: ['Generalised Procrustes: centre each jaw, scale it to unit size, rotate it onto the current mean, average, and repeat until the mean stops moving.', 'Variance shares are relative to the first 10 modes.'] },
};

/* ================================================================ L4: spectral analysis */
const spectral = {
  id: 'spectral', group: 'shape', nav: 'Spectral analysis', tag: 'IT575 L4', title: 'Laplace–Beltrami spectrum: a Fourier basis for surfaces',
  question: 'Fourier descriptors work on a curve. What plays the role of sines and cosines on a 3D surface, and how many does a cat need?',
  params: [
    { k: 'mesh', type: 'seg', label: 'Mesh', value: 'cat4', options: [['cat4', 'Cat'], ['bunny', 'Bunny']], group: 'Mesh',
      help: 'Cat: 3,400 vertices, open at the base. Bunny: 14,290 vertices, closed. The bunny uses 100 eigenfunctions to keep its first solve near one second.' },
    { k: 'view', type: 'seg', label: 'Task', value: 'smooth', options: [['smooth', 'Smoothing'], ['segment', 'Segmentation']], group: 'Task',
      help: 'Smoothing rebuilds the mesh from its first k eigenfunctions. Segmentation clusters vertices by their eigenfunction values.' },
    { k: 'k', label: 'Eigenfunctions k', min: 1, max: 120, step: 1, value: 30, group: 'Task', show: s => s.view === 'smooth',
      help: 'How many of the lowest-frequency eigenfunctions to keep. The bunny stops at 100.' },
    { k: 'parts', label: 'Parts', min: 2, max: 10, step: 1, value: 5, group: 'Task', show: s => s.view === 'segment',
      help: 'Number of clusters for k-means on the spectral embedding.' },
  ],
  presets: [
    { label: 'Blob (k=5)', values: { view: 'smooth', k: 5 } },
    { label: 'Recognisable (k=15)', values: { view: 'smooth', k: 15 } },
    { label: 'Detailed (k=60)', values: { view: 'smooth', k: 60 } },
    { label: 'Segment into 5 parts', values: { view: 'segment', parts: 5 } },
  ],
  debounce: 60,
  request: async (s, sig) => {
    const basis = BASIS[s.mesh], mesh = meshData(s.mesh);
    const r = s.view === 'segment'
      ? await call('/api/shape/segment', { mesh: s.mesh, parts: s.parts, basis }, { signal: sig })
      : await call('/api/shape/spectral', { mesh: s.mesh, k: Math.min(s.k, basis), basis }, { signal: sig });
    return Object.assign(r, { mesh_data: await mesh });
  },
  kpis: d => {
    const n = d.mesh_data.n_vertices;
    if (d.labels) {
      const lo = Math.min(...d.sizes);
      return [
        { label: 'Parts', value: d.parts, note: 'k-means++, best of 10 starts' },
        { label: 'Largest part', value: pct(d.sizes[0] / n, 0), note: `${int(d.sizes[0])} of ${int(n)} vertices` },
        { label: 'Smallest part', value: pct(lo / n, 1), tone: lo / n < 0.01 ? 'warn' : null, note: `${int(lo)} vertices` },
        { label: 'Embedding', value: d.gps_dims, unit: 'dimensions', note: 'φ₁ … φ₂₀, each scaled by 1/√λ' },
      ];
    }
    const c10 = d.coefficients_for_10pct;
    return [
      { label: 'Eigenfunctions used', value: d.k, unit: `of ${d.basis}`, note: 'basis solved once per mesh, then reused' },
      { label: 'Relative error', value: pct(d.error, 1), tone: d.error <= 0.1 ? 'good' : d.error <= 0.2 ? 'warn' : 'bad', note: 'area-weighted, vs the centred mesh' },
      { label: 'Needed for 10% error', value: c10 ?? `> ${d.basis}`, unit: c10 ? 'eigenfunctions' : '', note: `${MESH[d.mesh].label}, ${int(n)} vertices` },
      { label: 'Numbers kept', value: pct(d.k / n, 2), note: `${d.k} × 3 coefficients instead of ${int(n)} × 3 coordinates` },
    ];
  },
  illustration: {
    title: 'Eigenfunctions of the mesh, then a projection', height: 210,
    caption: 'The eigenproblem is the expensive step and runs once per mesh. Changing k or the number of parts reuses it.',
    draw(svg, d, s) {
      const same = d && d.mesh === s.mesh, seg = s.view === 'segment', basis = BASIS[s.mesh], k = Math.min(s.k, basis);
      const ec = same && d.error_curve ? d.error_curve : null, err = ec ? ec[k - 1] : null;
      const n0 = node(svg, { x: 4, y: 22, w: 96, h: 80, title: MESH[s.mesh].label, value: int(MESH[s.mesh].v), sub: 'vertices' });
      const n1 = node(svg, { x: 126, y: 22, w: 96, h: 80, title: 'Operators', value: 'L, M', sub: 'cotan, area' });
      const n2 = node(svg, { x: 248, y: 22, w: 96, h: 80, title: 'Eigenpairs', value: String(basis), sub: '−Lφ = λMφ' });
      const n3 = seg ? node(svg, { x: 370, y: 22, w: 96, h: 80, title: 'Embed', value: '20 dims', sub: 'φᵢ / √λᵢ', hot: true })
        : node(svg, { x: 370, y: 22, w: 96, h: 80, title: 'Project', value: `k = ${k}`, sub: 'first k', hot: true });
      const n4 = seg ? node(svg, { x: 492, y: 22, w: 96, h: 80, title: 'k-means', value: `${s.parts} parts` })
        : node(svg, { x: 492, y: 22, w: 96, h: 80, title: 'Rebuilt', value: err != null ? pct(err, 1) : '…', sub: 'error' });
      [n0, n1, n2, n3, n4].reduce((a, b) => { edge(svg, [a.r + 3, a.cy], [b.x - 5, b.cy], { width: 2 }); return b; });
      text(svg, n2.cx, n2.b + 16, 'solved once', 'n-sub');
      if (seg) {
        if (same && d.sizes) {
          const tot = d.sizes.reduce((a, b) => a + b, 0), col = partColours(d.sizes.length);
          let x = 126;
          d.sizes.forEach((c, i) => { const w = 462 * c / tot; S('rect', { x: x + 0.5, y: 146, width: Math.max(1, w - 1), height: 24, rx: 3, style: `fill:${col[i]}` }, svg); x += w; });
          text(svg, 116, 163, 'part sizes', 'n-sub', 'end');
        } else text(svg, 357, 165, 'clustering the spectral embedding…', 'n-sub');
      } else if (ec) {
        const xs = ec.map((_, i) => i + 1), lo = 0, hi = Math.min(1, Math.max(...ec)), box = [126, 136, 462, 58];
        const sp = spark(svg, xs, ec, box, 'st1', [lo, hi]);
        S('line', { x1: box[0], x2: box[0] + box[2], y1: sp.Y(0.1), y2: sp.Y(0.1), class: 'ln-m', 'stroke-dasharray': '3 3' }, svg);
        S('circle', { cx: sp.X(k - 1), cy: sp.Y(ec[k - 1]), r: 5, class: 'fill2' }, svg);
        text(svg, 116, 160, 'error vs k', 'n-sub', 'end');
        text(svg, box[0] + box[2], sp.Y(0.1) - 5, '10%', 'n-sub', 'end');
      } else text(svg, 357, 165, same ? '' : 'solving the eigenproblem for this mesh…', 'n-sub');
    },
  },
  chartLayout: 'two',
  charts: [
    { id: 'mesh', kind: '3d', title: 'The mesh, rebuilt or segmented',
      sub: d => d.labels ? `${d.parts} parts from k-means on the first ${d.gps_dims} eigenfunctions. Colours match the part-size bars.`
        : `Vertex positions projected on the first ${d.k} of ${d.basis} eigenfunctions.`,
      draw(v, d, s, ctx) {
        const m = d.mesh_data;
        if (d.labels) put(v, ctx, 'spec3d', d.mesh, m.vertices, m.faces, categorical(d.labels));
        else put(v, ctx, 'spec3d', d.mesh, d.vertices, m.faces, null);
      } },
    { id: 'error', title: 'Error against eigenfunctions used', height: 300, show: d => !d.labels,
      sub: 'Low eigenvalues are smooth, global shapes; high ones add detail. Click the curve to choose k.',
      legend: [['s1', 'line', 'Relative error'], ['s2', 'line', 'Current k']],
      draw(host, d, s, ctx) {
        const ys = d.error_curve, xs = ys.map((_, i) => i + 1);
        const P = frame(host, { h: 300, x: [1, xs.length], y: [0, Math.min(1, Math.max(...ys) * 1.05)], yf: v => Math.round(v * 100) + '%', xl: 'Eigenfunctions used k', yl: 'Relative error' });
        errorCurve(host, P, xs, ys, d.k, k => ctx.set({ k }, { immediate: true }),
          i => tip(`k = ${xs[i]}`, [['Error', pct(ys[i], 2)], ['Eigenvalue λ', fmt(d.eigenvalues[i], 4)]]));
      } },
    { id: 'parts', title: 'Part sizes', height: 300, show: d => !!d.labels,
      sub: 'Share of the vertices in each part, largest first.',
      draw(host, d) {
        const n = d.sizes.length, tot = d.sizes.reduce((a, b) => a + b, 0), col = partColours(n), idx = d.sizes.map((_, i) => i + 1);
        const P = frame(host, { h: 300, x: [0.5, n + 0.5], y: [0, d.sizes[0] / tot * 1.15], xt: idx, yf: v => Math.round(v * 100) + '%', xl: 'Part', yl: 'Share of vertices' });
        d.sizes.forEach((c, i) => { const x0 = P.X(i + 1 - 0.34), x1 = P.X(i + 1 + 0.34); E('rect', { x: x0, y: P.Y(c / tot), width: x1 - x0, height: P.Y(0) - P.Y(c / tot), rx: 2, fill: col[i] }, P.body); });
        hover(host, P, px => { const i = Math.max(0, Math.min(n - 1, Math.round(P.xInv(px)) - 1));
          return { cross: P.X(i + 1), html: tip(`Part ${i + 1}`, [['Vertices', int(d.sizes[i])], ['Share', pct(d.sizes[i] / tot, 1)]]) }; });
      } },
  ],
  insights: d => {
    const name = MESH[d.mesh].label;
    if (d.labels) return [
      `Each vertex gets ${d.gps_dims} numbers, φᵢ(v)/√λᵢ. Low eigenfunctions change slowly across the surface, so vertices on the same limb get similar numbers and k-means groups them.`,
      `The largest part holds <b>${pct(d.sizes[0] / d.mesh_data.n_vertices, 0)}</b> of the vertices; the smallest holds ${pct(Math.min(...d.sizes) / d.mesh_data.n_vertices, 1)}. Small parts tend to be the extremities.`,
      'No labels or training data are used: the parts come from the geometry of the mesh alone.',
    ];
    const c10 = d.coefficients_for_10pct;
    return [
      c10 ? `The ${name.toLowerCase()} needs <b>${c10} of its eigenfunctions</b> for 10% error. The rest of the ${int(d.mesh_data.n_vertices)} possible ones carry fine detail.` : `Even ${d.basis} eigenfunctions leave more than 10% error on this mesh.`,
      `λ₀ = ${fmt(d.eigenvalues[0], 4)}: the first eigenfunction is constant, so k = 1 puts every vertex at the centre (100% error). Each next one adds a smooth bend.`,
      `With k = ${d.k} the error is <b>${pct(d.error, 1)}</b>. Changing k is one matrix product, because the ${d.basis} eigenpairs were solved once and kept.`,
    ];
  },
  maths: { eq: '−L φᵢ = λᵢ M φᵢ,   φᵢᵀ M φⱼ = δᵢⱼ\nLᵢⱼ = ½ (cot αᵢⱼ + cot βᵢⱼ),   M = diag(Voronoi areas)\nx̂ = Σ_{i<k} φᵢ (φᵢᵀ M x)\nGPS(v) = (φ₁(v)/√λ₁, …, φ₂₀(v)/√λ₂₀)',
    notes: ['The smallest eigenpairs come from shift-invert Lanczos near zero. The error is the area-weighted distance to the original, relative to the centred mesh.'] },
};

/* ================================================================ L3: curvature */
const TYPE = { hyperbolic: 'a saddle point', elliptic: 'a dome', planar: 'flat to second order', parabolic: 'curved one way only' };
const glyphOf = t => t === 'hyperbolic' ? 0 : t === 'elliptic' ? 2 : 1;
function surfGlyph(svg, cx, cy, f, colour) {
  const P = (u, v) => [cx + (u - v) * 36, cy + (u + v) * 15 - f(u, v) * 24];
  const g = [-1, -0.5, 0, 0.5, 1], t = [...Array(13).keys()].map(i => -1 + i / 6);
  const line = pts => S('path', { d: 'M' + pts.map(p => p.map(q => q.toFixed(1)).join(',')).join('L'), fill: 'none', style: `stroke:${colour}`, 'stroke-width': 1.4, opacity: 0.9 }, svg);
  g.forEach(a => { line(t.map(b => P(a, b))); line(t.map(b => P(b, a))); });
}

const curvature = {
  id: 'curvature', group: 'shape', nav: 'Curvature', tag: 'IT575 L3', title: 'Curvature on meshes and on a surface patch',
  question: 'Where does a surface bend like a dome, and where like a saddle? And why do the two always balance out?',
  params: [
    { k: 'mesh', type: 'seg', label: 'Mesh', value: 'cat4', options: [['cat4', 'Cat'], ['bunny', 'Bunny'], ['cactus', 'Cactus']], group: 'Mesh',
      help: 'The cat is open at the base. The bunny and the cactus are closed surfaces.' },
    { k: 'field', type: 'seg', label: 'Colour by', value: 'gaussian', options: [['gaussian', 'Gaussian K'], ['mean', 'Mean H']], group: 'Mesh',
      help: 'Gaussian curvature K = k₁k₂: negative at saddles, positive at domes. Mean curvature H = (k₁ + k₂)/2.' },
    { k: 'surface', type: 'seg', label: 'Patch z = f(u, v)', value: 'saddle', options: [['saddle', 'Saddle'], ['monkey', 'Monkey saddle'], ['paraboloid', 'Paraboloid']], group: 'Surface patch',
      help: 'Saddle u² − v², monkey saddle u³ − 3uv², paraboloid u² + v², on [−1, 1]².' },
  ],
  presets: [
    { label: 'Cat (open mesh)', values: { mesh: 'cat4', field: 'gaussian', surface: 'saddle' } },
    { label: 'Bunny (closed, 4π)', values: { mesh: 'bunny', field: 'gaussian' } },
    { label: 'Cactus mean curvature', values: { mesh: 'cactus', field: 'mean' } },
    { label: 'Monkey saddle patch', values: { surface: 'monkey' } },
  ],
  debounce: 30,
  request: async (s) => {
    const [cP, h1] = cached('curv:' + s.mesh, () => call('/api/shape/curvature', { mesh: s.mesh }));
    const [pP, h2] = cached('surf:' + s.surface, () => call('/api/shape/surface-curvature', { surface: s.surface, n: 41 }));
    const [m, c, p] = await Promise.all([meshData(s.mesh), cP, pP]);
    return { ...c, mesh_data: m, surf: p, meta: { compute_ms: (h1 ? 0 : c.meta.compute_ms) + (h2 ? 0 : p.meta.compute_ms), cached: h1 && h2 } };
  },
  kpis: (d, s) => {
    const gb = d.gauss_bonnet, ok = Math.abs(gb.integral - gb.expected) < 1e-6 + 1e-4 * Math.abs(gb.expected);
    const vals = d[s.field], neg = vals.filter(v => v < 0).length / vals.length, o = d.surf.at_origin;
    return [
      { label: 'Gauss–Bonnet Σ K·A', value: Math.abs(gb.integral) < 1e-6 ? gb.integral.toExponential(1) : fmt(gb.integral, 4), tone: ok ? 'good' : 'warn',
        note: `expected 2πχ = ${gb.closed ? '4π ≈ ' + fmt(gb.expected, 4) : fmt(gb.expected, 0)}` },
      { label: 'Euler characteristic χ', value: gb.euler_characteristic, note: gb.closed ? 'closed surface, χ = V − E + F' : `open: ${gb.boundary_vertices} boundary vertices` },
      { label: s.field === 'gaussian' ? 'Vertices with K < 0' : 'Vertices with H < 0', value: pct(neg, 0), note: s.field === 'gaussian' ? 'saddle-shaped places' : 'bending toward the normal' },
      { label: 'Patch at the origin', value: `K = ${fmt(o.K, 2)}`, note: `H = ${fmt(o.H, 2)}, ${TYPE[o.type] || o.type}` },
    ];
  },
  illustration: {
    title: 'The sign of Gaussian curvature', height: 236,
    caption: 'K is the product of the two principal curvatures. The highlighted shape matches the patch at its origin. The bar is the colour scale on the mesh.',
    draw(svg, d, s) {
      const sameS = d && d.surf.surface === s.surface, sameM = d && d.mesh === s.mesh;
      const pick = sameS ? glyphOf(d.surf.at_origin.type) : { saddle: 0, monkey: 1, paraboloid: 2 }[s.surface];
      const G = [
        { f: (u, v) => u * u - v * v, c: NEG, t: 'K < 0', sub: 'saddle' },
        { f: u => u * u, c: MID, t: 'K = 0', sub: 'cylinder or plane' },
        { f: (u, v) => u * u + v * v, c: POS, t: 'K > 0', sub: 'dome or bowl' },
      ];
      G.forEach((g, i) => {
        const cx = 100 + i * 200;
        if (i === pick) S('rect', { x: cx - 92, y: 8, width: 184, height: 150, rx: 10, class: 'node hot' }, svg);
        surfGlyph(svg, cx, 80, g.f, g.c);
        text(svg, cx, 132, g.t, 'n-title'); text(svg, cx, 149, g.sub, 'n-sub');
      });
      const gr = S('linearGradient', { id: 'kscale', x1: 0, x2: 1, y1: 0, y2: 0 }, svg.querySelector('defs'));
      [[0, NEG], [0.5, 'rgb(219,222,227)'], [1, POS]].forEach(([o, c]) => S('stop', { offset: o, 'stop-color': c }, gr));
      S('rect', { x: 150, y: 180, width: 300, height: 10, rx: 5, fill: 'url(#kscale)' }, svg);
      const lim = sameM ? Math.max(...d.range[s.field].map(Math.abs)) : null, sym = s.field === 'gaussian' ? 'K' : 'H';
      text(svg, 150, 206, lim != null ? `${sym} ≤ −${fmt(lim, lim < 1 ? 3 : 1)}` : `${sym} < 0`, 'n-sub', 'start');
      text(svg, 300, 206, '0', 'n-sub');
      text(svg, 450, 206, lim != null ? `${sym} ≥ +${fmt(lim, lim < 1 ? 3 : 1)}` : `${sym} > 0`, 'n-sub', 'end');
      if (sameM) {
        const gb = d.gauss_bonnet;
        text(svg, 300, 228, `${MESH[d.mesh].label}: Σ K·A = ${Math.abs(gb.integral) < 1e-6 ? '0.000' : fmt(gb.integral, 3)}, 2πχ = ${fmt(gb.expected, 3)}`, 'edge-lbl');
      }
    },
  },
  chartLayout: 'two',
  charts: [
    { id: 'mesh', kind: '3d', title: 'Curvature on the mesh',
      sub: (d, s) => { const lim = Math.max(...d.range[s.field].map(Math.abs)); return `${s.field === 'gaussian' ? 'Gaussian K' : 'Mean H'} per vertex: blue negative, red positive, full colour at ±${fmt(lim, lim < 1 ? 3 : 1)} (2nd–98th percentile).`; },
      draw(v, d, s, ctx) {
        const m = d.mesh_data, lim = Math.max(...d.range[s.field].map(Math.abs)) || 1;
        put(v, ctx, 'curv3d', d.mesh, m.vertices, m.faces, diverging(d[s.field], lim));
      } },
    { id: 'patch', title: 'Gaussian curvature of the patch', height: 380,
      sub: d => `${d.surf.formula} on a ${d.surf.n} × ${d.surf.n} grid, from the first and second fundamental forms. Blue K < 0, red K > 0.`,
      legend: [['neutral', 'dot', 'Origin']],
      draw(host, d) {
        const Sf = d.surf, n = Sf.u.length, w = Math.max(260, host.clientWidth), side = Math.min(w - 70, 330), ml = Math.max(44, (w - side) / 2);
        const P = frame(host, { h: side + 60, x: [Sf.u[0], Sf.u[n - 1]], y: [Sf.v[0], Sf.v[n - 1]], m: { l: ml, r: Math.max(8, w - ml - side), t: 24, b: 36 }, xl: 'u', yl: 'v', xn: 5, yn: 5 });
        const lim = Math.max(...Sf.K.flat().map(Math.abs)) || 1, du = (Sf.u[1] - Sf.u[0]) / 2, dv = (Sf.v[1] - Sf.v[0]) / 2;
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
          const t = Math.max(-1, Math.min(1, Sf.K[i][j] / lim)), a = Math.abs(t);
          E('rect', { x: P.X(Sf.u[j] - du), y: P.Y(Sf.v[i] + dv), width: P.X(Sf.u[j] + du) - P.X(Sf.u[j] - du) + 0.5, height: P.Y(Sf.v[i] - dv) - P.Y(Sf.v[i] + dv) + 0.5,
            fill: t < 0 ? `rgba(41,120,214,${a.toFixed(3)})` : `rgba(227,74,71,${a.toFixed(3)})` }, P.body);
        }
        P.dot(0, 0, 'fn', 4);
        const vAt = py => Sf.v[0] + (P.h - P.M.b - py) / (P.h - P.M.t - P.M.b) * (Sf.v[n - 1] - Sf.v[0]);
        hover(host, P, (px, py) => { const j = nearest(Sf.u, P.xInv(px)), i = nearest(Sf.v, vAt(py));
          return { pts: [[P.X(Sf.u[j]), P.Y(Sf.v[i]), 'fn']], html: tip(`u ${fmt(Sf.u[j])}, v ${fmt(Sf.v[i])}`, [['K', fmt(Sf.K[i][j], 4)], ['H', fmt(Sf.H[i][j], 4)], ['Closed form K', fmt(Sf.closed_form.K[i][j], 4)]]) }; });
      } },
  ],
  insights: (d, s) => {
    const gb = d.gauss_bonnet, name = MESH[d.mesh].label, K = d.gaussian, o = d.surf.at_origin, cf = d.surf.closed_form;
    const neg = K.filter(v => v < 0).length / K.length, pos = K.filter(v => v > 0).length / K.length;
    return [
      gb.closed ? `The ${name.toLowerCase()} is closed with χ = ${gb.euler_characteristic}, so its total Gaussian curvature must be 2πχ = 4π ≈ 12.566. The discrete sum gives <b>${fmt(gb.integral, 4)}</b>, however bumpy the surface.`
        : `The ${name.toLowerCase()} is an open mesh (χ = ${gb.euler_characteristic}, ${gb.boundary_vertices} boundary vertices), so Gauss–Bonnet expects a total of 0. The discrete sum is <b>${gb.integral.toExponential(1)}</b>.`,
      s.field === 'gaussian' ? `${pct(neg, 0)} of the vertices have K < 0 and ${pct(pos, 0)} have K > 0. However they are spread, saddles and domes add up to the same total.`
        : 'H is the average of the two principal curvatures. It is positive where the surface bends away from its outward normal, like a sphere.',
      `${d.surf.formula}: at the origin K = ${fmt(o.K, 2)}, ${TYPE[o.type] || o.type}.${cf ? ` The fundamental-form calculation matches the closed form to ${(+cf.max_abs_diff).toExponential(0)}.` : ''}`,
    ];
  },
  maths: { eq: 'K(v) = (2π − Σⱼ θⱼ) / A(v)        π instead of 2π on the boundary\nH(v) = −½ (Δx · n)\nΣᵥ K(v) A(v) = 2π χ,   χ = V − E + F\nz = f(u, v):  K = (f_uu f_vv − f_uv²) / (1 + f_u² + f_v²)²',
    notes: ['K is the angle defect at each vertex divided by its Voronoi area, so the sum of K·A is the total angle defect. Δ is the cotangent Laplacian.'] },
};

/* ================================================================ L5: skinning */
const METHOD = { lbs: ['LBS', 'Linear blend'], nlerp: ['NLERP', 'Quaternion NLERP'], slerp: ['SLERP', 'Quaternion SLERP'], dqs: ['DQS', 'Dual quaternion'] };
const angleOf = s => s.mesh === 'cylinder' ? s.angle : s.bend;
const lbsMid = deg => Math.abs(Math.cos(deg * Math.PI / 360));

const skinning = {
  id: 'skinning', group: 'shape', nav: 'Skinning', tag: 'IT575 L5', title: 'Skinning: why dual quaternions keep a twisted arm round',
  question: 'Twist one end of an arm by 180°. Why does the standard method crush the middle, and how do quaternions fix it?',
  params: [
    { k: 'mesh', type: 'seg', label: 'Rig', value: 'cylinder', options: [['cylinder', 'Cylinder twist'], ['cactus', 'Cactus bend']], group: 'Rig',
      help: 'Cylinder: the bottom cap is fixed and the top cap twists about the axis. Cactus: four handles; both arms bend.' },
    { k: 'method', type: 'seg', label: 'Blending', value: 'dqs', options: [['lbs', 'LBS'], ['nlerp', 'NLERP'], ['slerp', 'SLERP'], ['dqs', 'DQS']], group: 'Blending',
      help: 'LBS averages matrices. NLERP and SLERP blend rotations as quaternions. DQS blends rotation and translation together as dual quaternions.' },
    { k: 'angle', label: 'Twist angle', min: -360, max: 360, step: 5, value: 180, unit: '°', group: 'Pose', show: s => s.mesh === 'cylinder',
      help: 'Rotation of the top cap about the cylinder axis.' },
    { k: 'bend', label: 'Bend angle', min: -60, max: 60, step: 1, value: 18, unit: '°', group: 'Pose', show: s => s.mesh === 'cactus',
      help: 'Bend of the first arm. The second arm bends by 0.84 of this.' },
  ],
  presets: [
    { label: 'DQS 180° twist', values: { mesh: 'cylinder', method: 'dqs', angle: 180 } },
    { label: 'LBS 180° twist (candy wrapper)', values: { mesh: 'cylinder', method: 'lbs', angle: 180 } },
    { label: 'LBS 90° twist', values: { mesh: 'cylinder', method: 'lbs', angle: 90 } },
    { label: 'Cactus bend', values: { mesh: 'cactus', method: 'dqs', bend: 18 } },
  ],
  debounce: 40,
  request: (s, sig) => call('/api/shape/skin', { mesh: s.mesh, method: s.method, angle: angleOf(s) }, { signal: sig }),
  kpis: d => {
    const vt = Math.abs(1 - d.volume_ratio) < 0.02 ? 'good' : Math.abs(1 - d.volume_ratio) < 0.1 ? 'warn' : 'bad';
    const cr = d.cross_section_ratio;
    return [
      { label: 'Blending', value: METHOD[d.method][0], note: `${METHOD[d.method][1]}, ${fmt(d.angle, 0)}°` },
      { label: 'Volume kept', value: pct(d.volume_ratio, 1), tone: vt, note: 'signed volume vs the rest pose' },
      cr != null ? { label: 'Thinnest cross-section', value: pct(cr, 1), tone: cr >= 0.95 ? 'good' : cr >= 0.7 ? 'warn' : 'bad', note: 'of the rest radius' }
        : { label: 'Handles', value: d.handles.length, note: 'bounded biharmonic weights' },
      { label: 'Mesh', value: int(d.n_vertices), unit: 'vertices', note: `${int(d.n_faces)} faces, deformed on the server` },
    ];
  },
  illustration: {
    title: 'Two ways to blend the handle transforms', height: 220,
    caption: 'The tube outline is the real radius along the cylinder once data arrives. Halfway along, each vertex is half bottom handle, half top handle.',
    draw(svg, d, s) {
      const ang = angleOf(s), same = d && d.mesh === s.mesh, lbs = s.method === 'lbs', cy = 96;
      let ratio;
      if (s.mesh === 'cylinder') {
        const R = 30, x0 = 26, L = 226;
        const prof = same && d.cross_section_profile ? d.cross_section_profile : null;
        const zs = prof ? prof.z : [...Array(61).keys()].map(i => i / 5), zmax = zs[zs.length - 1] || 1;
        const rs = prof ? prof.ratio : zs.map(z => { const w = z / zmax; return lbs ? Math.sqrt((1 - w) ** 2 + w * w + 2 * w * (1 - w) * Math.cos(ang * Math.PI / 180)) : 1; });
        const X = z => x0 + z / zmax * L;
        const top = zs.map((z, i) => `${X(z).toFixed(1)},${(cy - R * rs[i]).toFixed(1)}`), bot = zs.map((z, i) => `${X(z).toFixed(1)},${(cy + R * rs[i]).toFixed(1)}`).reverse();
        S('path', { d: 'M' + top.join('L') + 'L' + bot.join('L') + 'Z', class: 'node' }, svg);
        S('ellipse', { cx: x0, cy, rx: 7, ry: R, class: 'node' }, svg);
        S('ellipse', { cx: x0 + L, cy, rx: 7, ry: Math.max(1, R * rs[rs.length - 1]), class: 'node hot' }, svg);
        S('line', { x1: X(zmax / 2), x2: X(zmax / 2), y1: cy - R - 4, y2: cy + R + 4, class: 'ln-m', 'stroke-dasharray': '3 3' }, svg);
        S('circle', { cx: x0, cy, r: 4, class: 'fill1' }, svg); S('circle', { cx: x0 + L, cy, r: 4, class: 'fill2' }, svg);
        edge(svg, [x0 + L - 16, cy - R - 8], [x0 + L + 16, cy - R - 8], { curve: [0, -26], label: `twist ${ang}°`, width: 2 });
        text(svg, x0 - 8, cy + R + 22, 'fixed cap', 'n-sub', 'start'); text(svg, x0 + L + 10, cy + R + 22, 'twisted cap', 'n-sub', 'end');
        text(svg, X(zmax / 2), cy + R + 22, 'w = ½, ½', 'n-sub');
        ratio = same && d.cross_section_ratio != null ? d.cross_section_ratio : lbs ? lbsMid(ang) : 1;
      } else {
        const jx = 150, jy = 104, j2x = 104, j2y = 84;
        S('rect', { x: 104, y: 36, width: 46, height: 160, rx: 20, class: 'node' }, svg);
        S('rect', { x: jx, y: jy - 15, width: 74, height: 30, rx: 14, class: 'node hot', transform: `rotate(${-ang} ${jx} ${jy})` }, svg);
        S('rect', { x: j2x - 70, y: j2y - 14, width: 70, height: 28, rx: 13, class: 'node hot', transform: `rotate(${ang * 0.84} ${j2x} ${j2y})` }, svg);
        S('circle', { cx: jx, cy: jy, r: 4, class: 'fill2' }, svg); S('circle', { cx: j2x, cy: j2y, r: 4, class: 'fill2' }, svg);
        text(svg, 230, 60, `arm 1: ${ang}°`, 'edge-lbl', 'start'); text(svg, 24, 40, `arm 2: ${fmt(ang * 0.84, 0)}°`, 'edge-lbl', 'start');
        text(svg, 127, 214, '4 handles', 'n-sub');
        ratio = same ? d.volume_ratio : 1;
      }
      const A = node(svg, { x: 300, y: 20, w: 168, h: 64, title: 'Blend matrices', sub: 'LBS: Σ wᵢ Rᵢ', hot: lbs });
      const qsub = { nlerp: 'NLERP: ‖Σ wᵢ qᵢ‖', slerp: 'SLERP: arc qᵢ → qⱼ', dqs: 'DQS: Σ wᵢ (qᵢ + ε dᵢ)', lbs: 'NLERP, SLERP, DQS' }[s.method];
      const B = node(svg, { x: 300, y: 112, w: 168, h: 64, title: 'Blend quaternions', sub: qsub, hot: !lbs });
      const src = [266, cy];
      edge(svg, src, [A.x - 5, A.cy], { width: lbs ? 4 : 1.5, flow: lbs });
      edge(svg, src, [B.x - 5, B.cy], { width: lbs ? 1.5 : 4, flow: !lbs });
      const gx = 540, gr = 38, cyl = s.mesh === 'cylinder', rr = gr * (cyl ? ratio : Math.sqrt(Math.max(0, ratio)));
      S('circle', { cx: gx, cy, r: gr, fill: 'none', class: 'ln-m', 'stroke-dasharray': '4 4', 'stroke-width': 1.5 }, svg);
      S('circle', { cx: gx, cy, r: Math.max(1, rr), class: ratio > 0.95 ? 'fill3' : ratio > 0.7 ? 'fill4' : 'fill2', opacity: 0.55 }, svg);
      edge(svg, [(lbs ? A : B).r + 4, (lbs ? A : B).cy], [gx - gr - 6, cy], { width: 2 });
      text(svg, gx, cy - gr - 10, cyl ? 'thinnest ring' : 'volume', 'n-sub');
      text(svg, gx, cy + gr + 22, pct(ratio, 1), 'n-val');
      text(svg, 300, 208, cyl ? (lbs ? `halfway: ½R(0) + ½R(${ang}°) scales a ring by |cos(θ/2)| = ${fmt(lbsMid(ang), 3)}` : 'every vertex gets a true rotation, so rings keep their radius')
        : (lbs ? 'averaged matrices shrink the elbows' : 'blended rotations stay rigid'), 'n-sub');
    },
  },
  chartLayout: 'two',
  charts: [
    { id: 'mesh', kind: '3d', title: 'The deformed mesh',
      sub: d => `${METHOD[d.method][1]} skinning, ${d.mesh === 'cylinder' ? 'twist' : 'bend'} ${fmt(d.angle, 0)}°. Drag to look along the tube.`,
      draw(v, d, s, ctx) { put(v, ctx, 'skin3d', d.mesh, d.vertices, d.faces, null); } },
    { id: 'profile', title: 'Radius kept along the cylinder', height: 300, show: d => !!d.cross_section_profile,
      sub: 'Mean radius of each ring of vertices, as a share of its rest radius.',
      legend: [['s2', 'line', 'Radius kept'], ['neutral', 'dash', 'Rest pose']],
      draw(host, d) {
        const pr = d.cross_section_profile;
        const P = frame(host, { h: 300, x: extent(pr.z), y: [0, 1.1], yf: v => Math.round(v * 100) + '%', xl: 'Height along the cylinder', yl: 'Radius kept' });
        P.hline(1); P.line(zip(pr.z, pr.ratio), 's2');
        hover(host, P, px => { const i = nearest(pr.z, P.xInv(px)); return { cross: P.X(pr.z[i]), pts: [[P.X(pr.z[i]), P.Y(pr.ratio[i]), 'f2']], html: tip(`Height ${fmt(pr.z[i], 2)}`, [['Radius kept', pct(pr.ratio[i], 1)]]) }; });
      } },
    { id: 'runs', kind: 'html', title: 'Settings compared', sub: 'Each result you view is added here, newest first, so methods can be compared side by side.',
      draw(host, d, s, ctx) {
        const key = `${d.mesh}|${d.method}|${d.angle}`;
        const log = (ctx.extra.skinLog || []).filter(r => r.key !== key);
        log.unshift({ key, mesh: d.mesh, method: d.method, angle: d.angle, vol: d.volume_ratio, cr: d.cross_section_ratio });
        ctx.extra.skinLog = log.slice(0, 8);
        const td = (v, st = '') => `<td style="text-align:right;${st}">${v}</td>`;
        host.innerHTML = `<div class="tbl-wrap"><table><thead><tr><th style="text-align:left">Rig</th><th style="text-align:left">Method</th><th style="text-align:right">Angle</th><th style="text-align:right">Volume kept</th><th style="text-align:right">Thinnest ring</th></tr></thead><tbody>` +
          ctx.extra.skinLog.map((r, i) => `<tr style="${i === 0 ? 'font-weight:600' : ''}"><td>${MESH[r.mesh].label}</td><td>${METHOD[r.method][0]}</td>${td(fmt(r.angle, 0) + '°')}${td(pct(r.vol, 1), Math.abs(1 - r.vol) >= 0.1 ? 'color:var(--bad)' : '')}${td(r.cr == null ? '—' : pct(r.cr, 1), r.cr != null && r.cr < 0.7 ? 'color:var(--bad)' : '')}</tr>`).join('') +
          '</tbody></table></div>';
      } },
  ],
  insights: d => {
    const cyl = d.mesh === 'cylinder', full = d.angle !== 0 && Math.abs(d.angle) % 360 === 0;
    const out = [];
    if (full) out.push(`At ${fmt(d.angle, 0)}° both handles end with the same rotation matrix, so every method returns the rest pose: blending cannot see a full turn.`);
    if (full) { /* nothing to compare at a full turn */ }
    else if (cyl && d.method === 'lbs') out.push(`LBS averages the two rotation matrices. Halfway along, ½R(0) + ½R(${fmt(d.angle, 0)}°) is not a rotation: it scales the ring by |cos(θ/2)| = ${fmt(lbsMid(d.angle), 3)}. Here the thinnest ring keeps <b>${pct(d.cross_section_ratio, 1)}</b> of its radius and the volume falls to ${pct(d.volume_ratio, 0)}.`);
    else if (cyl) out.push(`${METHOD[d.method][1]} blending gives every vertex a true rotation, so the thinnest ring keeps <b>${pct(d.cross_section_ratio, 1)}</b> of its radius and the volume stays at ${pct(d.volume_ratio, 1)}.`);
    else out.push(`The cactus has ${d.handles.length} handles with smooth weights. With ${METHOD[d.method][0]} at a ${fmt(d.angle, 0)}° bend its volume ends at <b>${pct(d.volume_ratio, 1)}</b> of the rest pose. Switch methods to compare.`);
    out.push('NLERP and SLERP blend only the rotation and average the translations. DQS blends rotation and translation together as one dual quaternion, which matters when handles sit far apart, as on the cactus.');
    return out;
  },
  maths: { eq: "LBS    x′ = (Σᵢ wᵢ Rᵢ) x + Σᵢ wᵢ tᵢ\nNLERP  q = Σᵢ wᵢ qᵢ / ‖Σᵢ wᵢ qᵢ‖\nDQS    q̂ = Σᵢ wᵢ (qᵢ + ε dᵢ) / ‖Σᵢ wᵢ qᵢ‖,   dᵢ = ½ (0, tᵢ) qᵢ\n       x′ = R(q) x + 2 vec(d q*)",
    notes: ['Dual-quaternion skinning after Kavan et al. (2007). Weights wᵢ are bounded biharmonic weights loaded with each rig.', 'The cross-section is the mean distance of each ring of vertices from the axis, divided by its rest value.'] },
};

export default [fourier, jaw, spectral, curvature, skinning];
