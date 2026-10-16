// Shape tab: ShapeLab's five modules, computed by the Python library on each request.
import { $, $$, bind, call, fmt, int, livePanel, pct, readouts, segmented } from './api.js';
import { E, extent, frame, hover, nearest, responsive, zip } from './plot.js';
import { categorical, diverging, viewer } from './viewer3d.js';

const tip = (title, rows) => `<b>${title}</b>` + rows.map(([k, v]) => `<br>${k}: <span class="n">${v}</span>`).join('');

/* equal-aspect 2D frame for outlines and jawlines */
function equalFrame(host, pts, h, flipY = true) {
  const [x0, x1] = extent(pts.map(p => p[0])), [y0, y1] = extent(pts.map(p => p[1]));
  const w = Math.max(260, host.clientWidth), pw = w - 32, ph = h - 32;
  const s = Math.min(pw / (x1 - x0 || 1), ph / (y1 - y0 || 1)) * 0.95, cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  return frame(host, { h, x: [cx - pw / 2 / s, cx + pw / 2 / s], y: flipY ? [cy + ph / 2 / s, cy - ph / 2 / s] : [cy - ph / 2 / s, cy + ph / 2 / s], xt: [], yt: [], m: { l: 16, r: 16, t: 16, b: 16 } });
}

function fourier(root) {
  const s = { outline: 'bird1', keep: 6 };
  const panel = $('[data-panel="fourier"]', root);
  const live = livePanel(panel, { request: sig => call('/api/shape/fourier', s, { signal: sig }), render, delay: 60 });
  const k = bind(panel, s, 'keep', () => live.refresh(), v => v);
  segmented($('[data-seg="outline"]', panel), v => { s.outline = v; live.refresh(); });
  const [h1, h2] = $$('.plot-host', panel);
  const draw = () => {
    const d = live.data; if (!d) return;
    let P = equalFrame(h1, d.original, Math.min(380, Math.max(260, h1.clientWidth * 0.6)));
    P.line(d.original.concat([d.original[0]]), 's1 thin');
    P.line(d.reconstruction.concat([d.reconstruction[0]]), 's2');
    const ec = d.error_curve, K = ec.map((_, i) => i);
    P = frame(h2, { h: 220, x: [0, K.length - 1], y: [0, Math.min(1, Math.max(...ec.slice(1)) * 1.05)], yf: v => Math.round(v * 100) + '%', xl: 'Coefficients kept', yl: 'Relative error' });
    P.hline(0.1); P.line(zip(K, ec), 's1'); P.vline(Math.min(d.keep, K.length - 1), 'ln s2 thin');
    hover(h2, P, px => { const i = nearest(K, P.xInv(px)); return { cross: P.X(i), pts: [[P.X(i), P.Y(ec[i]), 'f1']], html: tip(`${i} coefficients`, [['Error', pct(ec[i])]]) }; }, { onClick: px => { const i = Math.max(1, nearest(K, P.xInv(px))); k.set(i); live.refresh(); } });
  };
  responsive(h1, draw); responsive(h2, draw);
  function render(d) {
    $('[data-k="keep"]', panel).max = Math.min(200, d.n_points);
    readouts($('.readout', panel), [['Points', d.n_points], ['Share of coefficients', pct(d.share, 2)], ['Relative error', pct(d.error)], ['Needed for 10%', d.coefficients_for_10pct]]);
    draw();
  }
  return live;
}

function jaw(root) {
  const s = { b1: 0, b2: 0, b3: 0 };
  const panel = $('[data-panel="jaw"]', root);
  let model = null;
  const live = livePanel(panel, { request: async sig => {
    if (!model) model = await call('/api/shape/jaw-model', { n_modes: 6 }, { signal: sig });
    const smp = await call('/api/shape/jaw-sample', s, { signal: sig });
    return Object.assign(smp, { model });
  }, render, delay: 40 });
  ['b1', 'b2', 'b3'].forEach(k => bind(panel, s, k, () => live.refresh(), v => (v > 0 ? '+' : '') + fmt(v, 1) + ' sd'));
  const [h1, h2] = $$('.plot-host', panel);
  const draw = () => {
    const d = live.data; if (!d) return;
    const m = d.model, all = m.aligned_sample.flat().concat(d.shape);
    let P = equalFrame(h1, all, 320);
    m.aligned_sample.forEach(j => P.line(j, 'sn thin').setAttribute('opacity', '.3'));
    P.line(m.mean, 'sn dashed'); P.line(d.shape, 's2');
    d.shape.forEach(p => P.dot(p[0], p[1], 'f2', 2.5));
    const v = m.variance_explained, idx = v.map((_, i) => i + 1);
    P = frame(h2, { h: 200, x: [0.5, v.length + 0.5], y: [0, 1], xt: idx, yf: v => Math.round(v * 100) + '%', xl: 'Mode', yl: 'Variance explained' });
    P.bars(idx.map(i => i - 0.35).concat([v.length + 0.35]), v, 'f1', 0);
    P.line(zip(idx, m.cumulative), 's2'); idx.forEach(i => P.dot(i, m.cumulative[i - 1], 'f2', 3.5));
  };
  responsive(h1, draw); responsive(h2, draw);
  function render(d) {
    const m = d.model;
    readouts($('.readout', panel), [['Jawlines', m.n_shapes], ['Landmarks', m.n_landmarks], ['First three modes', pct(m.cumulative[2])], ['Mode 1 alone', pct(m.variance_explained[0])]]);
    draw();
  }
  return live;
}

function meshCache() {
  const c = {};
  return (name, sig) => c[name] || (c[name] = call(`/api/shape/mesh/${name}`, {}, { signal: sig }).catch(e => { delete c[name]; throw e; }));
}
const getMesh = meshCache();

function spectral(root) {
  const s = { mesh: 'cat4', k: 30, basis: 120, view: 'smooth', parts: 5 };
  const panel = $('[data-panel="spectral"]', root);
  const v3 = viewer($('.viewer3d', panel));
  let shown = null;
  const live = livePanel(panel, { request: async sig => {
    const mesh = await getMesh(s.mesh, sig);
    if (s.view === 'segment') return Object.assign(await call('/api/shape/segment', { mesh: s.mesh, parts: s.parts, basis: s.basis }, { signal: sig }), { mesh_data: mesh });
    return Object.assign(await call('/api/shape/spectral', { mesh: s.mesh, k: s.k, basis: s.basis }, { signal: sig }), { mesh_data: mesh });
  }, render, delay: 60 });
  const ch = () => live.refresh();
  const kc = bind(panel, s, 'k', ch, v => v);
  bind(panel, s, 'parts', ch, v => v + ' parts');
  segmented($('[data-seg="mesh"]', panel), v => { s.mesh = v; s.basis = v === 'bunny' ? 100 : 120; $('[data-k="k"]', panel).max = s.basis; if (s.k > s.basis) kc.set(s.basis); ch(); });
  segmented($('[data-seg="view"]', panel), v => { s.view = v; $$('.only-smooth', panel).forEach(e => e.hidden = v !== 'smooth'); $$('.only-seg', panel).forEach(e => e.hidden = v !== 'segment'); ch(); });
  $$('.only-seg', panel).forEach(e => e.hidden = true);
  const host = $('.plot-host', panel);
  const draw = responsive(host, () => {
    const d = live.data; if (!d || !d.error_curve) return;
    const ec = d.error_curve, K = ec.map((_, i) => i + 1);
    const P = frame(host, { h: 200, x: [1, K.length], y: [0, Math.min(1, Math.max(...ec) * 1.05)], yf: v => Math.round(v * 100) + '%', xl: 'Eigenfunctions used', yl: 'Relative error' });
    P.hline(0.1); P.line(zip(K, ec), 's1'); P.vline(d.k, 'ln s2 thin');
    hover(host, P, px => { const i = nearest(K, P.xInv(px)); return { cross: P.X(K[i]), pts: [[P.X(K[i]), P.Y(ec[i]), 'f1']], html: tip(`k = ${K[i]}`, [['Error', pct(ec[i])], ['Eigenvalue', fmt(d.eigenvalues[i], 3)]]) }; }, { onClick: px => { kc.set(K[nearest(K, P.xInv(px))]); ch(); } });
  });
  function render(d) {
    const m = d.mesh_data, faces = shown !== s.mesh ? m.faces : null;
    if (v3) {
      if (s.view === 'segment') v3.set(m.vertices, faces, { colors: categorical(d.labels), refit: !!faces });
      else v3.set(d.vertices, faces, { refit: !!faces });
    }
    shown = s.mesh;
    if (s.view === 'segment') readouts($('.readout', panel), [['Mesh', `${int(m.n_vertices)} vertices`], ['Parts', d.parts], ['Part sizes', d.sizes.join(' · ')]]);
    else readouts($('.readout', panel), [['Mesh', `${int(m.n_vertices)} vertices`], ['Eigenfunctions', `${d.k} of ${d.basis} computed`], ['Relative error', pct(d.error)], ['Needed for 10%', d.coefficients_for_10pct ?? `> ${d.basis}`]]);
    draw();
  }
  return live;
}

function curvature(root) {
  const s = { mesh: 'cat4', field: 'gaussian', surface: 'saddle' };
  const panel = $('[data-panel="curvature"]', root);
  const v3 = viewer($('.viewer3d', panel));
  let shown = null;
  const live = livePanel(panel, { request: async sig => {
    const [mesh, c, surf] = await Promise.all([getMesh(s.mesh, sig), call('/api/shape/curvature', { mesh: s.mesh }, { signal: sig }), call('/api/shape/surface-curvature', { surface: s.surface, n: 41 }, { signal: sig })]);
    return Object.assign(c, { mesh_data: mesh, surf, meta: { compute_ms: c.meta.compute_ms + surf.meta.compute_ms, cached: c.meta.cached && surf.meta.cached } });
  }, render, delay: 30 });
  segmented($('[data-seg="mesh"]', panel), v => { s.mesh = v; live.refresh(); });
  segmented($('[data-seg="field"]', panel), v => { s.field = v; live.redraw(); });
  segmented($('[data-seg="surface"]', panel), v => { s.surface = v; live.refresh(); });
  const host = $('.plot-host', panel);
  const draw = responsive(host, () => {
    const d = live.data; if (!d) return;
    const S = d.surf, n = S.u.length, size = Math.min(host.clientWidth, 360);
    const P = frame(host, { h: size, x: [S.u[0], S.u[n - 1]], y: [S.v[0], S.v[n - 1]], m: { l: 40, r: 8, t: 8, b: 30 }, xl: 'u', xn: 4, yn: 4 });
    const lim = Math.max(...S.K.flat().map(Math.abs)) || 1, du = (S.u[1] - S.u[0]) / 2, dv = (S.v[1] - S.v[0]) / 2;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const t = Math.max(-1, Math.min(1, S.K[i][j] / lim)), a = Math.abs(t);
      E('rect', { x: P.X(S.u[j] - du), y: P.Y(S.v[i] + dv), width: P.X(S.u[j] + du) - P.X(S.u[j] - du) + 0.5, height: P.Y(S.v[i] - dv) - P.Y(S.v[i] + dv) + 0.5, fill: t < 0 ? `rgba(42,120,214,${a})` : `rgba(227,73,72,${a})` }, P.body);
    }
    hover(host, P, (px, py) => { const j = nearest(S.u, P.xInv(px)), i = nearest(S.v, S.v[0] + (P.h - P.M.b - py) / (P.h - P.M.t - P.M.b) * (S.v[n - 1] - S.v[0])); return { html: tip(`u ${fmt(S.u[j])}, v ${fmt(S.v[i])}`, [['K', fmt(S.K[i][j], 4)], ['H', fmt(S.H[i][j], 4)]]) }; });
  });
  function render(d) {
    const m = d.mesh_data, vals = d[s.field], lim = d.range?.[s.field] ? Math.max(...d.range[s.field].map(Math.abs)) : Math.max(...vals.map(Math.abs));
    if (v3) v3.set(m.vertices, shown !== s.mesh ? m.faces : null, { colors: diverging(vals, lim), refit: shown !== s.mesh });
    shown = s.mesh;
    const gb = d.gauss_bonnet, o = d.surf.at_origin, cf = d.surf.closed_form;
    readouts($('.readout', panel), [['Gauss–Bonnet ∑K·A', `${fmt(gb.integral, 4)} (expected ${fmt(gb.expected, 4)}${gb.closed ? '' : ', open mesh'})`], ['Euler characteristic', gb.euler_characteristic]]);
    readouts($('[data-out="surf"]', panel), [['Surface', `${d.surf.formula}`], ['At the origin', `K ${fmt(o.K, 3)}, H ${fmt(o.H, 3)} (${o.type})`], ['vs closed form', cf ? `max |ΔK| ${(+cf.max_abs_diff).toExponential(1)}` : '—']]);
    draw();
  }
  return live;
}

function skinning(root) {
  const s = { mesh: 'cylinder', method: 'dqs', angle: 180 };
  const panel = $('[data-panel="skinning"]', root);
  const v3 = viewer($('.viewer3d', panel));
  let shown = null;
  const live = livePanel(panel, { request: sig => call('/api/shape/skin', s, { signal: sig }), render, delay: 40 });
  const ang = bind(panel, s, 'angle', () => live.refresh(), v => v + '°');
  segmented($('[data-seg="mesh"]', panel), v => { s.mesh = v; const el = $('[data-k="angle"]', panel); if (v === 'cactus') { el.min = -60; el.max = 60; ang.set(18); } else { el.min = -360; el.max = 360; ang.set(180); } live.refresh(); });
  segmented($('[data-seg="method"]', panel), v => { s.method = v; live.refresh(); });
  const host = $('.plot-host', panel);
  const draw = responsive(host, () => {
    const d = live.data; if (!d || !d.cross_section_profile) { host.innerHTML = ''; return; }
    const pr = d.cross_section_profile;
    const P = frame(host, { h: 200, x: extent(pr.z), y: [0, 1.1], yf: v => Math.round(v * 100) + '%', xl: 'Height along the cylinder', yl: 'Cross-section kept' });
    P.hline(1); P.line(zip(pr.z, pr.ratio), 's2');
    hover(host, P, px => { const i = nearest(pr.z, P.xInv(px)); return { cross: P.X(pr.z[i]), pts: [[P.X(pr.z[i]), P.Y(pr.ratio[i]), 'f2']], html: tip(`z = ${fmt(pr.z[i], 2)}`, [['Kept', pct(pr.ratio[i])]]) }; });
  });
  function render(d) {
    if (v3) v3.set(d.vertices, shown !== s.mesh ? d.faces : null, { refit: shown !== s.mesh });
    shown = s.mesh;
    readouts($('.readout', panel), [['Method', { lbs: 'Linear blend', nlerp: 'Quaternion NLERP', slerp: 'Quaternion SLERP', dqs: 'Dual quaternion' }[d.method]], ['Volume kept', pct(d.volume_ratio)]].concat(d.cross_section_ratio != null ? [['Thinnest cross-section kept', pct(d.cross_section_ratio)]] : []));
    draw();
  }
  return live;
}

export function initShape(root) {
  const views = { fourier, jaw, spectral, curvature, skinning };
  const lives = {};
  const show = name => {
    $$('[data-panel]', root).forEach(p => p.hidden = p.dataset.panel !== name);
    if (!lives[name]) { lives[name] = views[name](root); lives[name].now(); } else lives[name].redraw();
  };
  const sel = segmented($('.subnav', root), v => { show(v); try { localStorage.setItem('ml-shape', v); } catch (e) {} });
  let first = 'fourier'; try { first = localStorage.getItem('ml-shape') || 'fourier'; } catch (e) {}
  if (!views[first]) first = 'fourier';
  sel(first); show(first);
}
