// App shell: sidebar navigation, routing (#/group/view?params), quick-jump palette,
// theme toggle, server status, and the Home and API pages.
import { call } from './api.js';
import { h, icon } from './ui/dom.js';
import { S, edge, node, text } from './ui/illustrate.js';
import { mountWorkbench } from './ui/workbench.js';
import spread from './views/spread.js';
import shape from './views/shape.js';
import markets from './views/markets.js';
import crossover from './views/crossover.js';
import agents from './views/agents.js';

const GROUPS = [
  { id: 'spread', label: 'Spread', icon: 'spread', color: 's1', title: 'How things move through a population',
    blurb: 'Six models from the MC312 mathematical-modelling labs, solved with RK4 and Monte Carlo on every request.', views: spread },
  { id: 'shape', label: 'Shape', icon: 'shape', color: 's2', title: 'Geometry processing from first principles',
    blurb: 'The ShapeLab library: Fourier descriptors, shape models, curvature, Laplace–Beltrami spectra and skinning.', views: shape },
  { id: 'markets', label: 'Markets', icon: 'markets', color: 's3', title: 'Can machine learning predict the market?',
    blurb: 'Walk-forward forecasting of returns and volatility, 1993–2026. Neural nets train as background jobs.', views: markets },
  { id: 'crossover', label: 'Crossover', icon: 'crossover', color: 's4', title: 'Market data through the other two lenses',
    blurb: 'Random-walk tests, fear cleared like a drug dose, and the shapes of 68 market crashes.', views: crossover },
  { id: 'agents', label: 'Agents', icon: 'agents', color: 'accent', title: 'Snake: search versus deep Q-learning',
    blurb: 'Play Snake, watch four agents play it, train the original project\'s deep Q-network live, and compare them on the same boards.', views: agents },
];
const VIEWS = GROUPS.flatMap(g => g.views.map(v => ({ ...v, groupMeta: g })));
const view = document.getElementById('view');
const app = document.getElementById('app');
let current = null;

/* ------------------------------------------------------------ navigation */
// Sidebar shows only short lab codes ("L5–6"); long tags stay on the page header.
const navTag = v => v.navTag ?? (/^Labs? /.test(v.tag || '') ? v.tag.replace(/^Labs? /, 'L') : (v.tag || '').match(/\bL\d+(–\d+)?\b/)?.[0] || '');
function buildNav() {
  const nav = document.getElementById('nav');
  nav.replaceChildren(
    h('div.nav-group', h('a', { href: '#/', 'data-route': '' }, icon('home'), 'Home')),
    ...GROUPS.filter(g => g.views.length).map(g => h('div.nav-group',
      h('div.nav-label', h('span.dot', { style: { background: `var(--${g.color})` } }), g.label),
      ...g.views.map(v => h('a', { href: `#/${g.id}/${v.id}`, 'data-route': `${g.id}/${v.id}` }, v.nav || v.title, navTag(v) ? h('span.tag', navTag(v)) : null)))),
    h('div.nav-group', h('a', { href: '#/api', 'data-route': 'api' }, icon('api'), 'API')));
}

function parse() {
  const raw = location.hash.replace(/^#\/?/, '');
  const [path, query = ''] = raw.split('?');
  return { path, params: Object.fromEntries(new URLSearchParams(query)) };
}

const LEGACY = { spread: 'spread/sir', shape: 'shape/fourier', markets: 'markets/volatility', crossover: 'crossover/random-walk', overview: '' };

function route() {
  let { path, params } = parse();
  if (path in LEGACY) { history.replaceState(null, '', '#/' + LEGACY[path]); path = LEGACY[path]; }
  current?.destroy?.(); current = null;
  document.querySelectorAll('#nav a').forEach(a => { if (a.dataset.route === path) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
  app.classList.remove('nav-open');
  window.scrollTo({ top: 0 });

  if (path === '' ) { renderHome(); document.title = 'Modeling Lab'; return; }
  if (path === 'api') { renderApi(); document.title = 'API · Modeling Lab'; return; }
  let spec = VIEWS.find(v => `${v.group}/${v.id}` === path);
  if (!spec) { const g = GROUPS.find(x => x.id === path.split('/')[0] && x.views.length); if (g) { location.hash = `#/${g.id}/${g.views[0].id}`; return; } renderMissing(); return; }
  document.title = `${spec.nav || spec.title} · Modeling Lab`;
  const g = spec.groupMeta;
  const crumbs = [h('a', { href: '#/' }, 'Home'), '›', h('a', { href: `#/${g.id}/${g.views[0].id}` }, g.label), '›', h('span', spec.nav || spec.title)];
  const start = () => {
    current = mountWorkbench(view, spec, params, {
      crumbs,
      onState: st => {
        const q = new URLSearchParams(Object.entries(st).map(([k, v]) => [k, String(v)])).toString();
        history.replaceState(null, '', `#/${spec.group}/${spec.id}${q ? '?' + q : ''}`);
      },
    });
  };
  // three.js loads with `defer`; 3D views wait for it.
  const needs3d = spec.charts?.some(c => c.kind === '3d');
  if (needs3d && !window.THREE) window.addEventListener('load', start, { once: true }); else start();
  view.focus({ preventScroll: true });
}

/* ------------------------------------------------------------ home */
function renderHome() {
  const ill = h('div', { 'data-k': 'home' });
  const page = h('div.page',
    h('section.hero',
      h('div',
        h('p.eyebrow', 'Tirth Chaudhari · computational modelling'),
        h('h1', 'Epidemics, 3D shapes, stock markets and a Snake AI, computed live on one numerical toolkit'),
        h('p.lede', { html: 'Move any control and a Flask server re-solves the model: <b>RK4 integration, Monte Carlo, eigen-decompositions, least-squares fits, walk-forward model training and deep Q-learning.</b> Nothing is precomputed, and every panel shows how long the server took.' }),
        h('div.hero-actions',
          h('a.btn.primary', { href: '#/spread/sir' }, icon('play', ''), 'Start with the 1978 flu outbreak'),
          h('a.btn', { href: '#/crossover/random-walk' }, 'See the crossover study'),
          h('button.btn.ghost', { type: 'button', onclick: openPalette }, icon('search', ''), 'Jump to a model'))),
      h('div.card.illus', ill, h('p.cap', 'One shared core of numerical methods powers all five sections.'))),
    h('div.pillars', ...GROUPS.filter(g => g.views.length).map(g => h('section.card.pillar',
      h('div.pillar-head', h('span.badge', { style: { color: `var(--${g.color})` } }, icon(g.icon, '')), h('div', h('p.eyebrow', g.label), h('h3', g.title))),
      h('p', g.blurb),
      h('ul.model-list', ...g.views.map(v => h('li', h('a', { href: `#/${g.id}/${v.id}` }, v.nav || v.title, h('span', v.tag || '')))))))),
    h('section', { style: { display: 'grid', gap: '12px' } },
      h('h2', 'How every view works'),
      h('div.steps',
        h('div.card.step', h('b', '1 · Pick a scenario'), h('p', 'Each model opens with presets such as "Early lockdown" or "Loading dose", so there is something to compare straight away.')),
        h('div.card.step', h('b', '2 · Move a control'), h('p', 'The server recomputes and the charts, result tiles and illustration update together. The badge shows the compute time.')),
        h('div.card.step', h('b', '3 · Read the results'), h('p', 'Tiles give the headline numbers, each coloured when it signals something. "What this shows" explains them in plain words.')),
        h('div.card.step', h('b', '4 · Share it'), h('p', 'The address bar always holds the current settings. Copy link sends anyone to this exact setup.')))));
  view.replaceChildren(page);
  drawCore(ill);
}

function drawCore(host) {
  const svg = S('svg', { class: 'ill', viewBox: '0 0 600 300', role: 'img', 'aria-label': 'Shared numerical core feeding five sections' }, host);
  const defs = S('defs', {}, svg), m = S('marker', { id: 'ah-home', viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 12, markerHeight: 12, markerUnits: 'userSpaceOnUse', orient: 'auto' }, defs);
  S('path', { d: 'M0 0 L10 5 L0 10 z', style: 'fill:var(--ink-2)' }, m); svg.dataset.marker = 'ah-home';
  const core = node(svg, { x: 185, y: 112, w: 230, h: 76, title: 'Shared core', sub: 'RK4 · Monte Carlo · PCA · fits', hot: true });
  const pos = [[20, 20], [420, 20], [20, 216], [420, 216], [220, 236]];
  GROUPS.forEach((g, i) => {
    const [x, y] = pos[i];
    const n = node(svg, { x, y, w: 160, h: i === 4 ? 56 : 64, title: g.label, sub: `${g.views.length} live views` });
    const from = i === 4 ? [core.cx, core.b + 4] : [i % 2 ? core.r : core.x, i < 2 ? core.y + 10 : core.b - 10];
    const to = i === 4 ? [n.cx, n.y - 4] : [i % 2 ? n.x - 4 : n.r + 4, n.cy];
    edge(svg, from, to, { width: 2.5, cls: 'st' + ((i % 4) + 1), flow: true });
  });
  text(svg, 300, 98, 'Flask JSON API', 'edge-lbl');
}

/* ------------------------------------------------------------ API */
async function renderApi() {
  const list = h('div.api-list'), out = h('pre.out', { hidden: true });
  view.replaceChildren(h('div.page',
    h('header.wb-head', h('div.crumbs', h('a', { href: '#/' }, 'Home'), '›', 'API'),
      h('div', h('h1', 'Every endpoint the app uses'), h('p.question', { html: 'JSON in, JSON out. Errors are always <span class="mono">{"error": {"code", "message"}}</span>; successful responses carry <span class="mono">meta.compute_ms</span>. Click a GET endpoint to call it with its defaults.' }))),
    h('div.wb-body', { style: { gridTemplateColumns: 'minmax(0,1fr) minmax(0,1fr)' } }, h('section.card.card-pad', list), h('div', out))));
  try {
    const { routes } = await call('/api/routes');
    list.replaceChildren(...routes.map(r => {
      const row = h('div.api-row', { role: r.methods.includes('GET') && !r.path.includes('<') ? 'button' : null, tabindex: r.methods.includes('GET') && !r.path.includes('<') ? 0 : null },
        h('span.m', r.methods.join(' ')), h('code', r.path), r.summary ? h('span.d', r.summary) : null);
      if (r.methods.includes('GET') && !r.path.includes('<')) {
        const go = async () => {
          out.hidden = false; out.textContent = `GET ${r.path} …`;
          try { const j = await call(r.path); out.textContent = `GET ${r.path}${j.meta ? '  ·  ' + j.meta.compute_ms + ' ms' : ''}\n\n` + JSON.stringify(j, (k, v) => Array.isArray(v) && v.length > 10 ? [...v.slice(0, 10), `… ${v.length - 10} more`] : v, 2).slice(0, 8000); }
          catch (e) { out.textContent = `GET ${r.path}\n\n${e.message}`; }
        };
        row.addEventListener('click', go); row.addEventListener('keydown', e => { if (e.key === 'Enter') go(); });
      } else row.style.cursor = 'default';
      return row;
    }));
  } catch (e) { list.textContent = e.message; }
}

function renderMissing() {
  view.replaceChildren(h('div.page', h('h1', 'That page does not exist'), h('p', h('a', { href: '#/' }, 'Go to the home page'))));
}

/* ------------------------------------------------------------ palette */
function openPalette() {
  const items = [{ label: 'Home', sub: '', href: '#/' }, ...VIEWS.map(v => ({ label: v.nav || v.title, sub: `${v.groupMeta.label}${v.tag ? ' · ' + v.tag : ''}`, href: `#/${v.group}/${v.id}`, words: `${v.title} ${v.question || ''}` })), { label: 'API', sub: '', href: '#/api' }];
  const input = h('input', { type: 'search', placeholder: 'Search models, e.g. "lockdown", "LSTM", "curvature"', 'aria-label': 'Search models' });
  const list = h('ul', { role: 'listbox' });
  const wrap = h('div.palette-wrap', { onclick: e => { if (e.target === wrap) close(); } }, h('div.card.palette', { role: 'dialog', 'aria-label': 'Jump to a model' }, input, list));
  let sel = 0, shown = items;
  const draw = () => {
    const q = input.value.trim().toLowerCase();
    shown = items.filter(it => !q || `${it.label} ${it.sub} ${it.words || ''}`.toLowerCase().includes(q));
    sel = Math.min(sel, Math.max(0, shown.length - 1));
    list.replaceChildren(...(shown.length ? shown.map((it, i) => h('li', { role: 'option', 'aria-selected': String(i === sel), onclick: () => go(it), onmousemove: () => { sel = i; mark(); } }, it.label, h('span', it.sub))) : [h('li.empty', 'No match')]));
  };
  const mark = () => [...list.children].forEach((li, i) => li.setAttribute('aria-selected', String(i === sel)));
  const go = it => { close(); location.hash = it.href; };
  const close = () => { wrap.remove(); document.removeEventListener('keydown', key, true); };
  const key = e => {
    if (e.key === 'Escape') { e.preventDefault(); close(); }
    else if (e.key === 'ArrowDown') { e.preventDefault(); sel = Math.min(shown.length - 1, sel + 1); mark(); list.children[sel]?.scrollIntoView({ block: 'nearest' }); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); sel = Math.max(0, sel - 1); mark(); list.children[sel]?.scrollIntoView({ block: 'nearest' }); }
    else if (e.key === 'Enter' && shown[sel]) { e.preventDefault(); go(shown[sel]); }
  };
  input.addEventListener('input', () => { sel = 0; draw(); });
  document.addEventListener('keydown', key, true);
  document.body.append(wrap); draw(); input.focus();
}

/* ------------------------------------------------------------ theme, status, menu */
function themeNow() {
  const t = document.documentElement.dataset.theme;
  return t || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
}
function paintThemeButton() {
  const b = document.getElementById('theme'), dark = themeNow() === 'dark';
  b.replaceChildren(icon(dark ? 'sun' : 'moon', ''));
  b.setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
}
document.getElementById('theme').addEventListener('click', () => {
  const next = themeNow() === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem('ml-theme', next); } catch (e) {}
  paintThemeButton();
  current?.redraw?.();
});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', paintThemeButton);

async function ping() {
  const el = document.getElementById('status');
  try { const t0 = performance.now(); await call('/api/health'); el.className = 'status-pill ok'; el.lastElementChild.textContent = `server live · ${Math.round(performance.now() - t0)} ms`; }
  catch (e) { el.className = 'status-pill down'; el.lastElementChild.textContent = 'server unreachable'; }
}

const menu = document.getElementById('menu');
menu.append(icon('menu', ''));
menu.addEventListener('click', () => { const open = !app.classList.contains('nav-open'); app.classList.toggle('nav-open', open); menu.setAttribute('aria-expanded', String(open)); });
document.getElementById('scrim').addEventListener('click', () => app.classList.remove('nav-open'));
document.querySelectorAll('[data-palette]').forEach(b => { if (!b.children.length) b.append(icon('search', '')); b.addEventListener('click', openPalette); });
document.addEventListener('keydown', e => {
  if ((e.key === 'k' && (e.metaKey || e.ctrlKey)) || (e.key === '/' && !/input|textarea|select/i.test(document.activeElement?.tagName))) { e.preventDefault(); if (!document.querySelector('.palette-wrap')) openPalette(); }
});

buildNav();
paintThemeButton();
window.addEventListener('hashchange', route);
route();
ping(); setInterval(ping, 15000);
