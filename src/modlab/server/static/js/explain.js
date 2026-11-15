// The explainer page: a long-form, illustrated explanation of one model. All content,
// including every figure, comes from GET /api/explain/<group>/<view>; figures with
// sliders recompute on the server through GET /api/explain/figure/<id>.
import { call, fmtMs } from './api.js';
import { decorate, responsive } from './plot.js';
import { h, icon, seg } from './ui/dom.js';
import { drawFigure, legendOf, svgKind } from './ui/figures.js';

/* Inline markup from the server: keep a small set of formatting tags, drop everything else. */
const ALLOWED = new Set(['B', 'I', 'EM', 'STRONG', 'CODE', 'SUB', 'SUP', 'BR', 'SPAN']);
function rich(tag, html) {
  const el = h(tag), tpl = document.createElement('template');
  tpl.innerHTML = String(html ?? '');
  const walk = n => [...n.childNodes].forEach(c => {
    if (c.nodeType === 1) {
      if (!ALLOWED.has(c.tagName)) { c.replaceWith(document.createTextNode(c.textContent)); return; }
      [...c.attributes].forEach(a => c.removeAttribute(a.name));
      walk(c);
    } else if (c.nodeType !== 3) c.remove();
  });
  walk(tpl.content); el.append(tpl.content);
  return el;
}
const pad2 = i => String(i + 1).padStart(2, '0');
const qs = o => new URLSearchParams(Object.entries(o).map(([k, v]) => [k, String(v)])).toString();

export function renderExplainer(root, key, { groupMeta, spec }) {
  const disposers = [];
  let alive = true;
  const progress = h('div.ex-progress', h('i'));
  const page = h('div.page.explain', h('div.ex-loading', h('div.skeleton-plot', { style: { height: '180px' } }), h('div.skeleton-plot', { style: { height: '420px' } })));
  root.replaceChildren(progress, page);
  const modelHref = `#/${key}`;

  call(`/api/explain/${key}`).then(d => { if (alive) build(d); }).catch(e => {
    if (!alive) return;
    page.replaceChildren(h('div.ex-hero.card', h('h1', 'No explanation yet'), h('p.lede', e.message), h('div.hero-actions', h('a.btn.primary', { href: modelHref }, icon('play', ''), 'Open the live model'))));
  });

  function build(d) {
    const figCount = d.sections.flatMap(s => s.blocks).filter(b => b.type === 'figure').length;
    const crumbs = h('div.crumbs', h('a', { href: '#/' }, 'Home'), '›', h('a', { href: `#/${groupMeta.views[0].group}/${groupMeta.views[0].id}` }, groupMeta.label), '›',
      h('a', { href: modelHref }, spec.nav || spec.title), '›', h('span', 'Explanation'));
    const hero = h('header.ex-hero',
      crumbs,
      h('span.ex-kicker', icon('book', ''), d.kicker),
      rich('h1', d.title),
      rich('p.lede', d.summary),
      h('div.ex-meta',
        h('span', `${d.minutes} min read`), h('span', `${d.sections.length} sections`),
        h('span', `${figCount} live figures`), h('span.compute', `server ${fmtMs(d.meta?.compute_ms)}`)),
      h('div.hero-actions',
        h('a.btn.primary', { href: modelHref }, icon('play', ''), 'Open the live model'),
        h('a.btn', { href: '#', onclick: e => { e.preventDefault(); document.getElementById('ex-takeaways')?.scrollIntoView({ behavior: 'smooth' }); } }, 'Jump to the takeaways')));

    const tocLinks = d.sections.map((s, i) => h('a', { href: '#', 'data-sec': s.id, onclick: e => { e.preventDefault(); document.getElementById('ex-' + s.id)?.scrollIntoView({ behavior: 'smooth', block: 'start' }); } }, h('span.num', pad2(i)), s.title));
    const toc = h('aside.ex-toc', h('p.eyebrow', 'On this page'), h('nav', ...tocLinks,
      h('a', { href: '#', 'data-sec': 'takeaways', onclick: e => { e.preventDefault(); document.getElementById('ex-takeaways')?.scrollIntoView({ behavior: 'smooth' }); } }, h('span.num', '✓'), 'Takeaways')),
      h('a.btn.sm.block', { href: modelHref }, icon('play', ''), 'Live model'));

    const article = h('article.ex-article',
      ...d.sections.map((s, i) => h('section.ex-sec.reveal', { id: 'ex-' + s.id, 'data-sec': s.id },
        h('h2', h('span.num', pad2(i)), s.title), ...s.blocks.map(b => block(b)))),
      h('section.ex-sec.card.ex-takeaways.reveal', { id: 'ex-takeaways', 'data-sec': 'takeaways' },
        h('h2', h('span.num', '✓'), 'Takeaways'), h('ul', ...d.takeaways.map(t => rich('li', t)))),
      d.glossary?.length ? h('section.ex-sec.reveal', h('h2', h('span.num', 'Aa'), 'Glossary'),
        h('dl.ex-glossary', ...d.glossary.flatMap(([t, def]) => [h('dt', t), rich('dd', def)]))) : null,
      h('nav.ex-next',
        d.prev ? h('a.card', { href: `#/${d.prev.key}/explain` }, h('span.eyebrow', '← Previous explanation'), h('b', d.prev.title)) : h('span'),
        h('a.card.ex-live', { href: modelHref }, h('span.eyebrow', 'Now try it'), h('b', `Open ${spec.nav || spec.title}`)),
        d.next ? h('a.card', { href: `#/${d.next.key}/explain` }, h('span.eyebrow', 'Next explanation →'), h('b', d.next.title)) : h('span')));

    page.replaceChildren(hero, h('div.ex-body', toc, article));
    figures.forEach(f => f.first());

    // Scroll spy, reveal-on-scroll and the reading progress bar.
    const spy = new IntersectionObserver(es => es.forEach(e => {
      if (e.isIntersecting) { e.target.classList.add('in'); }
      if (e.isIntersecting) toc.querySelectorAll('nav a').forEach(a => a.classList.toggle('on', a.dataset.sec === e.target.dataset.sec));
    }), { rootMargin: '-30% 0px -60% 0px', threshold: 0 });
    const reveal = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); reveal.unobserve(e.target); } }), { threshold: 0.08 });
    article.querySelectorAll('.ex-sec').forEach(s => { spy.observe(s); reveal.observe(s); });
    const secs = [...article.querySelectorAll('.ex-sec')];
    const onScroll = () => {
      const max = document.documentElement.scrollHeight - innerHeight; progress.firstChild.style.width = (max > 0 ? Math.min(100, scrollY / max * 100) : 0) + '%';
      // Backup for the observer (which can be delayed, e.g. in background tabs): reveal anything on screen.
      for (const s of secs) if (!s.classList.contains('in')) { const r = s.getBoundingClientRect(); if (r.top < innerHeight && r.bottom > 0) s.classList.add('in'); }
    };
    addEventListener('scroll', onScroll, { passive: true }); onScroll();
    disposers.push(() => { spy.disconnect(); reveal.disconnect(); removeEventListener('scroll', onScroll); });
  }

  /* ---------------------------------------------------------------- blocks */
  const figures = [];
  function block(b) {
    switch (b.type) {
      case 'p': return rich('p.ex-p', b.text);
      case 'eq': return h('div.ex-eq', h('pre', b.text), b.where?.length ? h('dl', ...b.where.flatMap(([s, m]) => [h('dt', s), rich('dd', m)])) : null);
      case 'callout': return h(`div.ex-callout.${b.tone || 'info'}`, h('span.ex-callout-ico', b.tone === 'warn' ? '!' : b.tone === 'tip' ? '★' : 'i'), h('div', b.title ? h('b', b.title) : null, rich('p', b.text)));
      case 'steps': return h('div.ex-steps', b.title ? h('p.eyebrow', b.title) : null, h('ol', ...b.items.map(t => rich('li', t))));
      case 'list': return h('ul.ex-list', ...b.items.map(t => rich('li', t)));
      case 'cards': return h('div.ex-cards', ...b.items.map((c, i) => h('div.card.ex-card', h('span.ex-card-dot', String(i + 1)), h('b', c.title), rich('p', c.text))));
      case 'compare': return h('div.ex-compare', ...b.items.map(c => h('div.card', h('b', c.title), h('ul', ...c.points.map(p => rich('li', p))))));
      case 'quiz': return quiz(b);
      case 'try': return h('a.ex-try', { href: `#/${key}${b.params && Object.keys(b.params).length ? '?' + qs(b.params) : ''}` },
        h('span.ex-try-ico', icon('play', '')), h('span', h('b', b.label), b.note ? rich('span', b.note) : h('span', 'Opens the live model with these settings')), icon('arrow', ''));
      case 'figure': return figure(b);
      default: return null;
    }
  }

  function quiz(b) {
    const why = rich('p.ex-why', b.why); why.hidden = true;
    const opts = b.options.map((o, i) => h('button.ex-opt', { type: 'button', onclick: () => {
      opts.forEach((btn, j) => { btn.disabled = true; btn.classList.toggle('right', j === b.answer); btn.classList.toggle('wrong', j === i && i !== b.answer); });
      why.hidden = false; box.classList.add(i === b.answer ? 'ok' : 'miss');
    } }, h('span.ex-opt-key', String.fromCharCode(65 + i)), rich('span', o)));
    const box = h('div.card.ex-quiz', h('p.eyebrow', 'Check yourself'), rich('p.ex-q', b.q), h('div.ex-opts', ...opts), why);
    return box;
  }

  function figure(b) {
    const host = h('div.fig-host');
    const legend = h('div.legend');
    const notes = h('ul.fig-notes');
    const badge = h('span.compute', 'live');
    const state = Object.fromEntries((b.controls || []).map(c => [c.k, c.value]));
    let data = b.data, timer = 0, ctrl = null, drawn = false;
    const paint = () => {
      if (!host.isConnected || !data) return;
      drawFigure(host, data);
      if (svgKind(data.kind)) { decorate(host, { animate: !drawn }); }
      drawn = true;
      legend.replaceChildren(...legendOf(data).map(l => h('span', { style: { color: `var(--${l.color})` } }, h('i' + (l.dash ? '.dash' : l.box ? '.box' : '')), l.name)));
      legend.hidden = !legend.children.length;
      notes.replaceChildren(...(data.notes || []).map(n => rich('li', n)));
      notes.hidden = !notes.children.length;
    };
    const run = svgKind(b.data?.kind) || b.data?.kind === 'grid' ? responsive(host, paint) : paint;
    if (run.dispose) disposers.push(run.dispose);
    const refetch = () => {
      clearTimeout(timer);
      timer = setTimeout(async () => {
        ctrl?.abort(); ctrl = new AbortController();
        card.classList.add('is-loading'); badge.className = 'compute busy'; badge.textContent = 'computing…';
        try {
          const r = await call(`/api/explain/figure/${b.figure}`, { ...(b.params || {}), ...state }, { signal: ctrl.signal });
          data = r; badge.className = 'compute'; badge.textContent = `server ${fmtMs(r.meta?.compute_ms)}`; paint();
        } catch (e) { if (e.name !== 'AbortError') { badge.className = 'compute err'; badge.textContent = e.message; } }
        finally { card.classList.remove('is-loading'); }
      }, 90);
    };
    const controls = (b.controls || []).map(c => {
      if (c.options) {
        const s = seg(c.options, c.value, v => { state[c.k] = v; refetch(); }, { label: c.label });
        return h('div.fig-ctl.fig-choice', h('span', c.label), s);
      }
      const out = h('output', fmtCtl(c, state[c.k]));
      const input = h('input', { type: 'range', min: c.min, max: c.max, step: c.step, value: c.value, 'aria-label': c.label });
      const fill = () => input.style.setProperty('--fill', ((state[c.k] - c.min) / (c.max - c.min) * 100) + '%');
      input.addEventListener('input', () => { state[c.k] = +input.value; out.textContent = fmtCtl(c, state[c.k]); fill(); refetch(); });
      fill();
      return h('label.fig-ctl', h('span', c.label), out, input);
    });
    const card = h('figure.card.ex-fig',
      h('div.ex-fig-head', h('div', b.title ? h('h3', b.title) : null, b.caption ? rich('p.sub', b.caption) : null), badge),
      controls.length ? h('div.fig-ctls', ...controls) : null,
      legend, host, notes);
    badge.textContent = 'live';
    figures.push({ first: () => run() });
    return card;
  }
  const fmtCtl = (c, v) => `${(+v).toFixed(String(c.step).includes('.') ? String(c.step).split('.')[1].length : 0)}${c.unit ? ' ' + c.unit : ''}`;

  return { destroy() { alive = false; disposers.forEach(f => f()); progress.remove(); } };
}
