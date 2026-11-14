// Small DOM helpers, the icon set, help popovers and toasts. No dependencies.

/* h('div.card.pad', {onclick, 'aria-label': ...}, child, 'text', [more]) */
export function h(sel, attrs, ...kids) {
  if (attrs == null || typeof attrs !== 'object' || attrs instanceof Node || Array.isArray(attrs)) { kids.unshift(attrs); attrs = {}; }
  const [tag, ...cls] = sel.split('.');
  const el = document.createElement(tag || 'div');
  if (cls.length) el.className = cls.join(' ');
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else if (k === 'html') el.innerHTML = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else el.setAttribute(k, v === true ? '' : v);
  }
  const add = c => { if (c == null || c === false) return; if (Array.isArray(c)) c.forEach(add); else el.append(c instanceof Node ? c : document.createTextNode(String(c))); };
  kids.forEach(add);
  return el;
}

const P = {
  home: '<path d="M3 10.5 12 3l9 7.5V21h-6v-6H9v6H3z"/>',
  spread: '<circle cx="6" cy="17" r="2.5"/><circle cx="12" cy="7" r="2.5"/><circle cx="18" cy="15" r="2.5"/><path d="M7.6 15.2 10.6 9M13.6 8.6 16.6 13"/>',
  shape: '<path d="M12 3 20 7.5v9L12 21l-8-4.5v-9z"/><path d="M12 12 20 7.5M12 12v9M12 12 4 7.5"/>',
  markets: '<path d="M3 20h18"/><path d="m4 15 4.5-5 3.5 3 7-8"/><path d="M15 5h4v4"/>',
  crossover: '<circle cx="8" cy="12" r="5"/><circle cx="16" cy="12" r="5"/>',
  agents: '<path d="M4 17c0-3 2-4 4-4h6c2 0 3-1 3-3s-1-3-3-3H9"/><circle cx="8" cy="7" r="1.6"/><path d="M4 17h8"/><circle cx="18.5" cy="17" r="1.8"/>',
  api: '<path d="M8 4c-2 0-2.5 1-2.5 3s-.5 4-2.5 5c2 1 2.5 3 2.5 5s.5 3 2.5 3M16 4c2 0 2.5 1 2.5 3s.5 4 2.5 5c-2 1-2.5 3-2.5 5s-.5 3-2.5 3"/>',
  search: '<circle cx="11" cy="11" r="6.5"/><path d="m20 20-4.2-4.2"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  moon: '<path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/>',
  menu: '<path d="M4 7h16M4 12h16M4 17h16"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
  reset: '<path d="M4 12a8 8 0 1 0 2.4-5.7"/><path d="M4 4v4.5h4.5"/>',
  link: '<path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1"/><path d="M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1"/>',
  play: '<path d="M7 5v14l11-7z"/>',
  dice: '<rect x="4" y="4" width="16" height="16" rx="3"/><circle cx="9" cy="9" r="1.2"/><circle cx="15" cy="15" r="1.2"/><circle cx="15" cy="9" r="1.2"/><circle cx="9" cy="15" r="1.2"/>',
  fit: '<path d="M4 18c3-9 6-12 16-13"/><circle cx="7" cy="12" r="1.3"/><circle cx="11" cy="9.5" r="1.3"/><circle cx="15" cy="6.5" r="1.3"/>',
  arrow: '<path d="M5 12h14M13 6l6 6-6 6"/>',
  book: '<path d="M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2z"/><path d="M4 19V5"/>',
};
export function icon(name, cls = 'ico') {
  const s = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  s.setAttribute('viewBox', '0 0 24 24'); s.setAttribute('fill', 'none'); s.setAttribute('stroke', 'currentColor');
  s.setAttribute('stroke-width', '1.8'); s.setAttribute('stroke-linecap', 'round'); s.setAttribute('stroke-linejoin', 'round');
  s.setAttribute('class', cls); s.setAttribute('aria-hidden', 'true');
  s.innerHTML = P[name] || '';
  return s;
}

/* Help popover on hover/focus of a small (i) button. */
let pop = null;
export function helpButton(text) {
  const b = h('button.help', { type: 'button', 'aria-label': text }, icon('info', ''));
  const show = () => {
    hide(); pop = h('div.pop', { role: 'tooltip' }, text); document.body.append(pop);
    const r = b.getBoundingClientRect(), w = pop.offsetWidth;
    pop.style.left = Math.max(8, Math.min(window.innerWidth - w - 8, r.left + r.width / 2 - w / 2)) + 'px';
    pop.style.top = (r.top - pop.offsetHeight - 8 < 8 ? r.bottom + 8 : r.top - pop.offsetHeight - 8) + 'px';
  };
  const hide = () => { pop?.remove(); pop = null; };
  b.addEventListener('mouseenter', show); b.addEventListener('focus', show);
  b.addEventListener('mouseleave', hide); b.addEventListener('blur', hide);
  b.addEventListener('click', e => { e.preventDefault(); pop ? hide() : show(); });
  return b;
}

export function toast(msg, ms = 3200) {
  let box = document.querySelector('.toasts');
  if (!box) { box = h('div.toasts', { 'aria-live': 'polite' }); document.body.append(box); }
  const t = h('div.toast', msg); box.append(t);
  setTimeout(() => t.remove(), ms);
}

export function seg(options, value, onPick, { block = false, label = '' } = {}) {
  const el = h('div.seg' + (block ? '.block' : ''), { role: 'group', 'aria-label': label });
  const btns = options.map(([v, text]) => h('button', { type: 'button', 'aria-pressed': String(v === value), onclick: () => { set(v); onPick(v); } }, text));
  const set = v => btns.forEach((b, i) => b.setAttribute('aria-pressed', String(options[i][0] === v)));
  el.append(...btns); el.set = set;
  return el;
}
