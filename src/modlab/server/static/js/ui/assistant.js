// The "Ask" assistant: a panel on every page that sends the question (typed or spoken)
// together with what the page shows to /api/assistant/ask, streams the answer, and can read
// it aloud through /api/assistant/speak. Gemini is only ever called by the server.
// Nothing is stored: the conversation lives in this panel until the page is reloaded.
import { ApiError, call } from '../api.js';
import { h, icon, seg } from './dom.js';

/* Minimal, safe formatting for answers: **bold**, `code`, "- " bullets, paragraphs. */
function format(text) {
  const box = h('div.as-text');
  const inline = (s, el) => {
    s.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).forEach(part => {
      if (/^\*\*[^*]+\*\*$/.test(part)) el.append(h('b', part.slice(2, -2)));
      else if (/^`[^`]+`$/.test(part)) el.append(h('code', part.slice(1, -1)));
      else if (part) el.append(document.createTextNode(part));
    });
    return el;
  };
  let list = null;
  for (const line of text.split('\n')) {
    const m = line.match(/^\s*[-•*]\s+(.*)$/);
    if (m) { if (!list) { list = h('ul'); box.append(list); } list.append(inline(m[1], h('li'))); continue; }
    list = null;
    if (line.trim()) box.append(inline(line, h('p')));
  }
  return box;
}

async function streamAsk(body, on) {
  const res = await fetch('/api/assistant/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!res.ok) { const j = await res.json().catch(() => ({})); throw new ApiError(j.error?.message || `Request failed (${res.status})`, j.error?.code, res.status); }
  const reader = res.body.getReader(), dec = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf('\n\n')) >= 0) {
      const block = buf.slice(0, i); buf = buf.slice(i + 2);
      let ev = 'message', data = '';
      for (const line of block.split('\n')) { if (line.startsWith('event:')) ev = line.slice(6).trim(); else if (line.startsWith('data:')) data += line.slice(5).trim(); }
      if (data) on(ev, JSON.parse(data));
    }
  }
}

const blobToB64 = blob => new Promise((ok, bad) => { const r = new FileReader(); r.onload = () => ok(String(r.result).split(',')[1]); r.onerror = bad; r.readAsDataURL(blob); });

export function mountAssistant({ getContext }) {
  let status = null, lang = 'hinglish', open = false, busy = false, history = [], audio = null, rec = null;

  const fab = h('button.as-fab', { type: 'button', 'aria-label': 'Ask the AI tutor', title: 'Ask the AI tutor (Gemini)' }, icon('spark', ''), h('span', 'Ask'));
  const msgs = h('div.as-msgs', { 'aria-live': 'polite' });
  const input = h('textarea', { rows: 1, placeholder: 'Ask about this page… (Enter to send)', 'aria-label': 'Your question' });
  const mic = h('button.as-mic', { type: 'button', 'aria-label': 'Hold to speak', title: 'Hold to speak (or click to start and stop)' }, icon('mic', ''));
  const send = h('button.as-send', { type: 'button', 'aria-label': 'Send' }, icon('send', ''));
  const meter = h('div.as-meter', { hidden: true }, ...Array.from({ length: 18 }, () => h('i')), h('span.as-timer', '0.0 s'));
  const langSeg = seg([['hinglish', 'Hinglish'], ['english', 'English']], lang, v => { lang = v; refreshSuggestions(); input.placeholder = v === 'hinglish' ? 'Is page ke baare mein poochho…' : 'Ask about this page…'; }, { label: 'Answer language' });
  const about = h('span.as-about');
  const clear = h('button.icon-btn.as-clear', { type: 'button', 'aria-label': 'Clear the conversation', title: 'Clear the conversation' }, icon('trash', ''));
  const close = h('button.icon-btn', { type: 'button', 'aria-label': 'Close' }, icon('close', ''));
  const panel = h('section.as-panel', { role: 'dialog', 'aria-label': 'AI tutor', hidden: true },
    h('header.as-head', h('div.as-title', h('span.as-orb', icon('spark', '')), h('div', h('b', 'Ask the tutor'), about)), clear, close),
    h('div.as-bar', langSeg, h('span.as-model')),
    msgs,
    h('footer.as-foot', meter, h('div.as-input', input, mic, send)));
  document.body.append(fab, panel);

  input.placeholder = 'Is page ke baare mein poochho…';
  const autosize = () => { input.style.height = 'auto'; input.style.height = Math.min(140, input.scrollHeight) + 'px'; };
  input.addEventListener('input', autosize);
  input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(input.value); } });
  send.addEventListener('click', () => ask(input.value));
  fab.addEventListener('click', () => toggle(true));
  close.addEventListener('click', () => toggle(false));
  clear.addEventListener('click', () => { history = []; stopAudio(); renderEmpty(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && open) toggle(false); });

  async function toggle(v) {
    open = v; panel.hidden = !v; fab.classList.toggle('hide', v);
    if (!v) { stopAudio(); return; }
    const ctx = getContext();
    about.textContent = ctx.page.title ? `About: ${ctx.page.title}` : 'About this site';
    if (!status) {
      try { status = await call('/api/assistant/status'); } catch (e) { status = { enabled: false, error: e.message }; }
      panel.querySelector('.as-model').textContent = status.enabled ? `Gemini · ${status.model}` : '';
    }
    if (!history.length) renderEmpty();
    setTimeout(() => input.focus(), 50);
  }

  async function refreshSuggestions() {
    const box = msgs.querySelector('.as-sugg'); if (!box || !status?.enabled) return;
    const { view } = getContext();
    try {
      const { suggestions } = await call('/api/assistant/suggestions', { view, lang });
      box.replaceChildren(...suggestions.map(q => h('button.as-chip', { type: 'button', onclick: () => ask(q) }, q)));
    } catch (e) { box.replaceChildren(); }
  }

  function renderEmpty() {
    if (status && !status.enabled) {
      msgs.replaceChildren(h('div.as-empty', h('span.as-orb.big', icon('spark', '')), h('b', 'The AI tutor is not switched on here'),
        h('p', status.error || 'This server has no Gemini API key. To enable it locally, put GEMINI_API_KEY in the project\'s .env file and restart the server.')));
      input.disabled = mic.disabled = send.disabled = true;
      return;
    }
    input.disabled = mic.disabled = send.disabled = false;
    msgs.replaceChildren(h('div.as-empty', h('span.as-orb.big', icon('spark', '')),
      h('b', lang === 'hinglish' ? 'Kuch bhi poochho!' : 'Ask anything about this page'),
      h('p', lang === 'hinglish' ? 'Type karo ya mic dabake bolo. Jawab is page ke settings aur results pe based hoga, aur 🔊 se sun bhi sakte ho.'
        : 'Type, or hold the mic and speak. Answers use this page\'s settings and results, and 🔊 reads them aloud.'),
      h('div.as-sugg')));
    refreshSuggestions();
  }

  function bubble(role, text = '') {
    msgs.querySelector('.as-empty')?.remove();
    const body = role === 'user' ? h('div.as-text', text) : h('div.as-text', h('span.as-typing', h('i'), h('i'), h('i')));
    const el = h(`div.as-msg.${role}`, body);
    msgs.append(el); msgs.scrollTop = msgs.scrollHeight;
    return el;
  }

  async function ask(q, audioBlob = null) {
    q = (q || '').trim();
    if (busy || (!q && !audioBlob) || !status?.enabled) return;
    busy = true; send.disabled = mic.disabled = true; input.value = ''; autosize(); stopAudio();
    const userEl = bubble('user', audioBlob ? '🎙️ …' : q);
    const botEl = bubble('model');
    let text = '', asked = q;
    const ctx = getContext();
    const body = { lang, view: ctx.view, page: ctx.page, history: history.slice(-8) };
    try {
      if (audioBlob) body.audio = { data: await blobToB64(audioBlob), mime_type: audioBlob.type || 'audio/webm' };
      else body.question = q;
      await streamAsk(body, (ev, d) => {
        if (ev === 'transcript') { asked = d.text; userEl.firstChild.textContent = '🎙️ ' + d.text; }
        else if (ev === 'delta') { text += d.text; botEl.firstChild.replaceWith(format(text)); msgs.scrollTop = msgs.scrollHeight; }
        else if (ev === 'done') { text = d.text; botEl.firstChild.replaceWith(format(text)); addActions(botEl, text, d, !!audioBlob); }
        else if (ev === 'error') throw new ApiError(d.message, d.code);
      });
      if (text) history.push({ role: 'user', text: asked }, { role: 'model', text });
    } catch (e) {
      botEl.classList.add('err'); botEl.firstChild.replaceWith(h('div.as-text', e.message || 'Something went wrong.'));
    } finally {
      busy = false; send.disabled = mic.disabled = false; input.focus();
    }
  }

  function addActions(el, text, meta, autoplay) {
    const listen = h('button.as-act', { type: 'button' }, icon('speaker', ''), h('span', lang === 'hinglish' ? 'Suno' : 'Listen'));
    const spokenLang = lang;
    let url = null;
    listen.addEventListener('click', async () => {
      if (audio && audio._btn === listen) { stopAudio(); return; }
      stopAudio();
      listen.classList.add('loading'); listen.lastChild.textContent = '…';
      try {
        if (!url) {
          const res = await fetch('/api/assistant/speak', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, lang: spokenLang }) });
          if (!res.ok) { const j = await res.json().catch(() => ({})); throw new Error(j.error?.message || 'Could not read it aloud.'); }
          url = URL.createObjectURL(await res.blob());
        }
        audio = new Audio(url); audio._btn = listen;
        listen.classList.remove('loading'); listen.classList.add('playing'); listen.replaceChildren(icon('stop', ''), h('span', 'Stop'));
        audio.onended = () => stopAudio();
        await audio.play();
      } catch (e) {
        listen.classList.remove('loading'); listen.lastChild.textContent = e.message;
      }
    });
    el.append(h('div.as-acts', listen, h('span.as-meta', `${(meta.ms / 1000).toFixed(1)} s · ${meta.model}`)));
    if (autoplay) listen.click();
  }
  function stopAudio() {
    if (!audio) return;
    audio.pause(); const b = audio._btn; audio = null;
    if (b) { b.classList.remove('playing'); b.replaceChildren(icon('speaker', ''), h('span', lang === 'hinglish' ? 'Suno' : 'Listen')); }
  }

  /* ---------------------------------------------------------------- push-to-talk */
  async function startRec() {
    if (rec || busy || !status?.enabled) return;
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) { toastIn('This browser cannot record audio.'); return; }
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
    catch (e) { toastIn('Microphone permission was denied.'); return; }
    const type = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg'].find(t => MediaRecorder.isTypeSupported(t)) || '';
    const mr = new MediaRecorder(stream, type ? { mimeType: type } : undefined), chunks = [];
    const ac = new (window.AudioContext || window.webkitAudioContext)(), an = ac.createAnalyser(); an.fftSize = 64;
    ac.createMediaStreamSource(stream).connect(an);
    const bins = new Uint8Array(an.frequencyBinCount), bars = [...meter.querySelectorAll('i')], t0 = performance.now();
    rec = { mr, stream, ac, raf: 0, cancelled: false };
    mr.ondataavailable = e => e.data.size && chunks.push(e.data);
    mr.onstop = () => {
      stream.getTracks().forEach(t => t.stop()); ac.close(); cancelAnimationFrame(rec.raf);
      const cancelled = rec.cancelled; rec = null;
      mic.classList.remove('rec'); meter.hidden = true;
      const blob = new Blob(chunks, { type: (type || 'audio/webm').split(';')[0] });
      if (!cancelled && blob.size > 600) ask('', blob);
    };
    const draw = () => {
      an.getByteFrequencyData(bins);
      bars.forEach((b, i) => { b.style.transform = `scaleY(${0.15 + (bins[i + 1] || 0) / 255})`; });
      const s = (performance.now() - t0) / 1000;
      meter.querySelector('.as-timer').textContent = `${s.toFixed(1)} s`;
      if (s >= (status.max_record_s || 30)) { stopRec(); return; }
      rec.raf = requestAnimationFrame(draw);
    };
    mr.start(); mic.classList.add('rec'); meter.hidden = false; draw();
  }
  function stopRec(cancel = false) { if (!rec) return; rec.cancelled = cancel; if (rec.mr.state !== 'inactive') rec.mr.stop(); }
  let pressT = 0;
  mic.addEventListener('pointerdown', e => { e.preventDefault(); pressT = performance.now(); if (rec) { stopRec(); return; } startRec(); });
  mic.addEventListener('pointerup', () => { if (rec && performance.now() - pressT > 350) stopRec(); });   // hold-to-talk; a quick click toggles
  mic.addEventListener('keydown', e => { if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); rec ? stopRec() : startRec(); } });

  function toastIn(msg) { const n = h('div.as-msg.model.err', h('div.as-text', msg)); msgs.querySelector('.as-empty')?.remove(); msgs.append(n); }

  return { open: () => toggle(true) };
}
