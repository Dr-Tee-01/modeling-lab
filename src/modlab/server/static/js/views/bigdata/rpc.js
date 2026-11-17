// Big Data Lab · L4: RPC over an unreliable network. The server runs a seeded discrete-event
// simulation of N withdraw() calls (modlab.bigdata.rpc) and marshals the arguments; the browser
// only draws the message timeline, the tables and the sweep it is sent.
import { call, fmt, int, pct } from '../../api.js';
import { catalog } from '../../catalog.js';
import { E, frame, hover, zip } from '../../plot.js';
import { edge, node, text } from '../../ui/illustrate.js';

const C = catalog.bigdata_spark.rpc;
const NAME = Object.fromEntries(C.policies);
const COL = { none: 's4', retry: 's2', idempotent: 's3' };              // presentation only
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const tip = (title, rows) => `<b>${title}</b>` + rows.map(([k, v]) => `<br>${k}: <span class="n">${v}</span>`).join('');
const money = v => v == null ? '—' : int(v);
const signed = v => (v > 0 ? '+' : v < 0 ? '−' : '') + int(Math.abs(v));
const KEYS = ['calls', 'balance', 'amount', 'latency', 'jitter', 'loss', 'dup', 'timeout', 'retries', 'seed', 'policy', 'account'];

const rpc = {
  id: 'rpc', group: 'bigdata', nav: 'RPC & retries', tag: 'L4', title: 'Remote procedure calls over an unreliable network',
  question: 'A request or its reply can be lost or duplicated. When the client retries, does the money leave the account once, twice, or not at all?',
  params: [
    { k: 'policy', type: 'seg', label: 'Retry policy', value: 'retry', options: C.policies, group: 'Client',
      help: 'No retry: one attempt, then give up (at-most-once). Retry on timeout: resend with a new request id (at-least-once). Retry + idempotency key: resend with the same key; the server answers repeats from its table without debiting again.' },
    { k: 'timeout', label: 'Timeout', min: 20, max: 2000, step: 10, value: 250, unit: 'ms', group: 'Client', help: 'How long the client waits for a reply before retrying or giving up.' },
    { k: 'retries', label: 'Max retries', min: 0, max: 6, step: 1, value: 3, group: 'Client', show: s => s.policy !== 'none', help: 'Extra attempts after the first one times out.' },
    { k: 'latency', label: 'One-way latency', min: 1, max: 500, step: 1, value: 50, unit: 'ms', group: 'Network', help: 'Fixed delay of every message.' },
    { k: 'jitter', label: 'Jitter', min: 0, max: 300, step: 5, value: 30, unit: 'ms', group: 'Network', help: 'Each message adds a random 0 to jitter ms on top of the latency.' },
    { k: 'loss', label: 'Message loss', min: 0, max: 0.6, step: 0.01, value: 0.2, scale: 100, unit: '%', group: 'Network', help: 'Chance that a request or a reply never arrives.' },
    { k: 'dup', label: 'Duplication', min: 0, max: 0.5, step: 0.01, value: 0.05, scale: 100, unit: '%', group: 'Network', help: 'Chance that the network delivers a second copy of a message a little later.' },
    { k: 'calls', label: 'Calls', min: 1, max: 30, step: 1, value: 10, group: 'Workload', help: 'withdraw() calls made one after another.' },
    { k: 'balance', label: 'Opening balance', min: 100, max: 10000, step: 100, value: 2000, group: 'Workload' },
    { k: 'amount', label: 'Amount per call', min: 10, max: 1000, step: 10, value: 100, group: 'Workload', help: 'Also the float that is marshalled below.' },
    { k: 'seed', label: 'Random seed', min: 0, max: 999, step: 1, value: 4, group: 'Workload', help: 'The same seed gives the same losses and delays, so the three policies can be compared on identical luck.' },
    { k: 'account', label: 'Account number (int32)', min: 0, max: 99999, step: 1, value: C.account, group: 'Marshalling', help: 'The integer argument whose bytes are shown in big- and little-endian order.' },
  ],
  presets: [
    { label: 'Perfect network', values: { loss: 0, dup: 0, policy: 'retry', seed: 4 } },
    { label: 'Lossy · no retry', values: { loss: 0.2, dup: 0.05, policy: 'none', seed: 4 } },
    { label: 'Lossy · retry', values: { loss: 0.2, dup: 0.05, policy: 'retry', seed: 4 } },
    { label: 'Lossy · idempotency key', values: { loss: 0.2, dup: 0.05, policy: 'idempotent', seed: 4 } },
    { label: 'Duplicating network', values: { loss: 0, dup: 0.3, policy: 'none', seed: 4 } },
  ],
  request: (s, sig) => call('/api/bigdata/rpc', Object.fromEntries(KEYS.map(k => [k, s[k]])), { signal: sig }),
  kpis: d => [
    { label: 'Final balance', value: money(d.final_balance), tone: d.difference === 0 ? 'good' : 'bad',
      note: d.difference === 0 ? `matches the expected ${money(d.expected_balance)}` : `expected ${money(d.expected_balance)} (${signed(d.difference)})` },
    { label: 'Double debits', value: int(d.double_debits), tone: d.double_debits ? 'bad' : 'good', note: `${int(d.executions)} debits for ${int(d.calls)} calls` },
    { label: 'Outcome unknown', value: int(d.unknown), tone: d.unknown ? 'warn' : 'good', note: `${int(d.missed)} withdrawal${d.missed === 1 ? '' : 's'} never ran` },
    { label: 'Messages on the wire', value: int(d.messages), unit: `· ${int(d.bytes)} B`, note: `${int(d.lost)} lost, ${int(d.duplicates)} duplicated, ${int(d.retries)} retries` },
  ],
  illustration: {
    title: 'Client, network, server', height: 210,
    caption: 'The network can delay, drop or copy each message. The server keeps the balance and, with idempotency keys, a table of replies already sent.',
    draw(svg, d, s) {
      const cli = node(svg, { x: 10, y: 60, w: 140, h: 86, title: 'Client', value: d ? `${int(d.client_ok)} ok` : 'do_withdraw()', sub: NAME[s.policy] });
      const net = node(svg, { x: 225, y: 60, w: 150, h: 86, title: 'Network', value: `${Math.round(s.loss * 100)}% lost`, sub: `${Math.round(s.dup * 100)}% copied · ${s.latency}±${s.jitter} ms`, fill: s.loss / 0.6, fillCls: 'fill2' });
      const srv = node(svg, { x: 450, y: 60, w: 140, h: 86, title: 'Server', value: d ? money(d.final_balance) : money(s.balance), sub: d ? `${int(d.executions)} debits${s.policy === 'idempotent' ? ` · ${int(d.dedup_hits)} dedup` : ''}` : 'withdraw()', hot: true, fill: d ? d.final_balance / Math.max(1, s.balance) : 1, fillCls: d && d.difference ? 'fill2' : 'fill3' });
      edge(svg, [cli.r + 4, cli.cy - 14], [net.x - 6, net.cy - 14], { label: 'request', width: 3, flow: true, cls: 'st1' });
      edge(svg, [net.r + 4, net.cy - 14], [srv.x - 6, srv.cy - 14], { label: d ? `${int(d.messages - d.lost)} delivered` : '', width: 3, flow: true, cls: 'st1' });
      edge(svg, [srv.x - 6, srv.cy + 16], [net.r + 4, net.cy + 16], { below: 'reply', width: 2.5, flow: true, cls: 'st3' });
      edge(svg, [net.x - 6, net.cy + 16], [cli.r + 4, cli.cy + 16], { below: d ? `${int(d.late_replies)} late` : '', width: 2.5, flow: true, cls: 'st3' });
      text(svg, 300, 196, d ? `${d.semantics}: ${d.difference === 0 ? 'balance correct' : `balance off by ${signed(d.difference)}`}` : '', 'n-sub');
    },
  },
  charts: [
    { id: 'timeline', title: 'Message timeline', height: 300,
      sub: d => `Every message the ${int(d.calls)} calls put on the wire. Requests go down from the client, replies go up from the server. A cross is a lost message; dashed lines are network duplicates.`,
      legend: [['s1', 'line', 'Request'], ['s3', 'line', 'Reply'], ['bad', 'line', 'Lost'], ['s4', 'dash', 'Duplicate copy'], ['s2', 'dot', 'Debit'], ['good', 'dot', 'Answered from table']],
      draw(host, d) {
        const T = d.t_end * 1.02 || 1;
        const P = frame(host, { h: 300, x: [0, T], y: [-0.18, 1.18], xl: 'Time (ms)', yt: [0, 1], yf: v => (v ? 'client' : 'server'), m: { l: 58 } });
        P.hline(0); P.hline(1);
        const yOf = who => (who === 'client' ? 1 : 0);
        d.timeline.forEach(m => {
          const y0 = yOf(m.from), y1 = 1 - y0, cls = m.kind === 'request' ? 's1' : 's3';
          if (m.lost) {
            const tl = m.t_lost, ym = (y0 + y1) / 2;
            P.line([[m.t_send, y0], [tl, ym]], 'thin ' + cls);
            const X = P.X(tl), Y = P.Y(ym);
            E('path', { d: `M${X - 5},${Y - 5}L${X + 5},${Y + 5}M${X - 5},${Y + 5}L${X + 5},${Y - 5}`, class: 'ln thin', style: 'stroke:var(--bad)' }, P.body);
          } else P.line([[m.t_send, y0], [m.t_arrive, y1]], 'thin ' + (m.copy ? 's4 dashed' : cls));
          if (m.kind === 'request' && m.outcome && !m.lost) E('circle', { cx: P.X(m.t_arrive), cy: P.Y(0), r: 4, class: 'ring', style: `fill:var(--${m.outcome === 'debit' ? 's2' : m.outcome === 'dedup' ? 'good' : 'bad'})` }, P.body);
        });
        d.marks.forEach(k => {
          if (k.kind === 'retry' || k.kind === 'give up') E('path', { d: `M${P.X(k.t)},${P.Y(1) - 8}L${P.X(k.t)},${P.Y(1) + 8}`, class: 'ln thin', style: `stroke:var(--${k.kind === 'retry' ? 'warn' : 'bad'})` }, P.body);
        });
        const all = d.timeline;
        hover(host, P, px => {
          const t = P.xInv(px);
          let best = null, bd = Infinity;
          all.forEach(m => { const dd = Math.abs((m.t_send + (m.t_arrive ?? m.t_lost)) / 2 - t); if (dd < bd) { bd = dd; best = m; } });
          if (!best) return null;
          const pc = d.per_call[best.call - 1];
          return { cross: P.X((best.t_send + (best.t_arrive ?? best.t_lost)) / 2),
            html: tip(`Call ${best.call}, attempt ${best.attempt}: ${best.kind}${best.copy ? ' (duplicate)' : ''}`, [
              ['Sent', `${fmt(best.t_send, 0)} ms`], ['Arrived', best.lost ? 'lost' : `${fmt(best.t_arrive, 0)} ms`], ['Bytes', int(best.bytes)],
              ...(best.outcome ? [['At the receiver', best.outcome]] : []), ['Call ended', pc.status]]) };
        });
      } },
    { id: 'calls', kind: 'html', title: 'What happened to each call',
      sub: d => `Client view against server view. Red rows were debited more than once; amber rows never got an answer. Policy: ${NAME[d.policy]} (${d.semantics}).`,
      caption: d => `The client believes ${int(d.client_ok)} withdrawals succeeded, so it thinks the balance is ${money(d.client_believes_balance)}. The server holds ${money(d.final_balance)}.`,
      draw(host, d) {
        host.innerHTML = `<div class="tbl-wrap"><table><thead><tr><th class="num">Call</th><th class="num">Attempts</th><th class="num">Debits</th><th class="num">Answered from table</th><th class="num">Late replies</th><th>Client result</th><th class="num">Reply</th></tr></thead><tbody>`
          + d.per_call.map(c => `<tr class="${c.executions > 1 ? 'bds-bad' : c.status === 'unknown' ? 'bds-warn' : ''}"><td class="num">${c.call}</td><td class="num">${c.attempts}</td><td class="num">${c.executions}</td><td class="num">${c.dedup}</td><td class="num">${c.late}</td><td>${esc(c.status)}</td><td class="num">${c.reply == null ? '—' : money(c.reply)}</td></tr>`).join('')
          + '</tbody></table></div>';
      } },
    { id: 'compare', kind: 'html', title: 'The same network, three policies',
      sub: d => `Identical losses and delays (seed ${d.params.seed}); only the client's policy and the server's table change.`,
      draw(host, d) {
        const rows = [['Semantics', p => d.compare[p].semantics], ['Final balance', p => money(d.compare[p].final_balance)], ['Expected', p => money(d.compare[p].expected_balance)],
          ['Off by', p => signed(d.compare[p].difference)], ['Double debits', p => int(d.compare[p].double_debits)], ['Withdrawals that never ran', p => int(d.compare[p].missed)],
          ['Outcome unknown to the client', p => int(d.compare[p].unknown)], ['Messages', p => int(d.compare[p].messages)], ['Bytes', p => int(d.compare[p].bytes)]];
        const pols = Object.keys(d.compare);
        host.innerHTML = `<div class="tbl-wrap"><table><thead><tr><th></th>${pols.map(p => `<th class="num">${esc(NAME[p])}</th>`).join('')}</tr></thead><tbody>`
          + rows.map(([k, f]) => `<tr><td>${k}</td>${pols.map(p => `<td class="num${p === d.policy ? ' best' : ''}">${f(p)}</td>`).join('')}</tr>`).join('')
          + `<tr class="${''}"><td>Balance correct?</td>${pols.map(p => `<td class="num">${d.compare[p].difference === 0 ? '<span class="chip good">yes</span>' : '<span class="chip bad">no</span>'}</td>`).join('')}</tr>`
          + '</tbody></table></div>';
      } },
    { id: 'sweep', title: 'Wrong balances as the network gets worse', height: 270,
      sub: d => `Average over ${d.sweep.seeds} seeds at each loss rate, other settings as now. Solid: double debits per run. Dashed: withdrawals that never ran.`,
      legend: d => Object.keys(NAME).map(p => [COL[p], 'line', NAME[p]]),
      draw(host, d) {
        const S = d.sweep, hi = Math.max(1, ...Object.keys(NAME).flatMap(p => [...S[p].double_debits, ...S[p].missed]));
        const P = frame(host, { h: 270, x: [0, S.loss[S.loss.length - 1]], y: [0, hi * 1.1], xl: 'Message loss', yl: 'Withdrawals per run', xf: v => Math.round(v * 100) + '%' });
        Object.keys(NAME).forEach(p => { P.line(zip(S.loss, S[p].double_debits), COL[p]); P.line(zip(S.loss, S[p].missed), COL[p] + ' thin dashed'); });
        P.vline(d.params.loss);
        hover(host, P, px => {
          const i = Math.max(0, Math.min(S.loss.length - 1, Math.round(P.xInv(px) / 0.05)));
          return { cross: P.X(S.loss[i]), html: tip(`Loss ${pct(S.loss[i], 0)}`, Object.keys(NAME).map(p => [NAME[p], `${fmt(S[p].double_debits[i], 2)} double, ${fmt(S[p].missed[i], 2)} missing`])) };
        });
      } },
    { id: 'marshal', kind: 'html', title: 'Marshalling the arguments',
      sub: d => `withdraw(${d.marshal.account}, ${d.marshal.amount}) as bytes. Client and server must agree on the byte order, or the integer arrives as a different number.`,
      draw(host, d) {
        const M = d.marshal, bytes = (a, hi) => `<span class="bds-bytes">${a.map((b, i) => `<span class="${hi(i) ? 'hi' : ''}">${b}</span>`).join('')}</span>`;
        host.innerHTML = `<div class="tbl-wrap"><table><thead><tr><th>Value</th><th>Encoding</th><th>Bytes on the wire</th><th class="num">Read back with the wrong order</th></tr></thead><tbody>`
          + `<tr><td>account = ${M.account}</td><td>int32 big-endian (network order)</td><td>${bytes(M.int32.be, i => i === 3)}</td><td class="num" rowspan="2">${int(M.int32.misread)}</td></tr>`
          + `<tr><td></td><td>int32 little-endian (x86)</td><td>${bytes(M.int32.le, i => i === 0)}</td></tr>`
          + `<tr><td>amount = ${M.amount}</td><td>float64 big-endian (IEEE 754)</td><td>${bytes(M.float64.be, i => i < 2)}</td><td class="num" rowspan="2">${M.float64.misread.toExponential(4)}</td></tr>`
          + `<tr><td></td><td>float64 little-endian</td><td>${bytes(M.float64.le, i => i > 5)}</td></tr>`
          + `<tr><td></td><td>float32 big-endian</td><td>${bytes(M.float32.be, () => false)}</td><td class="num">reads back as ${M.float32.roundtrip}</td></tr>`
          + `<tr><td>both</td><td>JSON text (UTF-8)</td><td class="bds-hex">${esc(M.json)}</td><td class="num">${M.json_bytes} B against ${M.binary_bytes} B binary</td></tr>`
          + '</tbody></table></div>'
          + `<p class="cap">float64 fields: sign ${M.float64.sign}, exponent 2<sup>${M.float64.exponent}</sup>, mantissa 0x${M.float64.mantissa_hex}. The highlighted byte is the one holding the low-order bits of the integer.</p>`;
      } },
  ],
  insights: d => {
    const c = d.compare, out = [];
    out.push(`On this network a request and its reply both survive with probability (1 − ${pct(d.params.loss, 0)})² = <b>${pct((1 - d.params.loss) ** 2, 0)}</b>. ${int(d.lost)} of ${int(d.messages)} messages were lost and ${int(d.retries)} retries were sent.`);
    out.push(`<b>Retry without a key</b> ends at ${money(c.retry.final_balance)} (${signed(c.retry.difference)}): ${int(c.retry.double_debits)} withdrawal${c.retry.double_debits === 1 ? ' was' : 's were'} debited twice because the reply, not the request, was lost. <b>With an idempotency key</b> the server answered ${int(c.idempotent.dedup_hits)} repeat${c.idempotent.dedup_hits === 1 ? '' : 's'} from its table and ends at ${money(c.idempotent.final_balance)} (${signed(c.idempotent.difference)}).`);
    out.push(`<b>No retry</b> never resends, yet ${int(c.none.unknown)} call${c.none.unknown === 1 ? '' : 's'} ended with the client not knowing whether the money left (${signed(c.none.difference)} against the expected balance)` + (c.none.double_debits ? `, and a network duplicate still ran ${int(c.none.double_debits)} withdrawal${c.none.double_debits === 1 ? '' : 's'} twice: only the server's table gives true at-most-once.` : '.'));
    if (d.late_replies) out.push(`${int(d.late_replies)} repl${d.late_replies === 1 ? 'y' : 'ies'} arrived after the client had moved on and were ignored: a timeout shorter than the round trip (about ${fmt(d.round_trip_ms, 0)} ms here) causes needless retries.`);
    return out;
  },
  maths: { eq: 'P(call answered in one try) = (1 − p)²\nP(still unanswered after k tries) = (1 − (1 − p)²)ᵏ\nretry, no key:  debits per call = number of requests that reach the server\nidempotency key: debits per call = 1 if any request arrives, else 0',
    notes: ['p is the loss rate per message. A lost reply looks exactly like a lost request to the client; only the server can tell them apart, which is why the fix (remembering replies by key) lives on the server.',
      'This is a simulation of RPC semantics with a seeded random network, not real sockets. The slide\'s client calls <code>deposit</code> by mistake; here the client calls <code>withdraw</code>.'] },
};

export default [rpc];
