// Big Data Lab · Spark SQL: join strategies, the Catalyst optimiser, and tables on disk.
// The server plans, executes and measures everything (GET /api/bigdata/sparksql/...); labels and
// defaults come from /api/catalog → bigdata_sparksql. The browser only formats and draws.
import { call, fmt, int, pct } from '../../api.js';
import { catalog } from '../../catalog.js';
import { E, frame, hover } from '../../plot.js';
import { edge, node, text, widthFor } from '../../ui/illustrate.js';

const C = catalog.bigdata_sparksql;
const J = C.joins, Q = C.sql, T = C.tables;
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const tip = (title, rows) => `<b>${esc(title)}</b>` + rows.map(([k, v]) => `<br>${esc(k)}: <span class="n">${esc(v)}</span>`).join('');
const get = (path, s, sig, keys) => call(path, Object.fromEntries(keys.map(k => [k, s[k]])), { signal: sig });

/* ---------------------------------------------------------------- formatting only */
const bytes = b => {
  if (b == null) return '—';
  const u = [['TB', 1024 ** 4], ['GB', 1024 ** 3], ['MB', 1024 ** 2], ['KB', 1024]].find(([, s]) => Math.abs(b) >= s);
  if (!u) return `${Math.round(b)} B`;
  const v = b / u[1];
  return `${v < 100 ? v.toFixed(1) : Math.round(v)} ${u[0]}`;
};
const big = n => n == null ? '—' : n >= 1e12 ? (n / 1e12).toPrecision(3) + ' trillion' : n >= 1e9 ? (n / 1e9).toPrecision(3) + ' billion' : n >= 1e6 ? (n / 1e6).toPrecision(3) + ' million' : int(n);
const plural = (n, w) => `${int(n)} ${w}${n === 1 ? '' : 's'}`;
const secs = s => s == null ? '—' : s < 0.001 ? '< 1 ms' : s < 1 ? `${Math.round(s * 1000)} ms` : s < 120 ? `${s.toFixed(1)} s` : s < 7200 ? `${Math.round(s / 60)} min`
  : s < 172800 ? `${(s / 3600).toFixed(1)} h` : s < 63072000 ? `${Math.round(s / 86400)} days` : `${Math.round(s / 31536000).toLocaleString('en-US')} years`;
const cell = v => v == null ? '<span class="sq-null">null</span>' : typeof v === 'number' ? `<span class="num">${esc(Number.isInteger(v) ? v : +v.toFixed(2))}</span>` : esc(v);
function mini(cols, rows, { total, cls = '' } = {}) {
  const more = total != null && total > rows.length ? `<tr><td colspan="${cols.length}" class="sq-more">… ${int(total - rows.length)} more</td></tr>` : '';
  const body = rows.length ? rows.map(r => `<tr>${r.map(v => `<td>${cell(v)}</td>`).join('')}</tr>`).join('') : `<tr><td colspan="${cols.length}" class="sq-empty">no rows</td></tr>`;
  return `<table class="sq-mini ${cls}"><thead><tr>${cols.map(c => `<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>${body}${more}</tbody></table>`;
}
const treeLi = n => `<li><span class="sq-node${n.tag ? ' sq-' + n.tag : ''}"><b>${esc(n.label)}</b>${n.detail ? ` <code>${esc(n.detail)}</code>` : ''}</span>${n.children?.length ? `<ul>${n.children.map(treeLi).join('')}</ul>` : ''}</li>`;
const tree = n => `<ul class="sq-tree">${treeLi(n)}</ul>`;
const chip = (ok, text) => `<span class="chip ${ok ? 'good' : 'bad'}">${esc(text)}</span>`;

/* ================================================================ 1 · Join strategies */
const SHORT = Object.fromEntries(Object.entries(J.strategies).map(([k, v]) => [k, v.short]));
const isSyn = s => s.source === 'synthetic';

const joins = {
  id: 'spark-joins', group: 'bigdata', nav: 'Spark joins', tag: 'Join strategies', title: 'How Spark chooses and runs a join',
  question: 'Customers ⋈ orders: which of Spark\'s join strategies does the planner pick, what does each one move across the network, and where does it break?',
  params: [
    { k: 'source', type: 'seg', label: 'Tables', value: 'course', group: 'Data', options: [['course', 'RDD deck'], ['mr', 'MR deck'], ['synthetic', 'Scale-up']],
      help: `RDD deck p40–41: ${J.source_rows.course[0]} customers, ${J.source_rows.course[1]} orders. MapReduce deck 2 p12: ${J.source_rows.mr[0]} customers, ${J.source_rows.mr[1]} orders. Scale-up: synthetic tables of any size, with a sample joined for real.` },
    { k: 'join', type: 'seg', label: 'Join type', value: 'inner', group: 'Query', options: [['inner', 'Inner'], ['left', 'Left outer'], ['full', 'Full outer']],
      help: 'customers c <type> JOIN orders o. A left join keeps customers without orders; a full outer join also keeps orders without a customer.' },
    { k: 'cond', type: 'seg', label: 'Condition', value: 'eq', group: 'Query', options: [['eq', 'c.id = o.id'], ['band', 'BETWEEN ±1']],
      help: 'An equality lets Spark hash or sort on the key. The range condition o.cust_id BETWEEN c.cust_id − 1 AND c.cust_id + 1 forces a nested loop.' },
    { k: 'hint', type: 'seg', label: 'Join hint', value: 'none', group: 'Query', options: [['none', 'none'], ['broadcast', 'BROADCAST'], ['shuffle_hash', 'SHUFFLE_HASH'], ['merge', 'MERGE'], ['shuffle_replicate_nl', 'REPLICATE_NL']],
      help: 'SQL hints (/*+ BROADCAST(c) */ …) are tried before any size rule. A hint the join type cannot honour is ignored. Beyond the slides.' },
    { k: 'layout', type: 'seg', label: 'Table layout', value: 'plain', group: 'Query', options: [['plain', 'Plain files'], ['bucketed', 'Bucketed by cust_id']],
      help: 'Bucketed: both tables were written CLUSTERED BY (cust_id) INTO n BUCKETS with n = shuffle partitions, so equal keys are already together: the deck\'s "map-side" (co-partitioned) join.' },
    { k: 'threshold', label: 'autoBroadcastJoinThreshold', min: -1, max: 100, step: 1, value: J.threshold_default_mb, unit: 'MB', group: 'Spark settings',
      help: `Spark broadcasts a side whose estimated size is at most this. The default is ${J.threshold_default_mb} MB; −1 turns auto-broadcast off.` },
    { k: 'partitions', label: 'spark.sql.shuffle.partitions', min: 1, max: 400, step: 1, value: 8, group: 'Spark settings',
      help: `Partitions after a shuffle (and buckets for the bucketed layout). Spark's default is ${J.shuffle_partitions_default}; 8 keeps the course tables readable.` },
    { k: 'executors', label: 'Executors', min: 1, max: 64, step: 1, value: 4, group: 'Spark settings', help: 'Each executor receives one copy of a broadcast table.' },
    { k: 'prefer_smj', type: 'switch', label: 'preferSortMergeJoin', value: true, group: 'Spark settings',
      help: 'On (the default), Spark skips shuffle hash join unless hinted. Off, it uses shuffle hash when one side is 3× smaller and fits a hash map per partition.' },
    { k: 'aqe', type: 'switch', label: 'AQE skew-join split', value: false, group: 'Spark settings',
      help: `Adaptive execution splits a partition over ${J.skew.factor}× the median and over ${bytes(J.skew.min_bytes)} into pieces, reading the other side's partition once per piece. Beyond the slides.` },
    { k: 'lrows', label: 'Customers, 10^x rows', min: 2, max: 9, step: 0.1, value: 5, group: 'Scale-up', show: isSyn, help: 'The left (dimension) table: one row per customer id.' },
    { k: 'lbytes', label: 'Customer row width', min: 8, max: 1024, step: 4, value: 100, unit: 'B', group: 'Scale-up', show: isSyn },
    { k: 'rrows', label: 'Orders, 10^x rows', min: 2, max: 10, step: 0.1, value: 8, group: 'Scale-up', show: isSyn, help: 'The right (fact) table.' },
    { k: 'rbytes', label: 'Order row width', min: 8, max: 1024, step: 4, value: 50, unit: 'B', group: 'Scale-up', show: isSyn },
    { k: 'skew', label: 'Key skew (Zipf s)', min: 0, max: 2.5, step: 0.1, value: 0, group: 'Scale-up', show: isSyn,
      help: 'Orders per customer follow Zipf(s): 0 is uniform; at 1.2 a handful of customers own most orders and their shuffle partition becomes the straggler.' },
  ],
  presets: [
    { label: 'Course tables: broadcast', values: { source: 'course', join: 'inner', cond: 'eq', hint: 'none', layout: 'plain', threshold: 10, partitions: 8 } },
    { label: 'Threshold −1: sort-merge', values: { source: 'course', join: 'inner', cond: 'eq', hint: 'none', layout: 'plain', threshold: -1, partitions: 8 } },
    { label: 'Left join: customers with no order', values: { source: 'course', join: 'left', cond: 'eq', hint: 'none', layout: 'plain', threshold: -1, partitions: 8 } },
    { label: 'Full outer: no broadcast', values: { source: 'course', join: 'full', cond: 'eq', hint: 'none', layout: 'plain', threshold: 10, partitions: 8 } },
    { label: 'SHUFFLE_HASH hint', values: { source: 'course', join: 'inner', cond: 'eq', hint: 'shuffle_hash', layout: 'plain', threshold: 10, partitions: 8 } },
    { label: 'Co-partitioned (bucketed)', values: { source: 'course', join: 'inner', cond: 'eq', hint: 'none', layout: 'bucketed', threshold: -1, partitions: 8 } },
    { label: 'Non-equi: nested loop', values: { source: 'course', join: 'inner', cond: 'band', hint: 'none', layout: 'plain', threshold: 10, partitions: 8 } },
    { label: 'Non-equi, threshold −1: Cartesian', values: { source: 'course', join: 'inner', cond: 'band', hint: 'none', layout: 'plain', threshold: -1, partitions: 8 } },
    { label: `Scale-up at the ${J.threshold_default_mb} MB line`, values: { source: 'synthetic', join: 'inner', cond: 'eq', hint: 'none', layout: 'plain', threshold: 10, lrows: 5, lbytes: 100, rrows: 8, skew: 0, partitions: 200, aqe: false } },
    { label: 'Skewed keys: a straggler', values: { source: 'synthetic', join: 'inner', cond: 'eq', hint: 'none', layout: 'plain', threshold: -1, lrows: 6, lbytes: 100, rrows: 9, skew: 1.2, partitions: 200, aqe: false } },
    { label: 'Skew with AQE split', values: { source: 'synthetic', join: 'inner', cond: 'eq', hint: 'none', layout: 'plain', threshold: -1, lrows: 6, lbytes: 100, rrows: 9, skew: 1.2, partitions: 200, aqe: true } },
  ],
  request: (s, sig) => get('/api/bigdata/sparksql/joins', s, sig, ['source', 'join', 'cond', 'hint', 'layout', 'threshold', 'partitions', 'executors', 'prefer_smj', 'aqe', 'lrows', 'lbytes', 'rrows', 'rbytes', 'skew']),
  kpis: d => {
    const m = d.metrics, p = d.plan, st = m.strategy, shuffle = d.partitions.shuffle_based;
    const nl = st === 'bnlj' || st === 'cartesian';
    return [
      { label: 'Strategy', value: SHORT[st], note: p.build ? `builds the ${p.build === 'left' ? 'customers' : 'orders'} side` : (p.exchange ? 'both sides shuffled' : st === 'copart' ? 'no Exchange: tables bucketed' : 'pairs of partitions'), tone: 'good' },
      { label: 'Data moved', value: bytes(m.moved_bytes), note: m.broadcast_bytes ? `${bytes(m.broadcast_bytes / d.settings.executors)} × ${d.settings.executors} executors broadcast` : m.shuffle_bytes ? `shuffled${m.extra_read_bytes ? ` + ${bytes(m.extra_read_bytes)} re-read by AQE` : ''}` : 'nothing crosses the network', tone: m.moved_bytes > 1024 ** 3 ? 'warn' : null },
      { label: 'Peak memory per task', value: bytes(m.peak_mem_bytes), note: st === 'bhj' || st === 'bnlj' ? 'the broadcast table, on every executor' : st === 'shj' ? 'largest hash table of one partition' : st === 'cartesian' ? 'one partition of each side' : 'largest partition to sort (can spill)', tone: m.peak_mem_bytes > 2 * 1024 ** 3 ? 'bad' : null },
      nl ? { label: 'Comparisons', value: big(m.nl_comparisons), note: `every customer × every order · ≈ ${secs(m.est_seconds)} modelled`, tone: m.nl_comparisons > 1e12 ? 'bad' : 'warn' }
        : { label: shuffle ? 'Straggler partition' : 'Task balance', value: `${fmt(m.straggler_ratio, m.straggler_ratio >= 10 ? 0 : 1)}×`, note: shuffle ? `partition ${d.partitions.straggler} vs the median${m.skew_splits ? ` · AQE added ${m.skew_splits} splits` : ''}` : `${int(m.tasks)} tasks · stream side split evenly`, tone: m.straggler_ratio > 3 ? 'warn' : 'good' },
    ];
  },
  illustration: {
    title: 'What moves where', height: 220,
    caption: 'Arrow widths follow the bytes each side sends across the network (log scale); a thin grey arrow means the data stays where it is.',
    draw(svg, d, s) {
      const st = d?.metrics.strategy, E = s.executors;
      const lb = d?.tables.left.bytes, rb = d?.tables.right.bytes;
      const L = node(svg, { x: 6, y: 18, w: 128, h: 62, title: 'customers', sub: d ? `${big(d.tables.left.rows)} rows` : '', value: d ? bytes(lb) : null });
      const R = node(svg, { x: 6, y: 132, w: 128, h: 62, title: 'orders', sub: d ? `${big(d.tables.right.rows)} rows` : '', value: d ? bytes(rb) : null });
      if (!d) return;
      const mid = { bhj: 'Broadcast', bnlj: 'Broadcast', shj: 'Exchange', smj: 'Exchange', copart: 'Bucket files', cartesian: 'Replicate' }[st];
      const M = node(svg, { x: 214, y: 70, w: 124, h: 72, title: mid, sub: SHORT[st], hot: true });
      const w = b => b > 0 ? widthFor(Math.log10(1 + b), 13) : 1;
      const build = d.plan.build;
      const moveL = st === 'bhj' || st === 'bnlj' ? (build === 'left' ? lb * E : 0) : st === 'copart' ? 0 : st === 'cartesian' ? d.metrics.shuffle_bytes / 2 : lb;
      const moveR = st === 'bhj' || st === 'bnlj' ? (build === 'right' ? rb * E : 0) : st === 'copart' ? 0 : st === 'cartesian' ? d.metrics.shuffle_bytes / 2 : rb;
      edge(svg, [L.r + 4, L.cy], [M.x - 6, M.cy - 14], { width: w(moveL), label: moveL ? bytes(moveL) : 'stays', labelAt: [176, 46], cls: moveL ? '' : 'soft', flow: moveL > 0 });
      edge(svg, [R.r + 4, R.cy], [M.x - 6, M.cy + 14], { width: w(moveR), label: moveR ? bytes(moveR) : 'stays', labelAt: [176, 180], cls: moveR ? '' : 'soft', flow: moveR > 0 });
      const shown = Math.min(4, E);
      for (let i = 0; i < shown; i++) {
        const y = 12 + i * (196 / shown);
        const X = node(svg, { x: 400, y, w: 96, h: Math.min(40, 196 / shown - 8), title: `executor ${i}`, sub: '' });
        edge(svg, [M.r + 4, M.cy], [X.x - 6, X.cy], { width: 2 });
      }
      if (E > shown) text(svg, 448, 214, `+ ${E - shown} more`, 'n-sub');
      const O = node(svg, { x: 520, y: 78, w: 76, h: 56, title: 'result', value: int(d.result.total), sub: 'rows' });
      edge(svg, [496, O.cy], [O.x - 6, O.cy], { width: 2 });
    },
  },
  charts: [
    { id: 'plan', kind: 'html', title: 'The planner\'s decision', sub: d => esc(d.label) + ' · ' + esc(d.scale_note),
      draw(host, d) {
        const p = d.plan;
        host.innerHTML = `<div class="sq-split"><div><p class="eyebrow">Query</p><pre class="sq-code">${esc(d.sql)}</pre>
          <p class="eyebrow">Checks, in the order JoinSelection tries them</p><ol class="sq-checks">${p.checks.map(c => `<li class="${c.ok ? 'ok' : 'no'}"><b>${esc(c.rule)}</b> ${esc(c.detail)}</li>`).join('')}</ol>
          <p class="sq-why">Chosen: <b>${esc(p.label)}</b>, because ${esc(p.why)}.</p></div>
          <div><p class="eyebrow">Physical plan (as EXPLAIN prints it)</p>${tree(p.tree)}
          <p class="eyebrow">Inputs</p>${mini(['table', 'rows', 'row width', 'size estimate'], [['customers', d.tables.left.rows, `${d.tables.left.row_bytes} B`, bytes(d.tables.left.bytes)], ['orders', d.tables.right.rows, `${d.tables.right.row_bytes} B`, bytes(d.tables.right.bytes)]])}
          <p class="c-note">Threshold: ${d.threshold_bytes < 0 ? 'auto-broadcast off (−1)' : bytes(d.threshold_bytes)} · ${d.settings.partitions} shuffle partitions · ${d.settings.executors} executors</p></div></div>`;
      } },
    { id: 'steps', kind: 'html', title: 'Step by step, partition by partition', sub: d => d.source === 'synthetic' ? 'The sample joined for real; the first partitions and the first rows of each are shown.' : 'Every row of the course tables, at every stage.',
      draw(host, d) {
        const lc = d.tables.left.cols, rc = d.tables.right.cols, oc = d.result.cols;
        const part = p => `<div class="sq-part"><p class="sq-part-h">${esc(p.label)}${p.comparisons != null ? ` <span class="sq-cmp">${int(p.comparisons)} ${d.metrics.strategy === 'bhj' ? 'probes' : 'comparisons'}</span>` : ''}</p>`
          + (p.left ? `<p class="sq-lbl">customers (${p.left.n})</p>${mini(lc, p.left.rows, { total: p.left.n })}` : '')
          + (p.right ? `<p class="sq-lbl">orders (${p.right.n})</p>${mini(rc, p.right.rows, { total: p.right.n })}` : '')
          + (p.stream ? `<p class="sq-lbl">streamed rows (${p.stream.n})</p>${mini(p.stream_side === 'left' ? lc : rc, p.stream.rows, { total: p.stream.n })}` : '')
          + (p.out ? `<p class="sq-lbl">output (${p.out.n})</p>${mini(oc, p.out.rows, { total: p.out.n, cls: 'out' })}` : '') + '</div>';
        host.innerHTML = d.steps.map((st, i) => {
          let body = '';
          if (st.build_table) body += `<div class="sq-bcast"><div><p class="sq-lbl">${st.build === 'left' ? 'customers' : 'orders'}: ${st.build_table.n} rows${d.metrics.strategy === 'bhj' ? ', hashed on cust_id' : ''}</p>${mini(st.build === 'left' ? lc : rc, st.build_table.rows, { total: st.build_table.n })}</div><p class="sq-copies">× ${st.copies} copies, one per executor</p></div>`;
          if (st.kind === 'exchange' && d.key_map.length) body += `<details class="more"><summary>Routing table: pmod(murmur3(cust_id), ${d.settings.partitions})</summary>${mini(['cust_id', 'murmur3 hash', 'partition'], d.key_map.map(k => [k.key, k.hash, k.partition]))}</details>`;
          if (st.parts) body += `<div class="sq-parts">${st.parts.map(part).join('')}</div>${st.parts_total > st.parts.length ? `<p class="c-note">… and ${st.parts_total - st.parts.length} more partitions.</p>` : ''}`;
          if (st.out) body += mini(oc, st.out.rows, { total: st.out.n, cls: 'out' });
          return `<section class="sq-step"><h4><span class="sq-num">${i + 1}</span>${esc(st.title)}</h4><p class="c-note">${esc(st.note)}</p>${body}</section>`;
        }).join('');
      } },
    { id: 'parts', title: 'Rows per task at full size', height: 270,
      sub: d => d.partitions.shuffle_based ? `${d.partitions.n} shuffle partitions${d.partitions.shown < d.partitions.n ? ` (first ${d.partitions.shown} shown)` : ''}: customers and orders that hash to each one. The outlined bar is the straggler.` : `${int(d.partitions.tasks)} tasks over the streamed side; each also holds the whole broadcast table.`,
      legend: d => d.partitions.shuffle_based ? [['s1', 'box', 'customers'], ['s2', 'box', 'orders'], ['neutral', 'line', 'median']] : [['s3', 'box', 'rows streamed per task']],
      draw(host, d) {
        const P0 = d.partitions;
        if (P0.shuffle_based) {
          const n = P0.left.length, tot = P0.left.map((v, i) => v + P0.right[i]), mx = Math.max(1, ...tot);
          const sorted = [...tot].sort((a, b) => a - b), med = sorted[Math.floor(n / 2)];
          const P = frame(host, { h: 270, x: [-0.6, n - 0.4], y: [0, mx * 1.1], xl: 'Shuffle partition', yl: 'Rows', yf: v => big(v), xn: Math.min(10, n) });
          const bw = Math.max(1, (P.X(1) - P.X(0)) * 0.8);
          for (let i = 0; i < n; i++) {
            const x = P.X(i) - bw / 2;
            E('rect', { x, width: bw, y: P.Y(P0.left[i]), height: Math.max(0, P.Y(0) - P.Y(P0.left[i])), class: 'f1', rx: 1 }, P.body);
            E('rect', { x, width: bw, y: P.Y(tot[i]), height: Math.max(0, P.Y(P0.left[i]) - P.Y(tot[i])), class: 'f2', rx: 1 }, P.body);
          }
          const s = P0.straggler;
          if (s < n) E('rect', { x: P.X(s) - bw / 2 - 2, width: bw + 4, y: P.Y(tot[s]) - 2, height: P.Y(0) - P.Y(tot[s]) + 2, class: 'sq-strag' }, P.body);
          P.hline(med, 'ref');
          hover(host, P, px => { const i = Math.max(0, Math.min(n - 1, Math.round(P.xInv(px)))); return { cross: P.X(i), html: tip(`Partition ${i}`, [['customers', big(P0.left[i])], ['orders', big(P0.right[i])], ['bytes', bytes(P0.left_bytes[i] + P0.right_bytes[i])], ['vs median', `${fmt(tot[i] / Math.max(med, 1e-9), 1)}×`]]) }; });
        } else {
          const t = P0.task_rows, n = t.length, mx = Math.max(1, ...t);
          const P = frame(host, { h: 240, x: [-0.6, n - 0.4], y: [0, mx * 1.15], xl: 'Task', yl: 'Streamed rows', yf: v => big(v), xn: Math.min(10, n) });
          const bw = Math.max(1, (P.X(1) - P.X(0)) * 0.8);
          t.forEach((v, i) => E('rect', { x: P.X(i) - bw / 2, width: bw, y: P.Y(v), height: Math.max(0, P.Y(0) - P.Y(v)), class: 'f3', rx: 1 }, P.body));
          hover(host, P, px => { const i = Math.max(0, Math.min(n - 1, Math.round(P.xInv(px)))); return { cross: P.X(i), html: tip(`Task ${i}`, [['rows streamed', big(t[i])]]) }; });
        }
      } },
    { id: 'compare', kind: 'html', title: 'Every strategy on these tables', sub: `Computed for the full table sizes; the time is a simple cost model (${J.cost.net_bps * 8 / 1e9} Gbit/s per executor, fixed row and comparison rates), good for orders of magnitude only.`,
      draw(host, d) {
        const rows = d.compare.map(c => `<tr class="${c.chosen ? 'best' : ''}${c.feasible ? '' : ' sq-off'}"><td>${esc(c.label)}${c.chosen ? ' ✓' : ''}</td>`
          + (c.feasible || c.chosen ? `<td class="num">${bytes(c.broadcast_bytes)}</td><td class="num">${bytes(c.shuffle_bytes + (c.extra_read_bytes || 0))}</td><td class="num">${big(c.sort_comparisons)}</td><td class="num">${big(c.nl_comparisons)}</td><td class="num">${bytes(c.peak_mem_bytes)}</td><td class="num">${fmt(c.straggler_ratio, 1)}×</td><td class="num">${secs(c.est_seconds)}</td>`
            : `<td colspan="7" class="sq-why-not">not possible: ${esc(c.reason)}</td>`) + '</tr>').join('');
        host.innerHTML = `<div class="tbl-wrap"><table><thead><tr><th>Strategy</th><th class="num">Broadcast</th><th class="num">Shuffled</th><th class="num">Sort comparisons</th><th class="num">Loop comparisons</th><th class="num">Peak memory</th><th class="num">Straggler</th><th class="num">Modelled time</th></tr></thead><tbody>${rows}</tbody></table></div>`;
      } },
    { id: 'result', kind: 'html', title: 'The joined rows, checked',
      sub: d => `${int(d.result.total)} rows${d.result.unmatched_left ? ` · ${d.result.unmatched_left} customers with no order (null-extended)` : ''}${d.result.unmatched_right ? ` · ${d.result.unmatched_right} orders with no customer` : ''}.`,
      draw(host, d) {
        const v = d.validation.map(x => x.feasible ? chip(x.same_as_reference, `${x.label}: ${int(x.rows)} rows ${x.same_as_reference ? '✓' : '✗'}`) : `<span class="chip">${esc(x.label)}: n/a</span>`).join(' ');
        host.innerHTML = `<p class="sq-chips">${v}</p><p class="c-note">Each possible strategy was executed on the same rows and compared with a direct all-pairs join.</p>${mini(d.result.cols, d.result.rows, { total: d.result.total, cls: 'out' })}`;
      } },
  ],
  chartLayout: 'one',
  insights: d => {
    const m = d.metrics, st = m.strategy, s = d.settings, L = d.tables.left, R = d.tables.right, out = [];
    if (st === 'bhj') out.push(`The ${d.plan.build === 'left' ? 'customers' : 'orders'} side is <b>${bytes(d.plan.build === 'left' ? L.bytes : R.bytes)}</b>, so Spark broadcasts it: <b>${bytes(m.broadcast_bytes)}</b> crosses the network (${s.executors} copies) and the other table never moves.`);
    if (st === 'smj' || st === 'shj') out.push(`Neither side is broadcast, so both are shuffled by cust_id: <b>${bytes(m.shuffle_bytes)}</b> written and fetched. The busiest partition holds <b>${fmt(m.straggler_ratio, 1)}×</b> the median rows${m.straggler_ratio > 3 ? ': the whole stage waits for it' : ''}.`);
    if (st === 'copart') out.push('Both tables are bucketed the same way, so Spark reads bucket i of each side in the same task: <b>no Exchange, 0 bytes shuffled</b>. Inside each task it is still a sort-merge join; Spark SQL has no separate "map-side join" operator.');
    if (st === 'bnlj') out.push(`A range condition has no key to hash, so every streamed row is compared with every broadcast row: <b>${big(m.nl_comparisons)}</b> comparisons for ${big(L.rows)} × ${big(R.rows)} rows.`);
    if (st === 'cartesian') out.push(`With nothing small enough to broadcast, the inner range join becomes a Cartesian product: ${int(m.tasks)} tasks, each comparing a pair of partitions, <b>${big(m.nl_comparisons)}</b> comparisons.`);
    if (d.result.unmatched_left) out.push(`${plural(d.result.unmatched_left, 'customer')} (cust_id ${d.result.unmatched_left_ids.join(', ')}${d.result.unmatched_left > d.result.unmatched_left_ids.length ? ', …' : ''}) ${d.result.unmatched_left === 1 ? 'has' : 'have'} no order: the ${s.join === 'full' ? 'FULL' : 'LEFT'} OUTER join keeps them with nulls in the order columns.`);
    if (d.result.unmatched_right) out.push(`${plural(d.result.unmatched_right, 'order')} point${d.result.unmatched_right === 1 ? 's' : ''} to a customer that does not exist; only the FULL OUTER join keeps ${d.result.unmatched_right === 1 ? 'it' : 'them'}.`);
    if (s.join === 'full' && st !== 'bhj') out.push('A FULL OUTER join cannot use broadcast hash: the broadcast side\'s unmatched rows would have to be found on every executor at once.');
    if (s.join === 'left' && d.plan.checks.some(c => /cannot build the left/.test(c.detail))) out.push('A LEFT OUTER join may only broadcast the <b>right</b> side (orders), even when customers is the smaller table.');
    if (m.skew_splits) out.push(`AQE split the skewed partitions into ${m.skew_splits} extra tasks, re-reading ${bytes(m.extra_read_bytes)} of the other side.`);
    out.push(d.validation.filter(v => v.feasible).every(v => v.same_as_reference) ? `All ${d.validation.filter(v => v.feasible).length} possible strategies return the same ${int(d.result.total)} rows: the strategy changes the cost, never the answer.` : 'A strategy disagreed with the reference join (see the checks).');
    return out;
  },
  maths: { eq: 'partition(key) = pmod(murmur3(key, seed 42), n)\nbroadcast bytes = size(build side) × executors\nshuffle bytes = size(left) + size(right)\nnested-loop comparisons = |left| × |right|',
    notes: [`murmur3 is Spark's own hash, reproduced bit for bit (hash(${J.hash_check.value}) = ${J.hash_check.hash}), so the routing table shows the partition Spark would use.`,
      `Planner rules, hints, the ${bytes(J.broadcast_limit_bytes)} broadcast limit and AQE are Spark 3 behaviour beyond the slides; sizes are rows × an estimated row width.`] },
};

/* ================================================================ 2 · Spark SQL and the optimiser */
const QOPT = Object.entries(Q.queries).map(([k, v]) => [k, v.label]);
const hasJoin = s => s.query === 'join_agg' || s.query === 'slide16';
const VSHORT = { 'No optimisation': 'none', 'ConstantFolding only': 'fold', 'PushDownPredicates only': 'push', 'ColumnPruning only': 'prune', 'All three rules': 'all three', 'Your selection': 'yours' };

const sql = {
  id: 'sql', group: 'bigdata', nav: 'Spark SQL', tag: 'Catalyst', title: 'From a query to a physical plan',
  question: 'What does Spark SQL do between the query you write and the rows you get back, and how much reading does each optimiser rule save?',
  params: [
    { k: 'query', type: 'seg', label: 'Query', value: 'slide16', group: 'Query', options: QOPT, help: 'The course queries: DataFrame deck p19–24, the SQL intro deck p16 and the DDL deck\'s letterGrade UDF (p39–41).' },
    { k: 'k', label: 'Salary threshold (× 1000)', min: 0, max: 100, step: 1, value: 30, group: 'Query', show: s => s.query === 'filter' || s.query === 'filter_and',
      help: 'Written as k * 1000 in the query so that ConstantFolding has a constant expression to fold.' },
    { k: 'dno', type: 'seg', label: 'Department', value: '5', group: 'Query', options: [['1', '1'], ['4', '4'], ['5', '5']], show: s => s.query === 'filter_and' || s.query === 'project' },
    { k: 'gender', type: 'seg', label: 'Gender filter', value: 'F', group: 'Query', options: [['F', 'F'], ['M', 'M']], show: s => s.query === 'slide16',
      help: 'The slide filters gender === "female"; the course files code gender as M/F.' },
    { k: 'folding', type: 'switch', label: 'ConstantFolding', value: true, group: 'Optimiser rules', help: 'Evaluates expressions made only of literals once, at planning time.' },
    { k: 'pushdown', type: 'switch', label: 'PushDownPredicates', value: true, group: 'Optimiser rules', help: 'Moves filters below joins and hands simple ones (column vs literal) to the file scan.' },
    { k: 'pruning', type: 'switch', label: 'ColumnPruning', value: true, group: 'Optimiser rules', help: 'Reads only the columns the query uses; with Parquet the others are never touched.' },
    { k: 'broadcast', type: 'switch', label: `Auto-broadcast (${J.threshold_default_mb} MB)`, value: true, group: 'Physical planning', show: hasJoin, help: 'Off sets autoBroadcastJoinThreshold = −1, so the join becomes a sort-merge join with two shuffles.' },
  ],
  presets: [
    { label: 'p19: salary > 30000', values: { query: 'filter', k: 30, folding: true, pushdown: true, pruning: true } },
    { label: 'p19: ≥ 40000 AND dno = 5', values: { query: 'filter_and', k: 40, dno: '5', folding: true, pushdown: true, pruning: true } },
    { label: 'p21: project, dno 4', values: { query: 'project', dno: '4', folding: true, pushdown: true, pruning: true } },
    { label: 'p20: ORDER BY dno, salary DESC', values: { query: 'order', folding: true, pushdown: true, pruning: true } },
    { label: 'p22: avg and max by dno', values: { query: 'agg', folding: true, pushdown: true, pruning: true } },
    { label: 'p24: sum by department', values: { query: 'join_agg', folding: true, pushdown: true, pruning: true, broadcast: true } },
    { label: 'Intro p16: women per dept', values: { query: 'slide16', gender: 'F', folding: true, pushdown: true, pruning: true, broadcast: true } },
    { label: 'p16 with every rule off', values: { query: 'slide16', gender: 'F', folding: false, pushdown: false, pruning: false, broadcast: true } },
    { label: 'Pushed filter skips the file', values: { query: 'filter', k: 60, folding: true, pushdown: true, pruning: true } },
    { label: 'DDL p41: letterGrade UDF', values: { query: 'udf', folding: true, pushdown: true, pruning: true } },
  ],
  request: (s, sig) => get('/api/bigdata/sparksql/query', s, sig, ['query', 'k', 'dno', 'gender', 'folding', 'pushdown', 'pruning', 'broadcast']),
  kpis: d => {
    const m = d.metrics, sv = d.saving, join = m.rows_into_join || sv.none_join_rows;
    return [
      { label: 'Values read from files', value: int(m.cells_read), note: sv.cells ? `${int(sv.none_cells)} with no rules (−${pct(sv.cells_pct, 0)})` : `${int(sv.none_cells)} with no rules`, tone: sv.cells > 0 ? 'good' : null },
      join ? { label: 'Rows entering the join', value: int(m.rows_into_join), note: sv.join_rows ? `${int(sv.none_join_rows)} without pushdown` : 'no filter to push below it', tone: sv.join_rows > 0 ? 'good' : null }
        : { label: 'Rows read', value: int(m.rows_read), note: m.row_groups_skipped ? 'the row group was skipped by its min/max' : `${int(m.cols_read)} column(s) read`, tone: m.row_groups_skipped ? 'good' : null },
      { label: 'Expression evaluations', value: int(m.evals), note: sv.evals ? `${int(sv.none_evals)} without folding` : 'per-row operator and UDF calls' },
      { label: 'Result', value: plural(d.result.rows.length, 'row'), note: d.result.same_as_unoptimised ? 'same as the unoptimised plan ✓' : 'differs from the unoptimised plan ✗', tone: d.result.same_as_unoptimised ? 'good' : 'bad' },
    ];
  },
  illustration: {
    title: 'The Catalyst pipeline', height: 200,
    caption: 'A rule box is lit when it changed the plan for this query; switched-off rules are dimmed.',
    draw(svg, d, s) {
      const q = node(svg, { x: 4, y: 70, w: 92, h: 60, title: 'Query', sub: s.query === 'udf' ? 'SQL + UDF' : 'SQL / DataFrame' });
      const a = node(svg, { x: 116, y: 70, w: 96, h: 60, title: 'Analysed', sub: 'logical plan' });
      edge(svg, [q.r + 3, q.cy], [a.x - 5, a.cy], { width: 2 });
      const fired = d ? Object.fromEntries(d.stages.map(x => [x.rule, x.fired])) : {};
      Q.rules.forEach((r, i) => {
        const on = s[r.key], n = node(svg, { x: 236, y: 8 + i * 64, w: 150, h: 52, title: r.name, sub: !on ? 'off' : fired[r.key] ? 'changed the plan' : 'nothing to do', hot: on && fired[r.key] });
        if (!on) n && svg.lastChild.setAttribute('opacity', '0.45');
        edge(svg, [a.r + 3, a.cy], [n.x - 5, n.cy], { width: on ? 2.5 : 1, cls: on ? '' : 'soft', arrow: true });
        edge(svg, [n.r + 3, n.cy], [418, 100], { width: on ? 2.5 : 1, cls: on ? '' : 'soft', arrow: false });
      });
      const p = node(svg, { x: 420, y: 70, w: 92, h: 60, title: 'Physical', sub: d && hasJoin(s) ? (s.broadcast ? 'broadcast join' : 'sort-merge') : 'plan' });
      const r = node(svg, { x: 528, y: 70, w: 68, h: 60, title: 'Rows', value: d ? int(d.result.rows.length) : null });
      edge(svg, [p.r + 3, p.cy], [r.x - 5, r.cy], { width: 2 });
    },
  },
  charts: [
    { id: 'query', kind: 'html', title: 'The query and its tables', sub: d => `${esc(d.query.label)} · ${esc(d.query.slide)}`,
      draw(host, d) {
        const t = Object.entries(d.tables).map(([name, tb]) => `<div><p class="sq-lbl">${esc(name)} (${tb.rows.length} rows)</p>${mini(tb.cols.map((c, i) => `${c}: ${tb.types[i]}`), tb.rows)}</div>`).join('');
        const src = d.sources;
        host.innerHTML = `<div class="sq-split"><div><p class="eyebrow">SQL</p><pre class="sq-code">${esc(d.query.sql)}</pre></div><div><p class="eyebrow">DataFrame API (as on the slide)</p><pre class="sq-code">${esc(d.query.dataframe)}</pre></div></div>
          <div class="sq-tables">${t}</div>
          <details class="more"><summary>How spark.read builds these DataFrames (DataFrame deck p11–17)</summary>
          <p class="c-note">employee_m.csv, first lines:</p><pre class="sq-code">${esc(src.csv_lines.join('\n'))}</pre>
          <div class="sq-tables"><div><p class="sq-lbl">no header option: _c0 … and every column a string</p>${mini(src.no_header.cols.map((c, i) => `${c}: ${src.no_header.types[i]}`), src.no_header.rows)}</div>
          <div><p class="sq-lbl">header = true, inferSchema = true</p>${mini(src.infer.cols.map((c, i) => `${c}: ${src.infer.types[i]}`), src.infer.rows)}</div>
          <div><p class="sq-lbl">employees.json (JSON integers are inferred as long)</p>${mini(src.json.cols.map((c, i) => `${c}: ${src.json.types[i]}`), src.json.rows)}</div></div></details>`;
      } },
    { id: 'stages', kind: 'html', title: 'The plan after each stage', sub: 'Highlighted nodes are the ones the rule just changed.',
      draw(host, d) {
        host.innerHTML = `<div class="sq-stages">${d.stages.map((st, i) => `<section class="sq-stage${st.fired ? '' : ' idle'}"><h4><span class="sq-num">${i + 1}</span>${esc(st.name)}</h4>${st.log.length ? `<ul class="sq-log">${st.log.map(l => `<li>${esc(l)}</li>`).join('')}</ul>` : ''}${tree(st.tree)}</section>`).join('')}</div>`;
      } },
    { id: 'trace', kind: 'html', title: 'Execution, operator by operator', sub: 'The optimised plan run on the course tables, bottom-up: rows in, rows out, and the first rows each operator produced.',
      draw(host, d) {
        host.innerHTML = `<div class="sq-parts">${d.trace.map(t => `<div class="sq-part"><p class="sq-part-h">${esc(t.op)} <span class="sq-cmp">${int(t.rows_in)} → ${int(t.rows_out)} rows</span></p><p class="c-note"><code>${esc(t.detail)}</code></p>${mini(t.cols, t.rows, { total: t.rows_out })}</div>`).join('')}</div>`;
      } },
    { id: 'variants', title: 'What each rule saves', height: 270, sub: 'The same query executed with no rule, each rule alone, all three, and your selection. Every variant returns the same rows.',
      legend: d => [['s1', 'box', 'values read from files'], ...(d.saving.none_join_rows ? [['s2', 'box', 'rows entering the join']] : []), ['s3', 'box', 'expression evaluations']],
      draw(host, d) {
        const V = d.variants, keys = [['cells_read', 'f1'], ...(d.saving.none_join_rows ? [['rows_into_join', 'f2']] : []), ['evals', 'f3']];
        const mx = Math.max(1, ...V.flatMap(v => keys.map(([k]) => v[k])));
        const P = frame(host, { h: 270, x: [-0.6, V.length - 0.4], y: [0, mx * 1.15], xt: V.map((_, i) => i), xf: i => VSHORT[V[i]?.label] || '', yl: 'Count' });
        const bw = Math.min(26, (P.X(1) - P.X(0)) * 0.8 / keys.length);
        V.forEach((v, i) => keys.forEach(([k, cls], j) => {
          const x = P.X(i) - keys.length * bw / 2 + j * bw;
          E('rect', { x: x + 1, width: bw - 2, y: P.Y(v[k]), height: Math.max(0, P.Y(0) - P.Y(v[k])), class: cls, rx: 2 }, P.body);
          P.text(x + bw / 2, P.Y(v[k]) - 4, int(v[k]), { 'text-anchor': 'middle' });
        }));
        hover(host, P, px => { const i = Math.max(0, Math.min(V.length - 1, Math.round(P.xInv(px)))), v = V[i]; return { cross: P.X(i), html: tip(v.label, [['values read', int(v.cells_read)], ['columns read', int(v.cols_read)], ['rows read', int(v.rows_read)], ['rows into join', int(v.rows_into_join)], ['evaluations', int(v.evals)], ['same result', v.same_result ? 'yes' : 'no']]) }; });
      } },
    { id: 'result', kind: 'html', title: 'Result', sub: d => d.udf ? `letterGrade applied to all ${Q.marks} marks, exactly as the slide wrote it.` : 'The rows the optimised plan returns.',
      draw(host, d) {
        let extra = '';
        if (d.udf) {
          const u = d.udf;
          extra = `<div class="sq-split"><div><p class="eyebrow">The UDF's branches (DDL deck p40)</p><ol class="sq-checks">${u.rules.map(r => `<li class="${r.duplicate ? 'no' : 'ok'}"><b>${esc(r.test)}</b> → ${esc(r.grade)}${r.duplicate ? ' (duplicate DD)' : ''}</li>`).join('')}</ol></div>
            <div><p class="eyebrow">First 10 rows against the slide's output (p41)</p>${mini(['student', 'marks', 'computed', 'slide'], u.graded.slice(0, 10).map((g, i) => [g.student_id, g.marks, g.grade, u.slide_first10[i]]))}
            <p class="sq-chips">${chip(u.matches_slide, u.matches_slide ? 'matches the slide row for row' : 'differs from the slide')} ${chip(false, `"DD" twice: ${u.duplicate.branches.join(' and ')}`)}</p></div></div>`;
        }
        host.innerHTML = extra + mini(d.result.cols, d.result.rows, { cls: 'out' });
      } },
  ],
  chartLayout: 'one',
  insights: d => {
    const out = [], sv = d.saving, m = d.metrics;
    const push = d.stages.find(s => s.rule === 'pushdown'), prune = d.stages.find(s => s.rule === 'pruning'), fold = d.stages.find(s => s.rule === 'folding');
    if (sv.cells > 0) out.push(`The optimised plan reads <b>${int(m.cells_read)}</b> values instead of ${int(sv.none_cells)}: ${pct(sv.cells_pct, 0)} less, with an identical result.`);
    if (prune?.fired) out.push(`${esc(prune.log[0].replace('ColumnPruning: ', 'Column pruning: '))}. Parquet stores each column separately, so the rest are never read.`);
    if (push?.log.some(l => l.startsWith('PushPredicateThroughJoin'))) out.push(`The filter runs before the join, so <b>${int(m.rows_into_join)}</b> rows enter it instead of ${int(sv.none_join_rows)}.`);
    const pushedLog = push?.log.find(l => l.startsWith('Data-source pushdown'));
    if (pushedLog && !m.row_groups_skipped) out.push(`${esc(pushedLog.replace('Data-source pushdown: ', 'The scan receives '))}, but the file's salary range still overlaps it, so nothing can be skipped: pushdown pays off on big files with many row groups (see Tables & files).`);
    if (m.row_groups_skipped) out.push('The pushed filter compared the threshold with the file\'s salary min/max and skipped the whole row group: <b>no rows were read at all</b>.');
    if (fold?.fired) out.push(`Constant folding computed ${esc(fold.log[0].replace('ConstantFolding: ', ''))} once, saving ${int(sv.evals)} per-row evaluations; it also lets the filter be pushed to the scan (a scan only accepts column-versus-literal filters).`);
    if (d.query.id === 'udf') out.push('A Python UDF is a black box to Catalyst: it runs in a separate BatchEvalPython step, and no filter on its result can be pushed into the scan.');
    if (d.query.id === 'slide16') out.push('Headquarters has no women, so the inner join and the filter drop it: the slide\'s groupBy only reports departments that have a match.');
    if (!out.length) out.push('No rule has anything to remove from this query: it already reads only what it returns.');
    return out;
  },
  maths: { eq: 'values read = Σ over scans (rows read × columns read)\nevaluations = Σ over operators (rows in × operators per row)',
    notes: ['The rules are three of Catalyst\'s real ones; Spark runs them, with many others, in batches until the plan stops changing. They are shown one at a time here.',
      'Tables are stored as Parquet with one row group each, so a pushed filter can skip a file only when its min/max rule out every row.'] },
};

/* ================================================================ 3 · Tables and file layout */
const STATE_OPT = [['all', 'all'], ...T.states.map(s => [s, s])];
const tables = {
  id: 'tables', group: 'bigdata', nav: 'Tables & files', tag: 'DDL / DML', title: 'Tables, partitions, buckets and Parquet files',
  question: 'What does CREATE TABLE … PARTITIONED BY … CLUSTERED BY … INTO n BUCKETS put on disk, and how much of it does a query read?',
  params: [
    { k: 'rows', label: 'Employees', min: 100, max: 20000, step: 100, value: 4000, group: 'Write', help: `A synthetic table shaped like employee_m.csv (empno, name, dno, salary, state, gender): the deck\'s ${T.deck_rows.length} visible rows, then rows generated with a fixed seed from the course\'s ${T.pool_size} (dno, salary) pairs.` },
    { k: 'buckets', label: 'Buckets (INTO n BUCKETS)', min: 1, max: 16, step: 1, value: 4, group: 'Write', help: 'Bucket = pmod(murmur3(empno), n), Spark\'s own hash.' },
    { k: 'tasks', label: 'Write tasks', min: 1, max: 8, step: 1, value: 2, group: 'Write', help: 'Partitions of the DataFrame being written. Each task writes its own file for every directory and bucket it holds rows for.' },
    { k: 'distribute', type: 'switch', label: 'DISTRIBUTE BY state, gender first', value: false, group: 'Write', help: 'Shuffle so that each directory\'s rows come from one task: fewer, larger files (deck p30: "useful before writing bucketed data").' },
    { k: 'sorted', type: 'switch', label: 'SORTED BY (salary DESC)', value: false, group: 'Write', help: 'Sort rows inside each bucket file before writing, so row groups hold narrow salary ranges.' },
    { k: 'rg_rows', label: 'Row-group size', min: 4, max: 256, step: 4, value: 16, unit: 'rows', group: 'Write', help: `Parquet sets it in bytes (parquet.block.size, ${bytes(T.parquet_block_default_bytes)} by default); rows keep this small table readable.` },
    { k: 'state', type: 'seg', label: 'WHERE state =', value: 'CA', group: 'Query', options: STATE_OPT },
    { k: 'gender', type: 'seg', label: 'AND gender =', value: 'all', group: 'Query', options: [['all', 'all'], ['M', 'M'], ['F', 'F']] },
    { k: 'salary', label: 'AND salary >', min: 0, max: 130000, step: 1000, value: 90000, group: 'Query' },
    { k: 'lookup', type: 'switch', label: `AND empno = ${T.lookup_empno}`, value: false, group: 'Query', help: 'An equality on the bucket column lets Spark read one bucket file per directory (bucket pruning, beyond the slides). The empno is row 2 of employee_m.csv.' },
    { k: 'select', type: 'seg', label: 'SELECT', value: 'two', group: 'Query', options: [['two', 'name, salary'], ['all', '*']] },
    { k: 'kind', type: 'seg', label: 'Table type', value: 'managed', group: 'DROP TABLE', options: [['managed', 'Managed'], ['external', 'External (LOCATION)']] },
    { k: 'clause', type: 'seg', label: 'Clause', value: 'distribute', group: 'Row placement', options: [['none', 'none'], ['distribute', 'DISTRIBUTE BY'], ['distribute_sort', '+ SORT BY'], ['sort', 'SORT BY'], ['cluster', 'CLUSTER BY'], ['order', 'ORDER BY']] },
    { k: 'demo_rows', label: 'Rows shown', min: 4, max: 24, step: 1, value: 12, group: 'Row placement' },
    { k: 'out_parts', label: 'Partitions', min: 1, max: 6, step: 1, value: 3, group: 'Row placement' },
  ],
  presets: [
    { label: 'Deck p21: 8 buckets', values: { buckets: 8, tasks: 2, distribute: false, sorted: false, lookup: false } },
    { label: 'Sort before writing', values: { buckets: 4, tasks: 2, distribute: false, sorted: true, rg_rows: 16, state: 'CA', salary: 90000, lookup: false } },
    { label: 'Small-files problem', values: { buckets: 8, tasks: 8, distribute: false, sorted: false } },
    { label: 'DISTRIBUTE BY before the write', values: { buckets: 8, tasks: 8, distribute: true, sorted: false } },
    { label: `Bucket pruning: empno ${T.lookup_empno}`, values: { state: 'MA', gender: 'all', salary: 0, lookup: true } },
    { label: 'Drop an external table', values: { kind: 'external' } },
    { label: 'ORDER BY: a global sort', values: { clause: 'order' } },
    { label: 'CLUSTER BY state', values: { clause: 'cluster' } },
  ],
  request: (s, sig) => get('/api/bigdata/sparksql/tables', s, sig, ['rows', 'buckets', 'tasks', 'distribute', 'sorted', 'rg_rows', 'state', 'gender', 'salary', 'lookup', 'select', 'kind', 'clause', 'demo_rows', 'out_parts']),
  kpis: d => {
    const r = d.read, b = r.bytes, w = d.write;
    return [
      { label: 'Bytes read', value: bytes(b.columns), note: `of ${bytes(b.total)} on disk (${pct(b.columns / b.total, 1)})`, tone: b.columns / b.total < 0.25 ? 'good' : null },
      { label: 'Directories read', value: `${r.dirs_read} / ${r.dirs_total}`, note: 'partition pruning on state, gender' },
      { label: 'Row groups read', value: `${r.row_groups_read} / ${r.row_groups_total}`, note: `min/max skipping in ${r.files_read} file(s) read` },
      { label: 'Files written', value: int(w.files), note: `${int(w.ideal_files)} distinct directory × bucket pairs · avg ${bytes(w.avg_file_bytes)}`, tone: w.files > 1.5 * w.ideal_files ? 'warn' : 'good' },
    ];
  },
  illustration: {
    title: 'What the query reads', height: 210,
    caption: 'Each box fills with the share of the table\'s bytes still to be read after that step.',
    draw(svg, d) {
      const steps = [['On disk', 'total'], ['Partitions', 'partitions'], ['Buckets', 'buckets'], ['Row groups', 'row_groups'], ['Columns', 'columns']];
      const tot = d?.read.bytes.total || 1;
      let prev = null;
      steps.forEach(([t, k], i) => {
        const v = d ? d.read.bytes[k] : null;
        const n = node(svg, { x: 6 + i * 120, y: 50, w: 104, h: 104, title: t, value: v != null ? bytes(v) : null, sub: v != null ? pct(v / tot, 0) : '', fill: v != null ? v / tot : 0, fillCls: i === 4 ? 'fill3' : 'fill1', hot: i === 4 });
        if (prev) edge(svg, [prev.r + 2, prev.cy], [n.x - 4, n.cy], { width: v != null ? widthFor(v, tot) : 2 });
        prev = n;
      });
      if (d) text(svg, 300, 190, `${d.read.result_rows} rows match · ${d.read.same_as_full_scan ? 'same answer as reading everything ✓' : 'differs from a full scan ✗'}`, 'n-sub');
    },
  },
  charts: [
    { id: 'write', kind: 'html', title: '1 · The write: Hive-style directories and bucket files', sub: d => `${int(d.write.files)} files in ${d.write.dirs} directories under ${esc(d.root)}/ (partition columns are in the path, not in the files).`,
      draw(host, d) {
        const dirs = d.tree.map(st => `<li><details${st.status === 'read' ? ' open' : ''}><summary class="sq-dir ${st.status}">${esc(st.name)}/ <span class="sq-cmp">${int(st.rows)} rows</span></summary><ul>${st.children.map(g => `<li><details${g.status === 'read' && g.files_total <= 8 ? ' open' : ''}><summary class="sq-dir ${g.status}">${esc(g.name)}/ <span class="sq-cmp">${g.files_total} files · ${int(g.rows)} rows · ${bytes(g.bytes)}</span></summary><ul>${g.files.map(f => `<li class="sq-file ${f.status}"><code>${esc(f.name)}</code> <span class="sq-cmp">bucket ${f.bucket} · task ${f.task} · ${f.rows} rows · ${f.row_groups} row group${f.row_groups > 1 ? 's' : ''} · ${bytes(f.bytes)}</span></li>`).join('')}${g.files_total > g.files.length ? `<li class="sq-more">… ${g.files_total - g.files.length} more</li>` : ''}</ul></details></li>`).join('')}</ul></details></li>`).join('');
        host.innerHTML = `<div class="sq-split"><div><p class="eyebrow">DDL</p><pre class="sq-code">${esc(d.ddl)}\n${esc(d.insert)}</pre></div><div><p class="eyebrow">First rows and their bucket</p>${mini(d.sample_cols, d.sample)}</div></div>
          <p class="sq-chips"><span class="chip">rows per bucket: ${d.write.bucket_rows.join(' · ')}</span> <span class="chip">${int(d.write.row_groups)} row groups</span> <span class="chip">at most ${int(d.write.max_files)} files possible (dirs × buckets × tasks)</span></p>
          <ul class="sq-fs"><li><span class="sq-dir read">${esc(d.root)}/</span><ul>${dirs}</ul></li></ul>
          <p class="c-note">Green: read by the query below · grey: pruned by partition · amber: pruned by bucket.</p>`;
      } },
    { id: 'funnel', title: '2 · The query: bytes and rows at each pruning step', height: 250, sub: d => esc(d.query.sql),
      legend: [['s1', 'box', 'bytes still to read'], ['s3', 'box', 'after column pruning']],
      draw(host, d) {
        const keys = [['total', 'on disk'], ['partitions', 'partitions'], ['buckets', 'buckets'], ['row_groups', 'row groups'], ['columns', 'columns']];
        const b = d.read.bytes, rows = d.read.rows;
        const rk = { total: 'total', partitions: 'partitions', buckets: 'buckets', row_groups: 'row_groups', columns: 'row_groups' };
        const P = frame(host, { h: 250, x: [-0.6, 4.6], y: [0, b.total * 1.12], xt: [0, 1, 2, 3, 4], xf: i => keys[i]?.[1] || '', yl: 'Bytes', yf: v => bytes(v) });
        keys.forEach(([k], i) => {
          const w = (P.X(1) - P.X(0)) * 0.55;
          E('rect', { x: P.X(i) - w / 2, width: w, y: P.Y(b[k]), height: Math.max(1, P.Y(0) - P.Y(b[k])), class: i === 4 ? 'f3' : 'f1', rx: 3 }, P.body);
          P.text(P.X(i), P.Y(b[k]) - 6, `${bytes(b[k])} · ${int(rows[rk[k]])} rows`, { 'text-anchor': 'middle' });
        });
      } },
    { id: 'rowgroups', title: '3 · Row-group min/max against the filter', height: 270,
      sub: d => `Each bar spans one row group's salary range (Parquet footer statistics). Groups entirely ≤ the threshold are skipped without decoding. ${d.row_groups.length} file(s) shown.`,
      legend: [['s1', 'box', 'read'], ['neutral', 'box', 'skipped'], ['warn', 'line', 'salary threshold']],
      show: d => d.row_groups.length > 0,
      draw(host, d, s) {
        const list = [];
        d.row_groups.forEach((f, fi) => f.row_groups.forEach((g, gi) => list.push({ ...g, fi, gi, file: f.file })));
        const n = list.length, lo = Math.min(...list.map(g => g.min)), hi = Math.max(...list.map(g => g.max), s.salary);
        const P = frame(host, { h: 270, x: [-0.6, n - 0.4], y: [Math.max(0, lo - 5000), hi + 5000], xl: 'Row group (files in order)', yl: 'Salary', yf: v => int(v), xn: 8 });
        const bw = Math.max(2, (P.X(1) - P.X(0)) * 0.7);
        list.forEach((g, i) => {
          if (i && g.fi !== list[i - 1].fi) E('line', { x1: P.X(i - 0.5), x2: P.X(i - 0.5), y1: P.M.t, y2: P.h - P.M.b, class: 'sq-sep' }, P.body);
          E('rect', { x: P.X(i) - bw / 2, width: bw, y: P.Y(g.max), height: Math.max(2, P.Y(g.min) - P.Y(g.max)), class: g.skipped ? 'fn' : 'f1', rx: 1.5, opacity: g.skipped ? 0.45 : 0.9 }, P.body);
        });
        P.hline(s.salary, 'sq-thr');
        hover(host, P, px => { const i = Math.max(0, Math.min(n - 1, Math.round(P.xInv(px)))), g = list[i]; return { cross: P.X(i), html: tip(g.file.split('/').slice(0, 2).join('/') + ` · group ${g.gi}`, [['salary min', int(g.min)], ['salary max', int(g.max)], ['rows', g.rows], ['empno range', `${g.min_empno}–${g.max_empno}`], ['', g.skipped ? 'skipped' : 'read']]) }; });
      } },
    { id: 'answer', kind: 'html', title: '4 · The answer', sub: d => `${plural(d.read.result_rows, 'row')}, highest salaries first.`,
      draw(host, d) {
        host.innerHTML = `<p class="sq-chips">${chip(d.read.same_as_full_scan, d.read.same_as_full_scan ? 'identical to filtering every row of the table' : 'differs from a full scan')} <span class="chip">${d.query.need.join(', ')} decoded</span></p>${mini(d.result.cols, d.result.rows, { total: d.read.result_rows, cls: 'out' })}`;
      } },
    { id: 'drop', kind: 'html', title: '5 · DROP TABLE: managed versus external', sub: d => esc(d.drop.note),
      draw(host, d) {
        const fs = f => f ? `<ul class="sq-fs"><li><span class="sq-dir read">${esc(f.name)}/</span> <span class="sq-cmp">${f.files} files · ${bytes(f.bytes)}</span><ul>${f.children.map(c => `<li><span class="sq-dir read">${esc(c.name)}/</span> <span class="sq-cmp">${c.files} files</span></li>`).join('')}</ul></li></ul>` : '<p class="sq-gone">(directory deleted)</p>';
        const meta = m => m ? mini(['property', 'value'], Object.entries(m).map(([k, v]) => [k, Array.isArray(v) ? v.join(', ') : v])) : '<p class="sq-gone">(no entry: SHOW TABLES no longer lists it)</p>';
        const side = (t, x) => `<div><p class="eyebrow">${t}</p><p class="sq-lbl">metastore</p>${meta(x.metastore)}<p class="sq-lbl">file system</p>${fs(x.fs)}</div>`;
        host.innerHTML = `<pre class="sq-code">${esc(d.ddl)}\n…\n${esc(d.drop.sql)};</pre><div class="sq-split">${side('Before DROP TABLE', d.drop.before)}${side('After DROP TABLE', d.drop.after)}</div>`;
      } },
    { id: 'placement', kind: 'html', title: '6 · DISTRIBUTE BY, SORT BY, CLUSTER BY, ORDER BY: where rows land', sub: d => esc(d.placement.sql),
      draw(host, d) {
        const p = d.placement, c = p.checks;
        const parts = (list, title) => `<div><p class="eyebrow">${title}</p><div class="sq-parts">${list.map((rows, i) => `<div class="sq-part"><p class="sq-part-h">partition ${i} <span class="sq-cmp">${rows.length} rows</span></p>${mini(p.cols, rows)}</div>`).join('')}</div></div>`;
        const hashes = Object.entries(p.state_hash).map(([st, h]) => [st, h.hash, h.partition]);
        host.innerHTML = `<p class="sq-chips">${chip(c.states_split === 0, c.states_split === 0 ? 'every state in one partition' : `${c.states_split} state(s) spread over partitions`)} ${chip(c.sorted_within, c.sorted_within ? 'sorted by salary within each partition' : 'not sorted within partitions')} ${chip(c.globally_sorted, c.globally_sorted ? 'globally sorted' : 'not globally sorted')} <span class="chip">${c.shuffle ? 'needs a shuffle' : 'no shuffle'}</span></p>`
          + parts(p.before, 'Before: the input partitions') + parts(p.after, 'After')
          + (['distribute', 'distribute_sort', 'cluster'].includes(p.clause) ? `<details class="more"><summary>pmod(murmur3(state), ${p.after.length}) for each state</summary>${mini(['state', 'murmur3 hash', 'partition'], hashes)}</details>` : '');
      } },
  ],
  chartLayout: 'one',
  insights: d => {
    const r = d.read, w = d.write, b = r.bytes, out = [];
    out.push(`Partition pruning reads <b>${r.dirs_read} of ${r.dirs_total}</b> directories: the filter on state and gender is answered by the folder names alone.`);
    if (d.query.need.includes('empno')) out.push(`Bucket pruning: empno ${d.query.lookup_empno} hashes to bucket <b>${d.query.lookup_bucket}</b>, so only that bucket's files are opened.`);
    out.push(`Row-group min/max statistics skip <b>${r.row_groups_in_files_read - r.row_groups_read} of ${r.row_groups_in_files_read}</b> row groups in the files read, without decoding them; ${plural(r.row_groups_read, 'row group')} decoded.`);
    out.push(`Only ${d.query.need.join(', ')} are decoded from the surviving row groups: <b>${bytes(b.columns)}</b> instead of ${bytes(b.row_groups)}.`);
    if (w.files > 1.5 * w.ideal_files) out.push(`${w.tasks} write tasks each wrote their own file per directory and bucket: <b>${int(w.files)}</b> files where ${int(w.ideal_files)} would do. DISTRIBUTE BY state, gender before the write fixes it.`);
    out.push(d.drop.kind === 'managed' ? 'DROP TABLE on a managed table deletes the files too; an external table (created with LOCATION) keeps them.' : 'DROP TABLE on an external table removes only the metastore entry; the Parquet files remain under the LOCATION path.');
    return out;
  },
  maths: { eq: 'directory = state=<s>/gender=<g>/\nbucket = pmod(murmur3(empno), n)\nskip a row group if max(salary) ≤ threshold',
    notes: ['Bytes use a plain-encoding model (INT 4 B, STRING 4 B + its UTF-8 length, plus footer and column-chunk metadata); real files are compressed.',
      'ORDER BY range-partitions on salary (Spark samples the boundaries; here they are exact quantiles).'] },
};

export default [joins, sql, tables];
