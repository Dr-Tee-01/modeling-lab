// Big Data Lab · Spark: the RDD playground (RDD decks 1–3) and stages / the DAG (Spark
// implementation deck). The server runs every pipeline on a mini RDD engine, a plain-Python
// simulation of Spark's behaviour (modlab.bigdata.spark_core), and sends the records of every
// partition at every step; the browser only lays them out and draws.
import { call, fmt, int } from '../../api.js';
import { catalog } from '../../catalog.js';
import { E, frame, hover, zip } from '../../plot.js';
import { edge, node, text } from '../../ui/illustrate.js';

const C = catalog.bigdata_spark;
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const tip = (title, rows) => `<b>${title}</b>` + rows.map(([k, v]) => `<br>${k}: <span class="n">${v}</span>`).join('');
const plural = (n, w, ws = w + 's') => `${int(n)} ${n === 1 ? w : ws}`;
const PCOL = ['s1', 's3', 's4', 's2', 'neutral', 'accent'];         // partition colours, presentation only
const pick = (s, keys) => Object.fromEntries(keys.map(k => [k, s[k]]));

/* partition cards: records per partition, with executor, size and how often it was computed */
function partCards(parts, { hot = () => false, bad = () => false, note = null } = {}) {
  if (!parts) return '<p class="bds-note">Not computed: no action has needed this RDD yet.</p>';
  return '<div class="bds-parts">' + parts.map(p => {
    const cls = bad(p) ? ' bad' : hot(p) ? ' hot' : p.records && !p.records.length ? ' empty' : '';
    const recs = p.records == null ? '<li>not computed</li>' : p.records.length ? p.records.map(r => `<li>${esc(r)}</li>`).join('') : '<li>(empty)</li>';
    return `<div class="bds-part${cls}"><h4>P${p.p} <small>executor ${p.executor}${p.cached ? ' · cached' : ''}</small></h4><ul class="bds-recs">${recs}</ul>`
      + `<div class="bds-meta">${p.count == null ? '' : `${plural(p.count, 'record')} · ${int(p.bytes)} B · `}${p.computed == null ? '' : `computed ${p.computed}×`}${p.cache_hits ? ` · read from cache ${p.cache_hits}×` : ''}${note ? note(p) : ''}</div></div>`;
  }).join('') + '</div>';
}

/* the code listing, with the inspected step highlighted */
function codeBlock(steps, cur, calls) {
  const lines = steps.map(st => `<span class="${st.i === cur ? 'cur' : st.i < cur ? 'done' : 'todo'}">${esc(st.code)}</span>`);
  return `<pre class="bds-code">${lines.join('\n')}${calls ? '\n' + calls.map(c => `<span class="todo">${esc(c)}</span>`).join('\n') : ''}</pre>`;
}

/* map partition -> reduce partition record counts, coloured by whether they cross executors */
function matrix(sh) {
  const n = sh.matrix[0]?.length || 0;
  return `<div class="tbl-wrap"><table class="bds-matrix"><thead><tr><th>From map partition</th>${Array.from({ length: n }, (_, r) => `<th class="num">→ P${r}</th>`).join('')}<th class="num">Written</th></tr></thead><tbody>`
    + sh.map.map((m, i) => `<tr><td>P${m.p} · executor ${m.executor}</td>${sh.matrix[i].map((v, r) => {
      const cross = (r % sh.executors) !== m.executor;
      return `<td class="num ${v ? (cross ? 'cross' : 'local') : 'zero'}">${v ? `${v} <small>(${sh.bytes_matrix[i][r]} B)</small>` : '0'}</td>`;
    }).join('')}<td class="num">${m.records} <small>(${m.bytes} B)</small></td></tr>`).join('')
    + '</tbody></table></div>';
}

/* layered layout of a lineage: depth from the dependencies, lane = first parent's lane when free */
function layout(items, parentsOf) {
  const depth = new Map(), lane = new Map(), taken = new Set();
  let maxLane = -1;
  items.forEach(it => {
    const ps = parentsOf(it);
    depth.set(it, ps.length ? 1 + Math.max(...ps.map(p => depth.get(p))) : 0);
    let want = ps.length ? lane.get(ps[0]) : maxLane + 1;
    if (taken.has(depth.get(it) + ':' + want)) want = maxLane + 1;
    lane.set(it, want); taken.add(depth.get(it) + ':' + want); maxLane = Math.max(maxLane, want);
  });
  return { depth, lane, cols: Math.max(...depth.values()) + 1, lanes: maxLane + 1 };
}

/* ================================================================ RDD playground */
const RDD_KEYS = ['pipe', 'parts', 'persist', 'actions', 'stats', 'step'];
const rdd = {
  id: 'rdd', group: 'bigdata', nav: 'RDD playground', tag: 'RDD 1–3', title: 'RDD playground: partitions, shuffles and laziness',
  question: 'What does each partition hold after every transformation, what crosses the network in a shuffle, and when does Spark actually run anything?',
  params: [
    { k: 'pipe', type: 'seg', label: 'Pipeline', value: 'avg_reduce', options: C.rdd.pipes, group: 'Task',
      help: 'reduceByKey, groupByKey and aggregateByKey all compute the average salary per department of the 8 employees; join totals the orders per state; regions counts customers per region; the last three are the mapPartitions, broadcast and accumulator exercises.' },
    { k: 'stats', type: 'switch', label: 'Stats(sum, count, min, max) variant', value: false, group: 'Task', show: s => s.pipe === 'avg_aggregate',
      help: 'RDD deck 2 p12: aggregate a Stats object per department instead of a (sum, count) pair.' },
    { k: 'parts', label: 'Partitions', min: 1, max: 4, step: 1, value: 2, group: 'Cluster', help: 'Number of partitions of the input (one executor per partition). It changes what each map task sees, and so what is shuffled.' },
    { k: 'actions', label: 'Actions called', min: 0, max: 3, step: 1, value: 1, group: 'Execution', help: '0: only transformations are defined, so nothing runs. Each action runs a whole job over the lineage.' },
    { k: 'persist', type: 'switch', label: 'persist() the result RDD', value: false, group: 'Execution', help: 'Keep the computed partitions of the last RDD in memory, so later actions read them instead of recomputing the lineage.' },
    { k: 'step', label: 'Step to inspect', min: 1, max: 7, step: 1, value: 4, group: 'Inspect', help: 'Which line of the program to show partition by partition. The server clamps it to the pipeline\'s length.' },
  ],
  presets: [
    { label: 'reduceByKey · 2 partitions', values: { pipe: 'avg_reduce', parts: 2, actions: 1, persist: false, step: 4 } },
    { label: 'groupByKey shuffles all', values: { pipe: 'avg_group', parts: 2, actions: 1, persist: false, step: 4 } },
    { label: 'aggregateByKey Stats', values: { pipe: 'avg_aggregate', stats: true, parts: 2, actions: 1, persist: false, step: 4 } },
    { label: 'Join state totals', values: { pipe: 'join', parts: 2, actions: 1, persist: false, step: 5 } },
    { label: 'mapPartitions · 3 partitions', values: { pipe: 'mappartitions', parts: 3, actions: 1, persist: false, step: 2 } },
    { label: 'Accumulator double count', values: { pipe: 'accumulator', parts: 2, actions: 2, persist: false, step: 2 } },
    { label: 'Nothing ran yet', values: { pipe: 'avg_reduce', parts: 2, actions: 0, persist: false, step: 4 } },
  ],
  request: (s, sig) => call('/api/bigdata/rdd', pick(s, RDD_KEYS), { signal: sig }),
  kpis: d => {
    const fourth = d.pipe === 'accumulator'
      ? { label: 'Accumulator counter', value: d.counter == null ? '—' : int(d.counter), tone: d.counter === d.counter_expected ? 'good' : d.lazy ? null : 'bad', note: `elements: ${d.counter_expected}${d.counter > d.counter_expected ? ', counted again on recompute' : ''}` }
      : d.pipe === 'broadcast'
        ? { label: 'Broadcast bytes shipped', value: int(d.broadcast.broadcast_bytes), unit: 'B', note: `${int(d.broadcast.closure_bytes)} B if captured in every task's closure` }
        : { label: 'Stages · tasks', value: `${d.stages.length} · ${int(d.stages.reduce((a, x) => a + x.tasks, 0))}`, note: `${plural(d.stages.length - 1, 'shuffle boundary', 'shuffle boundaries')}` };
    return [
      { label: 'Records shuffled', value: d.lazy ? '0' : int(d.shuffled_records), unit: d.lazy ? '' : `· ${int(d.shuffled_bytes)} B`,
        note: d.lazy ? 'nothing ran yet' : d.shuffle_input ? `of ${int(d.shuffle_input)} records entering the shuffle; ${int(d.cross_records)} change executor` : 'no shuffle: every step is narrow', tone: d.shuffled_records && d.shuffled_records === d.shuffle_input ? 'warn' : null },
      { label: 'Result', value: d.lazy ? 'not run' : d.correct ? 'correct' : 'wrong', tone: d.lazy ? 'warn' : d.correct ? 'good' : 'bad',
        note: d.lazy ? `call ${esc(d.calls[0])} to run it` : `${plural(d.expected.length, 'row')}, same as plain Python` },
      { label: 'Recomputations', value: int(d.recomputations), tone: d.recomputations ? 'warn' : 'good',
        note: `${int(d.partitions_computed)} partition computations for ${plural(d.actions, 'action')}` },
      fourth,
    ];
  },
  illustration: {
    title: 'The lineage', height: 210,
    caption: 'One box per RDD in the program. Orange arrows are shuffles (wide dependencies); the tank level shows how many of its partitions have been computed. Nothing fills until an action runs.',
    draw(svg, d, s) {
      if (!d) return;
      const steps = d.steps, byVar = Object.fromEntries(steps.map(x => [x.var, x]));
      const L = layout(steps, x => x.deps.map(dp => byVar[dp.parent]).filter(Boolean));
      const colW = 580 / L.cols, w = Math.min(104, colW - 30), hgt = 58, laneH = L.lanes > 1 ? 86 : 0, y0 = L.lanes > 1 ? 34 : 70;
      const box = new Map();
      steps.forEach(x => {
        const done = x.partitions ? x.partitions.filter(p => p.computed).length / x.n : 0;
        box.set(x, node(svg, { x: 10 + L.depth.get(x) * colW + (colW - w) / 2, y: y0 + L.lane.get(x) * laneH, w, h: hgt, title: x.var,
          sub: `${x.op} · ${x.n}P`, fill: done, fillCls: x.kind === 'shuffle' ? 'fill2' : 'fill1', hot: x.i === d.step }));
      });
      steps.forEach(x => x.deps.forEach(dp => {
        const a = box.get(byVar[dp.parent]), b = box.get(x);
        if (!a) return;
        const wide = dp.kind === 'shuffle';
        edge(svg, [a.r + 2, a.cy], [b.x - 4, b.cy], { width: wide ? 3.5 : 1.6, cls: wide ? 'st2' : '', flow: wide && !d.lazy });
      }));
      text(svg, 300, 200, d.lazy ? 'Nothing ran yet: Spark has only recorded the lineage.' : `${d.jobs.length} job${d.jobs.length > 1 ? 's' : ''} ran · ${int(d.partitions_computed)} partition computations`, 'n-sub');
    },
  },
  charts: [
    { id: 'step', kind: 'html', title: 'Records per partition at this step',
      sub: (d) => { const st = d.steps[d.step - 1]; return `Step ${st.i} of ${d.steps.length}: <code>${esc(st.var)}</code>, ${st.op} (${st.kind === 'shuffle' ? 'wide dependency' : st.kind === 'source' ? 'data source' : 'narrow dependency'}), ${plural(st.n, 'partition')}${st.partitioner ? `, ${esc(st.partitioner)}` : ''}.`; },
      caption: d => `${esc(d.label)} · ${esc(d.slides)} · data ${esc(d.data)}. A simulation of Spark's behaviour: partitions are cut like <code>parallelize</code>, keys are placed with Java's <code>hashCode</code> mod n.`,
      draw(host, d) {
        const st = d.steps[d.step - 1];
        host.innerHTML = codeBlock(d.steps, d.step, d.calls.slice(0, Math.max(1, d.actions)))
          + (st.note ? `<p class="bds-note">${esc(st.note)}</p>` : '')
          + (d.lazy ? '<p class="bds-note"><b>Nothing ran yet.</b> Transformations are lazy: Spark has built the lineage above and stopped. Raise "Actions called" to run a job.</p>' : '')
          + partCards(st.partitions, { hot: p => p.cached });
      } },
    { id: 'shuffle', kind: 'html', title: 'What crosses the shuffle',
      show: d => !d.lazy && d.steps[d.step - 1].shuffles.length > 0,
      sub: d => { const sh = d.steps[d.step - 1].shuffles; return sh.map(x => `From <code>${esc(x.from)}</code> with ${esc(x.partitioner)}: ${x.combine ? 'combined inside each map partition first' : 'no map-side combine'}, ${int(x.records_in)} records in, ${int(x.records)} written (${int(x.bytes)} B).`).join(' '); },
      caption: 'Green cells stay on the same executor; orange cells travel to another machine. Byte counts are the length of each record\'s text form.',
      draw(host, d) {
        host.innerHTML = d.steps[d.step - 1].shuffles.map(sh => {
          const sh2 = { ...sh, executors: d.parts };
          return `<p class="bds-sec">Map side: what each partition writes${sh.combine ? ' (after combining)' : ''}</p>`
            + partCards(sh.map.map(m => ({ p: m.p, executor: m.executor, records: m.written, count: m.records, bytes: m.bytes, records_in: m.records_in, computed: null, cache_hits: 0 })), { note: m => `written from ${m.records_in} input records` })
            + `<p class="bds-sec">Map partition → reduce partition (records, bytes)</p>` + matrix(sh2);
        }).join('');
      } },
    { id: 'overview', title: 'Records per partition, step by step', height: 230, show: d => !d.lazy,
      sub: 'Each group of bars is one step of the program; each bar one partition. Shuffles change both the count and the placement.',
      legend: d => Array.from({ length: d.parts }, (_, i) => [PCOL[i], 'box', `P${i}`]),
      draw(host, d) {
        const steps = d.steps, hi = Math.max(1, ...steps.flatMap(st => (st.partitions || []).map(p => p.count || 0)));
        const P = frame(host, { h: 230, x: [0.4, steps.length + 0.6], y: [0, hi * 1.15], xt: steps.map(st => st.i), xf: i => steps[i - 1]?.var || '', yl: 'Records' });
        steps.forEach(st => (st.partitions || []).forEach((p, j) => {
          const bw = 0.8 / st.n, x0 = st.i - 0.4 + j * bw;
          E('rect', { x: P.X(x0) + 1, y: P.Y(p.count || 0), width: Math.max(2, P.X(x0 + bw) - P.X(x0) - 2), height: Math.max(0, P.Y(0) - P.Y(p.count || 0)), rx: 2,
            style: `fill:var(--${PCOL[j % PCOL.length]});opacity:${st.i === d.step ? 1 : 0.55}` }, P.body);
        }));
        hover(host, P, px => {
          const i = Math.max(1, Math.min(steps.length, Math.round(P.xInv(px)))), st = steps[i - 1];
          return { cross: P.X(i), html: tip(`${st.var} (${st.op})`, (st.partitions || []).map(p => [`P${p.p}`, `${p.count} records, ${p.bytes} B`])) };
        });
      } },
    { id: 'jobs', kind: 'html', title: 'Actions and recomputation',
      sub: d => d.lazy ? 'No action has been called, so no job has run.' : `Each action is a separate job over the whole lineage. persist() is ${d.persist ? 'on: later jobs read the result RDD from memory' : 'off: later jobs recompute it'}.`,
      draw(host, d) {
        if (d.lazy) { host.innerHTML = `<p class="bds-note">Defined: ${d.steps.map(st => `<code>${esc(st.var)}</code>`).join(' → ')}. Computed partitions: 0.</p>`; return; }
        host.innerHTML = `<div class="tbl-wrap"><table><thead><tr><th class="num">#</th><th>Action</th><th>Result</th><th class="num">Partitions computed</th><th>Recomputed</th><th class="num">Stages run / skipped</th><th class="num">Tasks</th>${d.counter != null ? '<th class="num">counter</th>' : ''}</tr></thead><tbody>`
          + d.jobs.map(j => `<tr class="${j.recomputed.length ? 'bds-warn' : ''}"><td class="num">${j.i}</td><td><code>${esc(j.call)}</code></td><td class="mono">${esc(j.result)}</td><td class="num">${j.partitions_computed}</td><td class="mono">${j.recomputed.length ? esc(compact(j.recomputed)) : '—'}</td><td class="num">${j.stages_run} / ${j.stages_skipped}</td><td class="num">${j.tasks}</td>${j.counter != null ? `<td class="num">${j.counter}</td>` : ''}</tr>`).join('')
          + '</tbody></table></div>';
      } },
    { id: 'expected', kind: 'html', title: 'Check your answer',
      sub: 'The expected output is computed directly in plain Python, without the engine, so the two can be compared.',
      draw(host, d) {
        const got = d.lazy ? '<i>not run yet</i>' : esc(d.result);
        host.innerHTML = `<div class="tbl-wrap"><table><tbody><tr><td>Expected</td><td class="mono">${esc('[' + d.expected.join(', ') + ']')}</td></tr>`
          + `<tr class="${d.lazy ? '' : d.correct ? 'bds-good' : 'bds-bad'}"><td>Spark simulation, action 1</td><td class="mono">${got}</td></tr></tbody></table></div>`
          + (d.counter != null ? `<p class="cap">Accumulator after ${plural(d.actions, 'action')}: <b>${d.counter}</b>; one count per element is ${d.counter_expected}.</p>` : '')
          + (d.broadcast ? `<p class="cap">N = ${esc(d.broadcast.value)} is ${d.broadcast.bytes} B. Broadcast: shipped once per executor (${d.broadcast.broadcast_bytes} B). Captured in a closure instead, it would travel with every task: ${d.broadcast.closure_bytes} B for ${d.broadcast.tasks} tasks.</p>` : '');
      } },
  ],
  insights: d => {
    if (d.lazy) return [`No action was called, so Spark computed <b>0</b> partitions: <code>${d.steps.map(st => esc(st.var)).join('</code>, <code>')}</code> are only recipes in the lineage.`];
    const out = [];
    const sh = d.steps.flatMap(st => st.shuffles);
    if (sh.length) out.push(`The shuffle${sh.length > 1 ? 's' : ''} wrote <b>${int(d.shuffled_records)}</b> record${d.shuffled_records === 1 ? '' : 's'} (${int(d.shuffled_bytes)} B) from ${int(d.shuffle_input)} input records` + (sh.some(x => x.combine) ? ': reduceByKey and aggregateByKey send one partial result per key per partition, groupByKey sends every record.' : ': with no map-side combine, every record crosses.'));
    else out.push('Every step is narrow: each partition is transformed where it is, and nothing is shuffled.');
    if (d.actions > 1) out.push(d.recomputations ? `${plural(d.actions, 'action')} recomputed <b>${int(d.recomputations)}</b> partition${d.recomputations === 1 ? '' : 's'}` + (d.persist ? '.' : '. Turn on persist() to keep the result in memory.') + (d.shuffled_records ? ' Steps before a shuffle are not redone: Spark keeps the shuffle files and skips that stage.' : '') : `With persist(), later actions read the cached result: <b>0</b> recomputations.`);
    if (d.pipe === 'accumulator') out.push(d.counter > d.counter_expected ? `The accumulator reads <b>${d.counter}</b> for ${d.counter_expected} elements: f1 ran again on every recompute. Update accumulators in actions, or persist the RDD.` : `The accumulator reads <b>${d.counter}</b>, one per element.`);
    if (d.pipe === 'mappartitions') out.push(`The counter restarts in every partition, so the output depends on the partitioning: with ${d.parts} partition${d.parts > 1 ? 's' : ''} it is ${esc(d.result)}.`);
    out.push(d.correct ? 'The result matches the plain-Python answer.' : 'The result does not match the plain-Python answer.');
    return out;
  },
  maths: { eq: 'average(dno) = Σ salary / Σ 1   over the employees of dno\nreduceByKey:    (s₁, c₁) ⊕ (s₂, c₂) = (s₁ + s₂, c₁ + c₂)   (associative, commutative)\naggregateByKey: zero = (0, 0);  seq(a, x) = (a.sum + x, a.count + 1);  comb = ⊕\nrecords shuffled = Σ over partitions of (distinct keys in the partition)   with map-side combine',
    notes: ['A pair (sum, count) can be combined in any order, which is what lets Spark combine inside each partition before the shuffle. An average cannot: averaging partial averages gives the wrong answer unless the counts are equal.',
      'A simulation of Spark\'s behaviour in plain Python, not PySpark. Keys are hashed with Java\'s String.hashCode, as in Spark\'s Java API.'] },
};
const compact = list => {
  const by = {};
  list.forEach(x => (by[x.step] ||= []).push('P' + x.p));
  return Object.entries(by).map(([k, v]) => `${k} ${v.join(',')}`).join('; ');
};

/* ================================================================ Stages and the DAG */
const DAG_KEYS = ['pipe', 'parts', 'cache', 'actions', 'reuse', 'copart', 'target', 'how', 'lose', 'stage', 'iterations', 'data_gb'];
const LOSE = [[-1, 'None'], ...Array.from({ length: C.dag.executors }, (_, i) => [i, `P${i}`])];
function drawDag(host, d) {
  const R = d.rdds, byId = Object.fromEntries(R.map(r => [r.id, r]));
  const sq = 13, gap = 4, colW = 150, bw = 46;
  const cols = Math.max(...R.map(r => r.depth)) + 1, lanes = Math.max(...R.map(r => r.lane)) + 1;
  const laneH = Array.from({ length: lanes }, (_, l) => Math.max(0, ...R.filter(r => r.lane === l).map(r => r.n * (sq + gap))) + 62);
  const laneY = laneH.map((_, l) => 8 + laneH.slice(0, l).reduce((a, b) => a + b, 0));
  const W = 30 + cols * colW, H = laneY[lanes - 1] + laneH[lanes - 1] + 6;
  const svg = E('svg', { class: 'bds-dag', viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Lineage graph with stages' });
  const X = r => 22 + r.depth * colW + (colW - bw) / 2 - 12, Y = r => laneY[r.lane] + 34;
  const sqY = (r, i) => Y(r) + 4 + i * (sq + gap);
  const last = d.jobs[d.jobs.length - 1], skipped = new Set((last?.stages || []).filter(s => s.skipped).map(s => s.id));
  // stage outlines: one rounded box per lane the stage occupies
  const gS = E('g', {}, svg);
  d.stages.forEach(st => {
    const mine = R.filter(r => r.stage === st.id), lanesOf = [...new Set(mine.map(r => r.lane))];
    lanesOf.forEach((l, k) => {
      const rr = mine.filter(r => r.lane === l), d0 = Math.min(...rr.map(r => r.depth)), d1 = Math.max(...rr.map(r => r.depth));
      const x0 = 22 + d0 * colW - 6, x1 = 22 + (d1 + 1) * colW - 30;
      E('rect', { x: x0, y: laneY[l] + 2, width: x1 - x0, height: laneH[l] - 10, rx: 12, class: 'stage' + (st.id === d.stage ? ' sel' : '') + (skipped.has(st.id) ? ' skipped' : '') }, gS);
      if (k === 0) E('text', { x: x0 + 8, y: laneY[l] + laneH[l] - 14, class: 'st-lbl' }, gS).textContent = skipped.has(st.id) ? `Stage ${st.id} skipped` : `Stage ${st.id} · ${st.tasks} task${st.tasks > 1 ? 's' : ''}`;
    });
  });
  // dependencies, partition to partition
  const gE = E('g', {}, svg);
  const rec = new Set((d.recovery?.recomputed || []).map(x => x.rdd + '|' + x.p));
  R.forEach(r => r.deps.forEach(dp => {
    const p = byId[dp.parent];
    for (let i = 0; i < r.n; i++) {
      const srcs = dp.kind === 'narrow' ? dp.map[i] : Array.from({ length: p.n }, (_, j) => j);
      srcs.forEach(j => {
        const x1 = X(p) + bw, y1 = sqY(p, j) + sq / 2, x2 = X(r), y2 = sqY(r, i) + sq / 2, mx = (x1 + x2) / 2;
        const hotl = rec.has(r.name + '|' + i) && rec.has(p.name + '|' + j);
        E('path', { d: `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`, class: 'dep' + (dp.kind === 'shuffle' ? ' wide' : '') + (hotl ? ' hotline' : '') }, gE);
      });
    }
  }));
  // RDDs and their partitions
  const lost = new Set((d.recovery?.lost_cache || []).map(x => x.rdd + '|' + x.p));
  R.forEach(r => {
    const g = E('g', {}, svg);
    E('text', { x: X(r) + bw / 2, y: Y(r) - 18, 'text-anchor': 'middle', class: 'rdd-name' }, g).textContent = r.name;
    E('text', { x: X(r) + bw / 2, y: Y(r) - 6, 'text-anchor': 'middle', class: 'rdd-sub' }, g).textContent = `${r.op}${r.persisted ? ' · persist' : ''}`;
    r.partitions.length ? r.partitions.forEach(p => {
      const cls = rec.has(r.name + '|' + p.p) ? 'recomp' : p.cached ? 'cached' : '';
      const sqr = E('rect', { x: X(r), y: sqY(r, p.p), width: bw, height: sq, rx: 3, class: 'part ' + cls }, g);
      E('title', {}, sqr).textContent = `${r.name} P${p.p} · executor ${p.executor} · ${p.count ?? 0} records · computed ${p.computed}×${p.cached ? ' · cached' : ''}`;
      if (lost.has(r.name + '|' + p.p)) E('rect', { x: X(r) - 3, y: sqY(r, p.p) - 3, width: bw + 6, height: sq + 6, rx: 4, class: 'part lost' }, g);
    }) : Array.from({ length: r.n }, (_, i) => E('rect', { x: X(r), y: sqY(r, i), width: bw, height: sq, rx: 3, class: 'part' }, g));
  });
  host.innerHTML = ''; host.appendChild(svg);
}

const dag = {
  id: 'dag', group: 'bigdata', nav: 'Stages & DAG', tag: 'Spark impl', title: 'Spark stages, the DAG and lineage recovery',
  question: 'Where does Spark cut a job into stages, how many tasks run, and what is recomputed when a partition is cached, co-partitioned or lost?',
  params: [
    { k: 'pipe', type: 'seg', label: 'Pipeline', value: 'quiz', options: C.dag.pipes, group: 'Job',
      help: 'Quiz: the deck\'s 4-stage exercise. p20: the paper\'s figure with a cached RDD B. Logs: log mining with three actions. Join: customers ⋈ orders. Resize: repartition against coalesce. Hash: where HashPartitioner(n) puts each key.' },
    { k: 'parts', label: 'Input partitions', min: 1, max: 6, step: 1, value: 3, group: 'Job', help: `Partitions of each input. Partition i runs on executor i mod ${C.dag.executors}.` },
    { k: 'copart', type: 'switch', label: 'Co-partition the join inputs', value: false, group: 'Job', show: s => s.pipe === 'quiz' || s.pipe === 'join',
      help: 'Quiz: use mapValues (keeps groupByKey\'s HashPartitioner) instead of map. Join: partitionBy the same HashPartitioner on both sides first. A co-partitioned input is joined through a narrow dependency.' },
    { k: 'how', type: 'seg', label: 'Resize with', value: 'repartition', options: [['repartition', 'repartition'], ['coalesce', 'coalesce']], group: 'Job', show: s => s.pipe === 'resize' },
    { k: 'target', label: 'Target partitions n', min: 1, max: 8, step: 1, value: 2, group: 'Job', show: s => s.pipe === 'resize' || s.pipe === 'hash' },
    { k: 'cache', type: 'switch', label: 'persist() the reuse point', value: false, group: 'Caching', show: s => s.pipe !== 'hash',
      help: 'Quiz: the grouped side. p20: B. Logs: errors. Join: the partitioned inputs. Resize: the input. Once its partitions are in memory, later jobs skip everything upstream of it.' },
    { k: 'actions', label: 'Actions (jobs)', min: 1, max: 3, step: 1, value: 1, group: 'Caching', help: 'Each action is a job. Caching and kept shuffle files only pay off from the second job on.' },
    { k: 'reuse', type: 'switch', label: 'Keep shuffle files between jobs', value: true, group: 'Caching',
      help: 'Spark keeps map outputs on disk and skips a finished shuffle-map stage in later jobs. Turn off to see the cost when they are gone.' },
    { k: 'lose', type: 'seg', label: 'Lose partition (its executor fails)', value: -1, options: LOSE, group: 'Failure',
      help: `After the jobs, the executor holding partition k of every RDD fails: its cached partitions and shuffle files are lost, and Spark rebuilds partition k of the final RDD from lineage.` },
    { k: 'stage', label: 'Stage to inspect', min: 1, max: 6, step: 1, value: 1, group: 'Inspect', help: 'Show the partitions of every RDD in this stage. The server clamps it to the number of stages.' },
    { k: 'iterations', label: 'Iterations', min: 1, max: 30, step: 1, value: 10, group: 'Iterative jobs', help: 'Logistic regression iterations over the same data (Spark intro p13).' },
    { k: 'data_gb', label: 'Data size', min: 10, max: 5000, step: 10, value: 100, unit: 'GB', group: 'Iterative jobs', help: `The slide's run: 100 GB on ${C.dag.iter.nodes} nodes. Data beyond about ${int(C.dag.iter.cache_gb)} GB of cluster memory has to be re-read from disk.` },
  ],
  presets: [
    { label: 'Quiz: 4 stages', values: { pipe: 'quiz', parts: 3, copart: false, cache: false, actions: 1, reuse: true, lose: -1, stage: 1 } },
    { label: 'Quiz, co-partitioned', values: { pipe: 'quiz', parts: 3, copart: true, cache: false, actions: 1, reuse: true, lose: -1, stage: 3 } },
    { label: 'p20: B already in RAM', values: { pipe: 'paper', parts: 3, cache: true, actions: 2, reuse: false, lose: -1, stage: 2 } },
    { label: 'Log mining, cached', values: { pipe: 'logmine', parts: 3, cache: true, actions: 3, reuse: true, lose: -1, stage: 1 } },
    { label: 'Lose partition 1', values: { pipe: 'quiz', parts: 3, copart: false, cache: false, actions: 1, reuse: true, lose: 1, stage: 4 } },
    { label: 'coalesce 3 → 2', values: { pipe: 'resize', parts: 3, how: 'coalesce', target: 2, actions: 1, lose: -1, stage: 1 } },
    { label: 'HashPartitioner(4)', values: { pipe: 'hash', parts: 3, target: 4, actions: 1, lose: -1, stage: 2 } },
  ],
  request: (s, sig) => call('/api/bigdata/dag', pick(s, DAG_KEYS), { signal: sig }),
  kpis: d => {
    const it = d.iter;
    return [
      { label: 'Stages', value: int(d.n_stages), note: `${plural(d.wide_deps, 'wide dependency', 'wide dependencies')} · ${plural(d.narrow_deps, 'narrow one')}` },
      { label: 'Tasks run', value: int(d.tasks_run), note: `${int(d.tasks_total)} per full run of the last job · one per partition` },
      { label: 'Stages skipped', value: int(d.stages_skipped), tone: d.stages_skipped ? 'good' : null, note: `over ${plural(d.actions, 'job')} (cached RDDs or kept shuffle files)` },
      d.recovery
        ? { label: 'Recomputed after the failure', value: int(d.recovery.recomputed.length), unit: 'partitions', tone: 'warn', note: `of ${int(d.recovery.partitions_total)}, in ${plural(d.recovery.tasks, 'task')}` }
        : { label: `Spark vs MapReduce at ${it.n} iterations`, value: `${fmt(it.speedup, 1)}×`, note: `${fmt(it.per_iter_ratio, 0)}× per iteration after the first load` },
    ];
  },
  illustration: {
    title: 'From an action to tasks', height: 210,
    caption: 'An action becomes a job; the DAGScheduler cuts it into stages at shuffles; the TaskScheduler runs one task per partition of each stage on the executors.',
    draw(svg, d, s) {
      const last = d?.jobs[d.jobs.length - 1];
      const run = last ? last.stages.filter(x => !x.skipped).length : 0;
      const tasks = last ? last.stages.reduce((a, x) => a + x.tasks_run, 0) : 0;
      const a = node(svg, { x: 6, y: 70, w: 110, h: 70, title: 'Action', sub: last ? last.call.split('.').pop().slice(0, 16) : '' });
      const b = node(svg, { x: 150, y: 70, w: 130, h: 70, title: 'DAGScheduler', value: d ? `${d.n_stages} stages` : '', sub: last ? `${run} run, ${last.stages.length - run} skipped` : '' });
      const c = node(svg, { x: 312, y: 70, w: 130, h: 70, title: 'TaskScheduler', value: d ? `${tasks} tasks` : '', sub: 'one per partition', hot: true });
      edge(svg, [a.r + 3, a.cy], [b.x - 5, b.cy], { width: 2 });
      edge(svg, [b.r + 3, b.cy], [c.x - 5, c.cy], { width: 3, flow: true });
      const n = C.dag.executors;
      for (let e = 0; e < n; e++) {
        const cached = d ? d.rdds.reduce((acc, r) => acc + r.partitions.filter(p => p.executor === e && p.cached).length, 0) : 0;
        const ex = node(svg, { x: 478, y: 14 + e * 64, w: 116, h: 52, title: `Executor ${e}`, sub: d ? `${cached} cached${d.lose === e ? ' · failed' : ''}` : '', fill: d && d.lose === e ? 1 : 0, fillCls: 'fill2' });
        edge(svg, [c.r + 3, c.cy], [ex.x - 5, ex.cy], { width: 1.6 });
      }
    },
  },
  charts: [
    { id: 'graph', kind: 'html', title: 'Lineage, partitions and stages',
      sub: d => `${esc(d.label)} (${esc(d.slides)}). Each column of squares is an RDD, each square a partition. Grey lines are narrow dependencies, orange lines shuffles. Dashed boxes are stages; the highlighted one is stage ${d.stage}.`,
      legend: [['ink-2', 'box', 'Cached partition'], ['bad', 'box', 'Recomputed after the failure'], ['s2', 'line', 'Shuffle (wide)'], ['neutral', 'line', 'Narrow']],
      caption: d => d.pipe === 'logmine' ? 'Each action of the log example is its own one-stage job; the boxes show the last job. <code>mysql</code> belongs to the second job only.' : d.jobs.length > 1 ? `Faded stages were skipped by job ${d.jobs.length}.` : '',
      draw(host, d) { drawDag(host, d); } },
    { id: 'stage', kind: 'html', title: 'Inside the selected stage',
      sub: d => { const st = d.stages[d.stage - 1]; return `Stage ${st.id} is a ${st.kind === 'result' ? 'result stage (it returns to the driver)' : 'shuffle-map stage (it writes shuffle files)'}: ${st.rdds.map(esc).join(' → ')} pipelined in ${plural(st.tasks, 'task')}${st.parents.length ? `, after stage${st.parents.length > 1 ? 's' : ''} ${st.parents.join(' and ')}` : ''}.`; },
      draw(host, d) {
        const st = d.stages[d.stage - 1];
        host.innerHTML = st.rdds.map(name => {
          const r = d.rdds.find(x => x.name === name);
          return `<p class="bds-sec">${esc(name)} · ${esc(r.op)} · ${plural(r.n, 'partition')}${r.partitioner ? ' · ' + esc(r.partitioner) : ''}</p>`
            + partCards(r.partitions.length ? r.partitions : null, { bad: p => p.recomputed, hot: p => p.cached });
        }).join('');
      } },
    { id: 'jobs', kind: 'html', title: 'Jobs: stages run and skipped',
      sub: d => `Shuffle files are ${d.reuse ? 'kept between jobs, as Spark does' : 'dropped between jobs'}; ${d.cache_point ? `${esc(d.cache_point)} is ${d.cache ? 'persisted' : 'not persisted'}` : 'nothing is persisted'}.`,
      draw(host, d) {
        host.innerHTML = `<div class="tbl-wrap"><table><thead><tr><th class="num">Job</th><th>Action</th><th>Stages</th><th class="num">Tasks</th><th class="num">Partitions computed</th><th class="num">Shuffle written</th></tr></thead><tbody>`
          + d.jobs.map(j => `<tr><td class="num">${j.i}</td><td><code>${esc(j.call)}</code></td><td><span class="bds-chips">${j.stages.map(st => `<span class="chip ${st.skipped ? '' : 'good'}">S${st.id}${st.skipped ? ' skipped' : ` · ${st.tasks_run}`}</span>`).join('')}</span></td><td class="num">${j.stages.reduce((a, x) => a + x.tasks_run, 0)}</td><td class="num">${j.partitions_computed}</td><td class="num">${j.shuffle_written} rec · ${j.shuffle_bytes} B</td></tr>`).join('')
          + (d.recovery ? `<tr class="bds-warn"><td class="num">↻</td><td>rebuild P${d.recovery.targets.join(', P')} after executor ${d.recovery.executor} fails</td><td><span class="bds-chips">${d.recovery.stages.map(st => `<span class="chip ${st.tasks_run ? 'warn' : ''}">S${st.id}${st.tasks_run ? ` · ${st.tasks_run}` : ' skipped'}</span>`).join('')}</span></td><td class="num">${d.recovery.tasks}</td><td class="num">${d.recovery.recomputed.length}</td><td class="num">—</td></tr>` : '')
          + '</tbody></table></div>'
          + `<p class="cap">Last result: <span class="mono">${esc(d.result.length > 260 ? d.result.slice(0, 260) + ' …' : d.result)}</span></p>`;
      } },
    { id: 'recovery', kind: 'html', title: 'Lineage recovery', show: d => !!d.recovery,
      sub: d => `Executor ${d.recovery.executor} failed. Lost: ${plural(d.recovery.lost_cache.length, 'cached partition')} and ${plural(d.recovery.lost_map_outputs.length, 'shuffle file')}. Spark does not restore them from a replica: it re-runs the lineage for exactly the missing pieces.`,
      draw(host, d) {
        const by = {};
        d.recovery.recomputed.forEach(x => (by[x.rdd] ||= []).push(x.p));
        host.innerHTML = `<div class="tbl-wrap"><table><thead><tr><th>RDD</th><th>Recomputed partitions</th><th class="num">Of</th></tr></thead><tbody>`
          + d.rdds.map(r => `<tr class="${by[r.name] ? 'bds-warn' : ''}"><td>${esc(r.name)}</td><td class="mono">${by[r.name] ? by[r.name].map(p => 'P' + p).join(', ') : '—'}</td><td class="num">${r.n}</td></tr>`).join('')
          + '</tbody></table></div>';
      } },
    { id: 'moves', kind: 'html', title: 'Where each record goes', show: d => d.pipe === 'resize' || d.pipe === 'hash',
      sub: d => d.pipe === 'hash' ? `partition = hashCode(key) mod ${d.target} (Java String.hashCode, made non-negative).` : d.how === 'coalesce' ? `coalesce(${d.target}) merges neighbouring partitions without a shuffle: a narrow dependency.` : `repartition(${d.target}) deals every record round-robin to the new partitions through a full shuffle.`,
      draw(host, d) {
        if (d.pipe === 'hash') {
          host.innerHTML = `<div class="tbl-wrap"><table><thead><tr><th>Key</th><th class="num">hashCode</th><th class="num">mod ${d.target}</th><th class="num">Records</th></tr></thead><tbody>`
            + d.placement.map(p => `<tr><td>${esc(p.key)}</td><td class="num">${int(p.hash)}</td><td class="num">P${p.partition}</td><td class="num">${p.records}</td></tr>`).join('')
            + '</tbody></table></div>' + `<p class="cap">${d.placement.length} distinct keys over ${d.target} partitions: ${(() => { const used = new Set(d.placement.map(p => p.partition)); return `${used.size} partition${used.size > 1 ? 's' : ''} hold data, ${d.target - used.size} stay empty`; })()}.</p>`;
          return;
        }
        host.innerHTML = `<div class="tbl-wrap"><table><thead><tr><th>From</th><th>To</th><th class="num">Records</th><th class="num">Bytes</th><th>Network?</th></tr></thead><tbody>`
          + d.moves.map(m => `<tr><td>P${m.from} (executor ${m.from % d.executors})</td><td>P${m.to} (executor ${m.to % d.executors})</td><td class="num">${m.records}</td><td class="num">${m.bytes}</td><td>${m.cross ? '<span class="chip warn">moves</span>' : '<span class="chip good">stays</span>'}</td></tr>`).join('')
          + '</tbody></table></div>'
          + `<p class="cap">Output partition sizes: ${d.sizes.join(', ')}. ${d.how === 'repartition' ? `All ${int(d.shuffle_records)} records go through shuffle files; ${int(d.moved_records)} change machine.` : `No shuffle files; ${int(d.moved_records)} records change machine when partitions on different executors are merged.`}</p>`;
      } },
    { id: 'iter', title: 'Iterative jobs: MapReduce against Spark', height: 270,
      sub: d => `Logistic regression on ${int(d.iter.data_gb)} GB (Spark intro p13): MapReduce re-reads the input every iteration (${fmt(d.iter.per_iter_mr, 0)} s); Spark reads it once, then ${fmt(d.iter.per_iter_spark, d.iter.per_iter_spark < 10 ? 1 : 0)} s per cached iteration.`,
      legend: [['s2', 'line', 'Hadoop MapReduce'], ['s1', 'line', 'Spark']],
      draw(host, d) {
        const it = d.iter, n = it.iterations;
        const P = frame(host, { h: 270, x: [1, n[n.length - 1]], y: [0, Math.max(...it.mapreduce) * 1.05], xl: 'Iterations', yl: 'Running time (s)' });
        P.line(zip(n, it.mapreduce), 's2'); P.line(zip(n, it.spark), 's1'); P.vline(it.n);
        P.text(P.X(it.n) + 6, P.Y(it.mr_total) - 6, `${fmt(it.speedup, 1)}× at ${it.n}`);
        hover(host, P, px => { const i = Math.max(0, Math.min(n.length - 1, Math.round(P.xInv(px)) - 1)); return { cross: P.X(n[i]), pts: [[P.X(n[i]), P.Y(it.mapreduce[i]), 'f2'], [P.X(n[i]), P.Y(it.spark[i]), 'f1']], html: tip(`${n[i]} iteration${n[i] > 1 ? 's' : ''}`, [['MapReduce', fmt(it.mapreduce[i], 0) + ' s'], ['Spark', fmt(it.spark[i], 0) + ' s'], ['Ratio', fmt(it.mapreduce[i] / it.spark[i], 1) + '×']]) }; });
      } },
  ],
  insights: d => {
    const out = [], st = d.stages;
    out.push(`The DAGScheduler cut this job into <b>${d.n_stages}</b> stage${d.n_stages > 1 ? 's' : ''} at ${plural(d.wide_deps, 'shuffle dependency', 'shuffle dependencies')}; narrow steps are pipelined inside a stage, one task per partition (${st.map(x => `S${x.id}: ${x.tasks}`).join(', ')}).`);
    if (d.pipe === 'quiz') out.push(d.copart ? 'With mapValues the grouped side keeps its HashPartitioner, so join reads it through a narrow dependency and the groupByKey stage and the join stage merge: one stage fewer than the quiz answer.' : 'groupByKey and join are the stage boundaries, as the deck\'s answer says: four stages.');
    if (d.stages_skipped) out.push(`<b>${int(d.stages_skipped)}</b> stage${d.stages_skipped > 1 ? 's were' : ' was'} skipped in later jobs: ${d.cache ? `the persisted ${esc(d.cache_point)} and ` : ''}${d.reuse ? 'the shuffle files Spark keeps' : 'cached data'} already held their output.`);
    if (d.recovery) out.push(`Losing partition ${d.recovery.executor} recomputed <b>${d.recovery.recomputed.length}</b> of ${d.recovery.partitions_total} partitions. Narrow dependencies need only the matching parent partition; a shuffle needs its map outputs, and only the lost executor's map tasks re-run because the other shuffle files survive.`);
    out.push(`For iterative jobs Spark is <b>${fmt(d.iter.speedup, 1)}×</b> faster at ${d.iter.n} iterations and approaches ${fmt(d.iter.limit, 0)}× as iterations grow: the first iteration still reads from disk. The slide's "100× at 10 iterations" holds only per iteration, not for the whole job.`);
    return out;
  },
  maths: { eq: 'stages = 1 + number of shuffle dependencies on the path from the inputs (shared parents counted once)\ntasks in a stage = partitions of its last RDD\nMapReduce:  T(n) = n · t_disk\nSpark:      T(n) = t_disk + (n − 1) · t_mem      →  T_MR / T_Spark → t_disk / t_mem',
    notes: [`With the slide's numbers, t_disk = ${C.dag.iter.mr_s} s and t_mem = ${C.dag.iter.spark_s} s for ${C.dag.iter.data_gb} GB, so 10 iterations take 1,100 s against 119 s: about 9×, as the slide's own chart shows.`,
      'A simulation of Spark\'s scheduling in plain Python: the stage cutting follows the DAGScheduler, but there is no real cluster, timing or locality-aware placement.'] },
};

export default [rdd, dag];
