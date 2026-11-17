// Big Data Lab · storage and batch processing: HDFS blocks and replication, a MapReduce engine with
// five jobs, and joins in MapReduce. Every block map, key/value pair, byte count and expected output
// on screen comes from the server (/api/bigdata/hdfs, /mapreduce, /mr-joins, and the catalog section
// bigdata_mapreduce); this file only lays the numbers out.
import { call, fmt, int, pct } from '../../api.js';
import { catalog } from '../../catalog.js';
import { E, frame, hover, nearest } from '../../plot.js';
import { S, edge, node, text, widthFor } from '../../ui/illustrate.js';

const C = catalog.bigdata_mapreduce;
const get = (path, s, sig, keys) => call(path, Object.fromEntries(keys.map(k => [k, s[k]])), { signal: sig });
const esc = v => String(v ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const tip = (title, rows) => `<b>${title}</b>` + rows.map(([k, v]) => `<br>${k}: <span class="n">${v}</span>`).join('');
const bytes = b => b >= 2 ** 30 ? `${fmt(b / 2 ** 30, 2)} GB` : b >= 2 ** 20 ? `${fmt(b / 2 ** 20, 1)} MB` : b >= 1024 ? `${fmt(b / 1024, 1)} KB` : `${int(b)} B`;
const mb = v => v >= 1024 ? `${fmt(v / 1024, 2)} GB` : `${fmt(v, v % 1 ? 1 : 0)} MB`;
const sum = a => a.reduce((x, y) => x + y, 0);
const tb = v => `${+Number(v).toFixed(3)} TB`;
const big = n => n >= 1e9 ? `${fmt(n / 1e9, 1)}B` : n >= 1e6 ? `${fmt(n / 1e6, 1)}M` : n >= 1e3 ? `${fmt(n / 1e3, 1)}k` : int(n);

/* A small table: headers [[text, 'num'?]], rows of cells (strings, already escaped where needed). */
function table(heads, rows, { cls = '' } = {}) {
  const th = heads.map(h => Array.isArray(h) ? `<th class="${h[1]}">${h[0]}</th>` : `<th>${h}</th>`).join('');
  const body = rows.map(r => `<tr${r.cls ? ` class="${r.cls}"` : ''}>${(r.cells || r).map((c, i) => `<td${Array.isArray(heads[i]) ? ` class="${heads[i][1]}"` : ''}>${c}</td>`).join('')}</tr>`).join('');
  return `<div class="tbl-wrap ${cls}"><table><thead><tr>${th}</tr></thead><tbody>${body || `<tr><td colspan="${heads.length}" class="mr-empty">nothing</td></tr>`}</tbody></table></div>`;
}
const box = (title, meta, inner, cls = '') => `<div class="mr-box ${cls}"><h4><span>${title}</span>${meta ? `<small>${meta}</small>` : ''}</h4>${inner}</div>`;
const kv = (pairs, heads = ['Key', 'Value']) => table(heads, pairs.map(([k, v]) => [esc(k), esc(v)]));

/* Grouped bars over categories: series [{name, cls, values}]. */
function groupBars(host, labels, series, { h = 240, yl = '', yf } = {}) {
  const vals = series.flatMap(s => s.values);
  const hi = Math.max(1, ...vals) * 1.12;
  const P = frame(host, { h, x: [-0.6, labels.length - 0.4], y: [0, hi], xt: labels.map((_, i) => i), xf: i => labels[i] ?? '', yl, yf });
  const k = series.length, w = Math.min(0.8 / k, 0.34);
  series.forEach((s, j) => s.values.forEach((v, i) => {
    const x0 = i - (k * w) / 2 + j * w;
    if (v > 0) P.bars([x0 + 0.02, x0 + w - 0.02], [v], s.cls, 0);
  }));
  hover(host, P, px => { const i = Math.max(0, Math.min(labels.length - 1, Math.round(P.xInv(px)))); return { cross: P.X(i), html: tip(labels[i], series.map(s => [s.name, (yf || int)(s.values[i])])) }; });
  return P;
}

/* A row of stage pills across the top of an illustration, with the current one lit. */
function stagePills(svg, stages, cur, y = 6) {
  const w = 600 / stages.length;
  stages.forEach(([k, label], i) => {
    const on = k === cur;
    S('rect', { x: i * w + 3, y, width: w - 6, height: 24, rx: 12, class: on ? 'node hot' : 'node' }, svg);
    text(svg, i * w + w / 2, y + 16.5, label, on ? 'n-title mr-pill' : 'n-sub mr-pill');
  });
}

/* ================================================================ HDFS */
const HS = C.hdfs;
const hdfsStages = [['split', 'Split'], ['place', 'Place'], ['fail', 'Fail'], ['heal', 'Re-replicate']];
const hdfsKeys = ['files', 'size_mb', 'block_mb', 'replication', 'datanodes', 'racks', 'rack_aware', 'kill', 'fail_rack', 'read_tb', 'mbps', 'readers'];

const hdfs = {
  id: 'hdfs', group: 'bigdata', nav: 'HDFS blocks', tag: 'L3', title: 'HDFS: blocks, replicas and a dying DataNode',
  question: 'How does HDFS cut a file into blocks, where do the copies go, and what happens when a DataNode dies?',
  params: [
    { k: 'files', label: 'Files', min: 1, max: 4, step: 1, value: 3, group: 'Files', help: 'How many files to store. Slide 7 stores three: file1, file2, file3.' },
    { k: 'size_mb', label: 'Size of each file', min: 1, max: 4096, step: 1, value: 384, unit: 'MB', group: 'Files', help: 'The last block of a file holds whatever is left over, so a 500 MB file is 128 + 128 + 128 + 116 MB.' },
    { k: 'block_mb', type: 'seg', label: 'Block size', value: 128, options: HS.block_sizes.map(b => [b, `${b} MB`]), group: 'Files', help: 'HDFS default is 128 MB (64 MB in Hadoop 1, and the original GFS chunk size).' },
    { k: 'replication', label: 'Replication factor', min: 1, max: 5, step: 1, value: 3, group: 'Cluster', help: 'Copies of every block, each on a different DataNode. The default is 3.' },
    { k: 'datanodes', label: 'DataNodes', min: 3, max: 12, step: 1, value: 4, group: 'Cluster', help: 'Machines that store blocks. Slide 7 draws DN1, DN2, DN3 and DN-N.' },
    { k: 'racks', label: 'Racks', min: 1, max: 4, step: 1, value: 2, group: 'Cluster', help: 'DataNodes are split into racks in order (DN1, DN2 in rack A, …). A rack shares one switch and one power feed, so it can fail as a unit.' },
    { k: 'rack_aware', type: 'switch', label: 'Rack-aware placement', value: true, group: 'Cluster', help: 'On: Hadoop’s default policy puts the 2nd copy on another rack and the 3rd next to the 2nd. Off: the writer pipelines to the next nodes in order, ignoring racks.' },
    { k: 'kill', label: 'Kill DataNode', min: 0, max: 12, step: 1, value: 0, group: 'Failure', help: '0 = everything healthy. Otherwise the DataNode with this number stops sending heartbeats and the NameNode marks it dead.' },
    { k: 'fail_rack', type: 'switch', label: 'Take its whole rack down', value: false, group: 'Failure', show: s => s.kill > 0, help: 'A switch or power failure: every DataNode in the killed node’s rack disappears at once.' },
    { k: 'stage', type: 'seg', label: 'Stage to show', value: 'place', options: hdfsStages, group: 'Stage', help: 'Split: the blocks. Place: where the copies go. Fail: what is left after the failure. Re-replicate: after the NameNode repairs it.' },
    { k: 'read_tb', label: 'Data to read', min: 0.001, max: 1000, step: 0.001, value: 1, unit: 'TB', group: 'Read-time calculator', help: '1 TB = 2⁴⁰ bytes.' },
    { k: 'mbps', label: 'Disk throughput', min: 10, max: 5000, step: 10, value: 100, unit: 'MB/s', group: 'Read-time calculator', help: '1 MB/s = 2²⁰ bytes per second. A hard disk reads about 100 MB/s.' },
    { k: 'readers', label: 'Disks reading in parallel', min: 1, max: 10000, step: 1, value: 1, group: 'Read-time calculator', help: 'With the file spread over many DataNodes, each one reads its own blocks at the same time.' },
  ],
  presets: [
    { label: 'Slide 7 layout', values: { files: 3, size_mb: 384, block_mb: 128, replication: 3, datanodes: 4, racks: 2, rack_aware: true, kill: 0, stage: 'place' } },
    { label: '500 MB file (Spark p4)', values: { files: 1, size_mb: 500, block_mb: 128, replication: 3, datanodes: 3, racks: 1, kill: 0, stage: 'split' } },
    { label: 'Kill DN1', values: { files: 3, size_mb: 384, replication: 3, datanodes: 4, racks: 2, kill: 1, fail_rack: false, stage: 'heal' } },
    { label: 'Lose a rack, no rack awareness', values: { files: 3, size_mb: 384, replication: 3, datanodes: 6, racks: 2, rack_aware: false, kill: 1, fail_rack: true, stage: 'fail' } },
    { label: 'Lose a rack, rack-aware', values: { files: 3, size_mb: 384, replication: 3, datanodes: 6, racks: 2, rack_aware: true, kill: 1, fail_rack: true, stage: 'fail' } },
    { label: 'L1 slide 12: 1 TB, one disk', values: { read_tb: 1, mbps: 100, readers: 1 } },
    { label: '100 TB over 1000 disks', values: { read_tb: 100, mbps: 100, readers: 1000 } },
  ],
  request: (s, sig) => get('/api/bigdata/hdfs', s, sig, hdfsKeys),
  kpis: (d, s) => {
    const f = d.failure, rr = d.rereplication, rd = d.read;
    return [
      { label: 'Blocks stored', value: int(d.n_blocks), note: `last block ${mb(d.storage.last_block_mb)} · raw ${mb(d.storage.raw_mb)} for ${mb(d.storage.logical_mb)} of data (${d.effective_replication}×)` },
      { label: 'Under-replicated / lost', value: d.dead.length ? `${f.under} / ${f.lost}` : '0 / 0', tone: f.lost ? 'bad' : f.under ? 'warn' : 'good',
        note: d.dead.length ? `${d.dead.join(', ')} down · ${f.ok} blocks untouched` : 'all DataNodes alive: set “Kill DataNode”' },
      { label: 'Re-replication traffic', value: mb(rr.mb), note: d.dead.length ? `${rr.copies} block copies · ${mb(rr.cross_rack_mb)} across racks` : 'nothing to repair', tone: f.lost ? 'bad' : null },
      { label: 'Read time', value: rd.hours >= 1 ? fmt(rd.hours, 2) : fmt(rd.minutes, 1), unit: rd.hours >= 1 ? 'h' : 'min',
        note: `${tb(rd.read_tb)} at ${int(rd.mbps)} MB/s on ${int(rd.readers)} disk${rd.readers > 1 ? 's' : ''} · one disk ${fmt(rd.hours_one, 2)} h (${fmt(rd.days_one, 1)} days)` },
    ];
  },
  illustration: {
    title: 'NameNode and DataNodes', height: 230,
    caption: 'The NameNode keeps only the block map; DataNodes hold the blocks and report in with heartbeats. Box fill shows how full each DataNode is.',
    draw(svg, d, s) {
      stagePills(svg, hdfsStages, s.stage);
      const n = d ? d.params.datanodes : s.datanodes;
      const view = d ? d.nodes[s.stage === 'split' ? 'place' : s.stage] : null;
      const nn = node(svg, { x: 220, y: 40, w: 160, h: 50, title: 'NameNode', sub: d ? `${d.n_blocks} blocks · r = ${d.effective_replication}` : 'block map', hot: true });
      const gap = 6, w = Math.min(92, (600 - gap * (n + 1)) / n), maxMb = view ? Math.max(1, ...view.map(v => v.mb)) : 1;
      for (let i = 0; i < n; i++) {
        const v = view?.[i], x = gap + i * (w + gap), dead = d && d.dead.includes(`DN${i + 1}`) && s.stage !== 'place' && s.stage !== 'split';
        const b = node(svg, { x, y: 150, w, h: 56, title: `DN${i + 1}`, sub: dead ? 'down' : v ? `${v.blocks.length} blk` : '', fill: v && !dead ? v.mb / maxMb : 0, fillCls: dead ? 'fill2' : 'fill1' });
        edge(svg, [b.cx, b.y - 4], [nn.cx + (i - (n - 1) / 2) * 8, nn.b + 4], { width: dead ? 1 : 1.5, cls: dead ? 'st2' : '', arrow: !dead, particles: !dead && s.stage !== 'split' });
        if (dead) S('path', { d: `M${x + 4},${154} L${x + w - 4},${202} M${x + w - 4},${154} L${x + 4},${202}`, class: 'edge st2', 'stroke-width': 2 }, svg);
      }
      if (d) {
        const racks = [...new Set(d.nodes.place.map(v => v.rack))];
        racks.forEach(r => {
          const idx = d.nodes.place.map((v, i) => v.rack === r ? i : -1).filter(i => i >= 0);
          const x0 = gap + idx[0] * (w + gap), x1 = gap + idx[idx.length - 1] * (w + gap) + w;
          S('line', { x1: x0, x2: x1, y1: 214, y2: 214, class: 'ln-m', 'stroke-width': 2 }, svg);
          text(svg, (x0 + x1) / 2, 227, r);
        });
      }
      text(svg, 300, 122, 'heartbeats + block reports');
    },
  },
  charts: [
    { id: 'blockmap', kind: 'html', title: 'Block map',
      sub: (d, s) => ({ split: 'Each file cut into blocks; the last block keeps the remainder.',
        place: 'Which DataNode holds which block, by rack. Chips are block ids.',
        fail: 'After the failure: amber blocks lost a copy, red blocks lost every copy.',
        heal: 'After re-replication: green chips are the new copies the NameNode scheduled.' }[s.stage]),
      draw(host, d, s) {
        if (s.stage === 'split') {
          host.innerHTML = `<div class="mr-grid">${d.files.map(f => box(f.file, mb(f.mb), table(['Block', ['Size', 'num'], ['Bytes', 'num']],
            d.blocks.filter(b => b.file === f.file).map(b => [b.id, mb(b.mb), int(b.mb * 2 ** 20)])))).join('')}</div>`;
          return;
        }
        const status = Object.fromEntries(d.blocks.map(b => [b.id, b]));
        const before = Object.fromEntries(d.nodes.place.map(v => [v.id, new Set(v.blocks)]));
        const view = d.nodes[s.stage];
        const racks = [...new Set(view.map(v => v.rack))];
        host.innerHTML = racks.map(r => `<div class="mr-rack"><span class="eyebrow">${esc(r)}</span><div class="mr-grid">${view.filter(v => v.rack === r).map(v => {
          const dead = s.stage !== 'place' && d.dead.includes(v.id);
          const chips = (dead ? [...before[v.id]] : v.blocks).map(id => {
            const b = status[id];
            const cls = dead ? '' : s.stage === 'heal' && !before[v.id].has(id) ? 'good' : s.stage === 'fail' ? ({ under: 'warn', lost: 'bad' }[b.status_fail] || '') : '';
            return `<span class="chip ${cls}" title="${esc(b.file)} · ${mb(b.mb)}">${id}</span>`;
          }).join('');
          return box(v.id, dead ? 'down' : mb(v.mb), `<div class="mr-chips">${chips || '<span class="mr-empty">empty</span>'}</div>`, dead ? 'dead' : '');
        }).join('')}</div></div>`).join('') + (s.stage === 'fail' && d.failure.lost ? `<p class="mr-note">Lost blocks (${d.failure.lost}, ${mb(d.failure.lost_mb)}): ${d.blocks.filter(b => b.status_fail === 'lost').map(b => b.id).join(', ')}. Every copy was on the failed ${d.dead.length > 1 ? 'rack' : 'node'}.</p>` : '');
      } },
    { id: 'blocks', kind: 'html', title: 'Every block and its replicas', show: (d, s) => s.stage !== 'split',
      sub: 'Replicas as placed, what is still alive after the failure, and the replica set after repair.',
      draw(host, d) {
        const tag = st => `<span class="chip ${{ ok: 'good', under: 'warn', lost: 'bad' }[st]}">${{ ok: 'ok', under: 'under-replicated', lost: 'lost' }[st]}</span>`;
        host.innerHTML = table(['Block', 'File', ['Size', 'num'], 'Replicas', 'Racks', 'Alive after failure', 'Status', 'After repair'],
          d.blocks.map(b => ({ cls: b.status_fail === 'lost' ? 'mr-lost' : '', cells: [b.id, b.file, mb(b.mb), b.replicas.join(', '), b.racks.join(', ') + (b.single_rack ? ' <span class="chip warn">one rack</span>' : ''),
            d.dead.length ? (b.live.join(', ') || '—') : '—', tag(b.status_fail), d.dead.length ? (b.healed.join(', ') || '—') : '—'] })));
      } },
    { id: 'traffic', kind: 'html', title: 'Re-replication transfers', show: (d, s) => d.dead.length > 0 && s.stage !== 'split',
      sub: d => `${d.rereplication.copies} copies, ${mb(d.rereplication.mb)} in total, sent DataNode to DataNode (the NameNode only gives the orders).`,
      draw(host, d) {
        const m = d.rereplication.moves;
        host.innerHTML = m.length ? table(['Block', 'From', 'To', ['Size', 'num'], 'Crosses racks'], m.map(x => [x.block, x.from, x.to, mb(x.mb), x.cross_rack ? '<span class="chip warn">yes</span>' : 'no']))
          : `<p class="mr-note">Nothing can be copied: ${d.failure.lost ? 'the missing blocks have no surviving copy.' : 'every block still has its full set of copies.'}</p>`;
      } },
    { id: 'load', title: 'Storage per DataNode', sub: d => d.dead.length ? 'As placed, and after the dead node’s blocks were re-replicated.' : 'How evenly the placement policy spreads the bytes.',
      legend: d => [['s1', 'box', 'As placed'], ...(d.dead.length ? [['s3', 'box', 'After repair']] : [])],
      draw(host, d) {
        const series = [{ name: 'As placed (MB)', cls: 'f1', values: d.nodes.place.map(v => v.mb) }];
        if (d.dead.length) series.push({ name: 'After repair (MB)', cls: 'f3', values: d.nodes.heal.map(v => d.dead.includes(v.id) ? 0 : v.mb) });
        groupBars(host, d.nodes.place.map(v => v.id), series, { h: 240, yl: 'MB stored', yf: v => int(v) });
      } },
    { id: 'rackloss', title: 'Blocks lost when a whole rack fails', sub: d => `Worst rack, for each replication factor, with this cluster (${d.params.datanodes} DataNodes in ${d.params.racks} racks).`,
      legend: [['s3', 'box', 'Rack-aware'], ['s2', 'box', 'Not rack-aware']],
      draw(host, d) {
        const L = d.rack_loss;
        groupBars(host, L.replication.map(r => `r = ${r}`), [{ name: 'Rack-aware', cls: 'f3', values: L.aware }, { name: 'Not rack-aware', cls: 'f2', values: L.naive }], { h: 240, yl: 'Blocks lost' });
      } },
    { id: 'readtime', title: 'Read time against parallel disks', sub: d => `${tb(d.read.read_tb)} at ${int(d.read.mbps)} MB/s. Time = bytes ÷ (MB/s × 2²⁰) ÷ disks.`, legend: [['s1', 'line', 'Read time'], ['s2', 'dot', 'This setting']],
      draw(host, d) {
        const c = d.read.curve, lo = Math.min(...c.hours), hi = Math.max(...c.hours);
        const P = frame(host, { h: 260, x: [1, c.readers[c.readers.length - 1]], y: [lo / 1.5, hi * 1.5], xlog: true, ylog: true, xl: 'Disks reading in parallel', yl: 'Hours',
          xt: [1, 10, 100, 1000, 10000], yt: [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000].filter(v => v >= lo / 1.5 && v <= hi * 1.5), yf: v => v < 1 ? String(v) : int(v), xf: int });
        P.line(c.readers.map((r, i) => [r, c.hours[i]]), 's1');
        P.dot(d.read.readers, d.read.hours, 'f2', 5.5);
        hover(host, P, px => { const i = nearest(c.readers, P.xInv(px)); return { cross: P.X(c.readers[i]), pts: [[P.X(c.readers[i]), P.Y(c.hours[i]), 'f1']], html: tip(`${int(c.readers[i])} disks`, [['Hours', fmt(c.hours[i], 3)], ['Minutes', fmt(c.hours[i] * 60, 1)]]) }; });
      } },
  ],
  chartLayout: 'two',
  insights: (d, s) => [
    `${d.files.length} file${d.files.length > 1 ? 's' : ''} of ${mb(d.params.size_mb)} with ${int(d.params.block_mb)} MB blocks make <b>${d.n_blocks} blocks</b>; ×${d.effective_replication} copies use <b>${mb(d.storage.raw_mb)}</b> of disk.${d.replication_capped ? ' (Replication is capped at the number of DataNodes.)' : ''}`,
    d.dead.length ? (d.failure.lost ? `Losing ${d.dead.join(', ')} destroyed <b>${d.failure.lost}</b> block${d.failure.lost > 1 ? 's' : ''}: all their copies sat on the failed ${d.dead.length > 1 ? 'rack' : 'node'}. ${d.params.rack_aware ? '' : 'Rack-aware placement would have kept a copy elsewhere.'}`
      : `Losing ${d.dead.join(', ')} left <b>${d.failure.under}</b> blocks under-replicated and none lost; the NameNode restores them with ${d.rereplication.copies} copies (${mb(d.rereplication.mb)}).`)
      : `Every block has ${d.effective_replication} copies on different DataNodes${d.params.rack_aware && d.params.racks > 1 ? ' spanning at least two racks' : ''}. ${d.failure.single_rack_blocks ? `${d.failure.single_rack_blocks} blocks have every copy in one rack.` : ''}`,
    `Reading ${tb(d.read.read_tb)} from one ${int(d.read.mbps)} MB/s disk takes <b>${fmt(d.read.hours_one, 2)} h</b>${d.read.readers > 1 ? `; ${int(d.read.readers)} disks in parallel cut it to <b>${d.read.hours >= 1 ? fmt(d.read.hours, 2) + ' h' : fmt(d.read.minutes, 1) + ' min'}</b>` : ''}. L1 slide 12 writes 2⁴⁰/(100·60·60), which is really ${int(d.read.slide_formula_value)} “hours”: the 2²⁰ bytes in a MB is missing.`,
  ],
  maths: { eq: 'blocks per file   = ⌈size / block size⌉   (last block = size mod block size)\nraw storage       = size × replication\nread time (s)     = bytes / (MB/s × 2²⁰) / parallel disks\n1 TB at 100 MB/s  = 2⁴⁰ / (100 × 2²⁰) s = 10 486 s ≈ 2.9 h',
    notes: ['Rack-aware placement (Hadoop default): copy 1 on the writer’s node, copy 2 on another rack, copy 3 on another node of copy 2’s rack, at most (r − 1)/racks + 2 copies per rack.'] },
};

/* ================================================================ MapReduce */
const mrStages = [['split', 'Split'], ['map', 'Map'], ['combine', 'Combine'], ['partition', 'Partition'], ['shuffle', 'Shuffle/sort'], ['reduce', 'Reduce']];
const mrKeys = ['job', 'combiner', 'partitioner', 'reducers', 'group_cmp', 'natural_part', 'topn'];
const JOB = C.jobs;
const mapperName = (d, m) => d.job === 'index' ? `Mapper ${m + 1} · ${Object.keys(C.docs)[m]}` : d.job === 'wordcount' ? `Mapper ${m + 1} · split ${m + 1}` : `Mapper ${m + 1} · DN${m + 1}`;

const mapreduce = {
  id: 'mapreduce', group: 'bigdata', nav: 'MapReduce engine', tag: 'MR 1', title: 'MapReduce, one stage at a time',
  question: 'What does each phase of a MapReduce job send to which node, and what do combiners and partitioners change?',
  params: [
    { k: 'job', type: 'seg', label: 'Job', value: 'avg', group: 'Job', options: [['avg', 'Avg salary'], ['wordcount', 'Word count'], ['index', 'Inverted index'], ['topn', 'Top-N'], ['secsort', 'Secondary sort']],
      help: 'Average: GROUP BY dno over 30 employees (L3 slide 17). Word count: the IBM figure (MR p18/p30). Inverted index: word → documents. Top-N: the N biggest customers (MR 2 p24). Secondary sort: salaries per department in descending order (MR p46).' },
    { k: 'topn', label: 'N', min: 1, max: 9, step: 1, value: 3, group: 'Job', show: s => s.job === 'topn', help: 'How many customers to keep. Each mapper keeps a min-heap of size N.' },
    { k: 'combiner', type: 'seg', label: 'Combiner', value: 'none', group: 'Combiner', options: [['none', 'None'], ['reducer', 'Reducer as combiner'], ['proper', 'Correct combiner']],
      help: 'Runs on each mapper’s own output before the shuffle. Reusing the reducer is only safe when the reduce function is associative and commutative: SUM is, AVG is not. The correct AVG combiner sends <sum, count>.' },
    { k: 'partitioner', type: 'seg', label: 'Partitioner', value: 'range', group: 'Shuffle', options: [['hash', 'Hash mod R'], ['range', 'Range']],
      help: 'Hash: (key.hashCode() & MAX) mod R, Hadoop’s default. Range: words by first letter (A–I, J–Q, R–Z for 3 reducers); numbers by split points sampled from the keys so reducers get similar record counts.' },
    { k: 'reducers', label: 'Reducers R', min: 1, max: 6, step: 1, value: 2, group: 'Shuffle', help: 'Number of reduce tasks. Every key goes to exactly one of them.' },
    { k: 'group_cmp', type: 'switch', label: 'Grouping comparator on dno', value: true, group: 'Secondary sort', show: s => s.job === 'secsort', help: 'Groups sorted <dno, salary> keys by dno only, so reduce runs once per department. Off: once per (dno, salary).' },
    { k: 'natural_part', type: 'switch', label: 'Partition on dno only', value: true, group: 'Secondary sort', show: s => s.job === 'secsort', help: 'The partitioner must ignore the salary, or one department is scattered over several reducers.' },
    { k: 'stage', type: 'seg', label: 'Stage to show', value: 'shuffle', group: 'Stage', options: mrStages, help: 'Step through the pipeline: what every node holds after each phase.' },
  ],
  presets: [
    { label: 'Slide p6: avg, 2 reducers', values: { job: 'avg', combiner: 'none', partitioner: 'range', reducers: 2, stage: 'shuffle' } },
    { label: 'AVG reused as combiner (bug)', values: { job: 'avg', combiner: 'reducer', partitioner: 'range', reducers: 2, stage: 'reduce' } },
    { label: '<sum, count> combiner (fix)', values: { job: 'avg', combiner: 'proper', partitioner: 'range', reducers: 2, stage: 'combine' } },
    { label: 'Hash partitioner skew', values: { job: 'avg', combiner: 'none', partitioner: 'hash', reducers: 2, stage: 'partition' } },
    { label: 'WordCount (p18)', values: { job: 'wordcount', combiner: 'none', partitioner: 'range', reducers: 3, stage: 'shuffle' } },
    { label: 'WordCount + combiner (p30)', values: { job: 'wordcount', combiner: 'proper', partitioner: 'range', reducers: 3, stage: 'shuffle' } },
    { label: 'Inverted index', values: { job: 'index', combiner: 'proper', partitioner: 'range', reducers: 3, stage: 'reduce' } },
    { label: 'Top-3 customers', values: { job: 'topn', topn: 3, combiner: 'proper', reducers: 1, stage: 'map' } },
    { label: 'Secondary sort', values: { job: 'secsort', combiner: 'none', partitioner: 'hash', reducers: 2, group_cmp: true, natural_part: true, stage: 'shuffle' } },
  ],
  request: (s, sig) => get('/api/bigdata/mapreduce', s, sig, mrKeys),
  kpis: d => {
    const sh = d.shuffle, L = d.load, ok = d.check;
    const saved = sh.records_no_combiner ? sh.saved_records / sh.records_no_combiner : 0;
    const busiest = Math.max(...L.reducer_records);
    return [
      { label: 'Records shuffled', value: int(sh.records), note: d.combiner === 'none' && d.job !== 'topn' ? 'no combiner: every map output record crosses the network' : `${int(sh.records_no_combiner)} without combining · ${pct(saved, 0)} fewer`, tone: saved > 0.3 ? 'good' : null },
      { label: 'Bytes shuffled', value: int(sh.bytes), unit: 'B', note: `${int(sh.bytes_no_combiner)} B without combining (key⇥value⏎ text)` },
      { label: 'Busiest reducer', value: `${int(busiest)} / ${int(sh.records)}`, unit: 'records', tone: L.skew > 1.5 ? 'warn' : 'good',
        note: `skew ${fmt(L.skew, 2)}× the mean${L.idle_reducers ? ` · ${L.idle_reducers} idle` : ''} · key ${esc(L.hot_key)} has ${L.hot_key_records} of ${L.map_records} map records` },
      { label: 'Output check', value: `${ok.n_ok} / ${ok.n}`, unit: 'correct', tone: ok.all_ok ? 'good' : 'bad', note: ok.all_ok ? 'matches the answer computed without MapReduce' : 'differs from the true answer: see the check table' },
    ];
  },
  illustration: {
    title: 'Who sends what to whom', height: 250,
    caption: 'Mappers on the left, reducers on the right. Arrow widths and labels are the records each mapper sends to each reducer at the shuffle.',
    draw(svg, d, s) {
      stagePills(svg, mrStages, s.stage);
      if (!d) return;
      const M = d.stages.M, R = d.stages.R, maxT = Math.max(1, ...d.stages.traffic.flat().map(t => t.records));
      const yM = i => 46 + (i + 0.5) * (196 / M) - 22, yR = j => 46 + (j + 0.5) * (196 / R) - Math.min(22, 90 / R);
      const hR = Math.min(44, 180 / R);
      const maps = Array.from({ length: M }, (_, i) => {
        const out = (d.stages.combine || d.stages.map)[i].length;
        return node(svg, { x: 10, y: yM(i), w: 150, h: 44, title: `Mapper ${i + 1}`, sub: `${d.stages.splits[i].length} in → ${out} out`, hot: ['split', 'map', 'combine'].includes(s.stage) });
      });
      const reds = Array.from({ length: R }, (_, j) => node(svg, { x: 430, y: yR(j), w: 160, h: hR, title: R > 4 ? `R${j + 1}` : `Reducer ${j + 1}`, sub: R > 4 ? `${d.load.reducer_records[j]}` : `${d.load.reducer_records[j]} rec · ${d.load.reduce_calls[j]} calls`,
        hot: ['shuffle', 'reduce'].includes(s.stage), fill: d.load.reducer_records[j] / Math.max(1, d.shuffle.records), fillCls: 'fill2' }));
      maps.forEach((a, i) => reds.forEach((b, j) => {
        const t = d.stages.traffic[i][j].records;
        if (!t) return;
        edge(svg, [a.r + 4, a.cy], [b.x - 6, b.cy], { width: widthFor(t, maxT) * 0.8, flow: s.stage === 'partition' || s.stage === 'shuffle', label: M * R <= 6 ? String(t) : null, labelAt: [a.r + 30 + (b.x - a.r - 40) * 0.25, a.cy + (b.cy - a.cy) * 0.25 - 6] });
      }));
    },
  },
  charts: [
    { id: 'stage', kind: 'html', title: 'Data at this stage',
      sub: (d, s) => ({
        split: `${JOB[d.job].task} Each mapper reads one input split.`,
        map: 'Each mapper turns its records into (key, value) pairs.',
        combine: d.stages.combine ? 'The combiner runs on each mapper’s own output, one call per key, before anything is sent.' : 'No combiner: the map output goes to the partitioner unchanged.',
        partition: `Each pair is assigned a reducer: ${d.partitions.map((p, i) => `R${i + 1} ${esc(p)}`).join(' · ')}.`,
        shuffle: 'Each reducer merges what it received, sorts by key and groups equal keys: one row is one reduce call.',
        reduce: 'Each reduce call writes its output pairs.' }[s.stage]),
      draw(host, d, s) {
        const st = d.stages;
        let html = '';
        if (s.stage === 'split') html = st.splits.map((sp, m) => box(mapperName(d, m), `${sp.length} records`, table(['#', 'Record'], sp.map((r, i) => [i + 1, esc(r)])))).join('');
        else if (s.stage === 'map') html = st.map.map((p, m) => box(mapperName(d, m), `${p.length} pairs`, kv(p))).join('');
        else if (s.stage === 'combine') html = (st.combine || st.map).map((p, m) => box(mapperName(d, m), `${st.map[m].length} → ${p.length}`, kv(p))).join('');
        else if (s.stage === 'partition') html = st.partition.map((parts, m) => box(mapperName(d, m), parts.map(p => p.length).join(' / '),
          table(['Key', 'Value', 'To'], parts.flatMap((p, r) => p.map(([k, v]) => [esc(k), esc(v), `<span class="chip mr-r${r % 6}">R${r + 1}</span>`]))))).join('');
        else if (s.stage === 'shuffle') html = st.shuffle.map((g, r) => box(`Reducer ${r + 1}`, `${esc(d.partitions[r])} · ${g.length} calls`, table(['Key', 'Values (sorted)'], g.map(x => [esc(x.key), esc(x.values.join(', '))])))).join('');
        else html = st.reduce.map((p, r) => box(`Reducer ${r + 1}`, `${p.length} out`, kv(p))).join('');
        host.innerHTML = `<div class="mr-grid${s.stage === 'shuffle' || s.stage === 'reduce' ? ' wide' : ''}">${html}</div>` + (d.notes.length ? d.notes.map(n => `<p class="mr-note">${esc(n)}</p>`).join('') : '');
      } },
    { id: 'traffic', kind: 'html', title: 'Shuffle traffic matrix', sub: 'Records and bytes from every mapper to every reducer.',
      draw(host, d) {
        const T = d.stages.traffic, R = d.stages.R;
        const rows = T.map((row, m) => [`Mapper ${m + 1}`, ...row.map(t => t.records ? `${t.records} · ${t.bytes} B` : '—'), `${sum(row.map(t => t.records))} · ${sum(row.map(t => t.bytes))} B`]);
        rows.push({ cls: 'mr-total', cells: ['Received', ...d.load.reducer_records.map((r, j) => `${r} · ${d.load.reducer_bytes[j]} B`), `${d.shuffle.records} · ${d.shuffle.bytes} B`] });
        host.innerHTML = table(['', ...Array.from({ length: R }, (_, j) => [`R${j + 1}`, 'num']), ['Sent', 'num']], rows);
      } },
    { id: 'load', title: 'Records per reducer', sub: 'With this combiner, and without any combining.', legend: d => [['s1', 'box', 'This run'], ['s4', 'box', 'No combiner']],
      draw(host, d) {
        groupBars(host, d.load.reducer_records.map((_, j) => `R${j + 1}`), [{ name: 'This run', cls: 'f1', values: d.load.reducer_records }, { name: 'No combiner', cls: 'f4', values: d.shuffle.reducer_records_no_combiner }], { h: 240, yl: 'Records received' });
      } },
    { id: 'check', kind: 'html', title: 'Expected output', sub: d => `${JOB[d.job].slides}. Expected values come from a direct computation without MapReduce.`,
      draw(host, d) {
        host.innerHTML = table(['Key', ['Expected', 'num'], ['This job', 'num'], ''], d.check.rows.map(r => ({ cls: r.ok ? '' : 'mr-lost',
          cells: [esc(r.key), esc(r.expected), esc(r.got), r.ok ? '<span class="chip good">ok</span>' : `<span class="chip bad">${r.error != null ? 'off by ' + fmt(r.error, 2) : r.calls != null ? r.calls + ' rows' : 'wrong'}</span>`] })));
      } },
  ],
  chartLayout: 'two',
  insights: (d, s) => {
    const sh = d.shuffle, out = [];
    if (d.job === 'avg') out.push(d.combiner === 'reducer'
      ? `Reusing the AVG reducer as a combiner gives an <b>average of averages</b>: ${d.check.rows.filter(r => !r.ok).map(r => `dno ${r.key} comes out ${r.got} instead of ${r.expected}`).join('; ')}. AVG is not associative.`
      : d.combiner === 'proper' ? `Sending <b>&lt;sum, count&gt;</b> pairs makes the combiner safe: ${sh.records_no_combiner} records shrink to ${sh.records} and every average is exact.`
        : `Without a combiner all ${sh.records} (dno, salary) pairs cross the network. Key ${esc(d.load.hot_key)} alone has ${d.load.hot_key_records}, so one reducer does most of the work.`);
    if (d.job === 'wordcount') out.push(d.combiner === 'none' ? `Each mapper emits (word, 1) for every token: ${sh.records} records reach the reducers.` : `Summing per mapper first sends ${sh.records} records instead of ${sh.records_no_combiner}; the reducers then see lists like Hadoop [2, 1] and Map [4, 3].`);
    if (d.job === 'topn') out.push(`All map output keys are NULL, so one reducer merges the heaps. ${d.combiner === 'none' ? `Without the in-mapper heap all ${sh.records} records go to it.` : `Each mapper sends only its local top ${d.topn}: ${sh.records} records instead of ${sh.records_no_combiner}.`}`);
    if (d.job === 'secsort') out.push(d.check.all_ok ? `The sort comparator orders <dno, salary> by dno then salary descending; grouping on dno gives ${sum(d.load.reduce_calls)} reduce calls, one per department.` : `This run makes ${sum(d.load.reduce_calls)} reduce calls for 4 departments: ${!d.group_cmp ? 'without the grouping comparator every (dno, salary) is its own group.' : 'partitioning on the whole composite key scatters a department over reducers.'}`);
    if (d.job === 'index') out.push(`Map emits (word, document) for every occurrence; ${d.combiner === 'none' ? 'repeats such as “hdfs” twice in d3 are shuffled too.' : `the combiner removes repeats within a document, saving ${sh.saved_records} record${sh.saved_records === 1 ? '' : 's'}.`}`);
    out.push(`${d.partitioner === 'hash' ? 'Hash partitioning' : 'Range partitioning'} over ${d.R} reducer${d.R > 1 ? 's' : ''} gives loads ${d.load.reducer_records.join(' / ')} (skew ${fmt(d.load.skew, 2)}×). No partitioner can split a single key: all of key ${esc(d.load.hot_key)} must meet in one reduce call.`);
    return out;
  },
  maths: { eq: 'map:      (k1, v1)        → list(k2, v2)\ncombine:  (k2, list(v2))  → list(k2, v2)     per mapper, optional\npartition: k2             → reducer (hash(k2) & MAX) mod R\nreduce:   (k2, list(v2))  → list(k3, v3)\nAVG fix:  map → (dno, <salary, 1>), combine/reduce add both, avg = Σsum / Σcount',
    notes: ['A combiner may run zero, one or many times, so the result must not depend on it: reduce must be associative and commutative over the combiner’s output type.'] },
};

/* ================================================================ Joins */
const jStages = [['map', 'Map / tag'], ['shuffle', 'Shuffle'], ['reduce', 'Reduce'], ['output', 'Output']];
const jKeys = ['algo', 'join', 'reducers', 'extra', 'group_cmp', 'natural_part', 'partitioner', 'log_cust', 'log_orders', 'nodes', 'heap_mb'];
const J = C.joins;
const tagChip = v => esc(v).replace(/^&lt;(C|O),/, (m, t) => `&lt;<b class="mr-tag-${t}">${t}</b>,`);

const joins = {
  id: 'mr-joins', group: 'bigdata', nav: 'MapReduce joins', tag: 'MR 2', title: 'Joins in MapReduce: repartition and broadcast',
  question: 'How do you join customers with orders when both are spread over a cluster, and when should the small table be broadcast?',
  params: [
    { k: 'algo', type: 'seg', label: 'Algorithm', value: 'standard', group: 'Algorithm', options: [['standard', 'Repartition'], ['improved', 'Improved'], ['broadcast', 'Broadcast']],
      help: 'Repartition (reduce-side): tag every record and shuffle both tables on cid; the reducer buffers the orders (MR 2 p9–15). Improved: composite key (cid, tag) so the customer arrives first and orders stream (p16–20). Broadcast: copy the small customers table to every mapper; no shuffle (p21–22).' },
    { k: 'join', type: 'seg', label: 'Join type', value: 'inner', group: 'Algorithm', options: [['inner', 'Inner'], ['left', 'Left'], ['right', 'Right'], ['full', 'Full']],
      help: 'Left keeps orders without a customer, right keeps customers without orders, full keeps both. The slide’s p15 reducer is really a FULL OUTER join.' },
    { k: 'extra', type: 'switch', label: 'Add unmatched rows', value: false, group: 'Algorithm', help: `Adds customer ${J.extra_customer[0]} (no orders) and order ${J.extra_order[0]} for customer ${J.extra_order[1]} (not in the table), so the join types differ.` },
    { k: 'group_cmp', type: 'switch', label: 'Grouping comparator on cid', value: true, group: 'Improved join', show: s => s.algo === 'improved', help: 'Groups the sorted (cid, tag) keys by cid, so the customer and its orders reach one reduce call. The slides never mention it.' },
    { k: 'natural_part', type: 'switch', label: 'Partition on cid only', value: true, group: 'Improved join', show: s => s.algo === 'improved', help: 'The partitioner must hash the join key without the tag (p19).' },
    { k: 'partitioner', type: 'seg', label: 'Partitioner', value: 'hash', group: 'Shuffle', options: [['hash', 'Hash mod R'], ['range', 'Range']], show: s => s.algo !== 'broadcast' },
    { k: 'reducers', label: 'Reducers R', min: 1, max: 4, step: 1, value: 3, group: 'Shuffle', show: s => s.algo !== 'broadcast' },
    { k: 'stage', type: 'seg', label: 'Stage to show', value: 'shuffle', group: 'Stage', options: jStages },
    { k: 'log_cust', label: 'Customers (log₁₀ scale-up)', min: 0, max: 7, step: 0.1, value: 0, group: 'Scale it up', help: 'Customers = 3 × 10^x rows with the same record size as the slide’s.' },
    { k: 'log_orders', label: 'Orders (log₁₀ scale-up)', min: 0, max: 9, step: 0.1, value: 0, group: 'Scale it up', help: 'Orders = 9 × 10^y rows, spread evenly over the customers.' },
    { k: 'nodes', label: 'Cluster nodes', min: 1, max: 1000, step: 1, value: 10, group: 'Scale it up', help: 'The broadcast table is copied once to every node that runs map tasks over the orders.' },
    { k: 'heap_mb', label: 'Task memory', min: 64, max: 16384, step: 64, value: 1024, unit: 'MB', group: 'Scale it up', help: 'A broadcast join must fit the whole customers hash table in each mapper’s memory.' },
  ],
  presets: [
    { label: 'Slide p12: repartition, inner', values: { algo: 'standard', join: 'inner', extra: false, reducers: 3, stage: 'shuffle' } },
    { label: 'Slide p15 reducer = full outer', values: { algo: 'standard', join: 'full', extra: true, stage: 'output' } },
    { label: 'Improved (p17–20)', values: { algo: 'improved', join: 'inner', group_cmp: true, natural_part: true, stage: 'shuffle' } },
    { label: 'Improved, no grouping comparator', values: { algo: 'improved', join: 'inner', group_cmp: false, natural_part: true, stage: 'reduce' } },
    { label: 'Broadcast (p21–22)', values: { algo: 'broadcast', join: 'inner', stage: 'map' } },
    { label: 'Huge orders, small customers', values: { log_cust: 3, log_orders: 8, nodes: 100 } },
    { label: 'Both tables huge', values: { log_cust: 7, log_orders: 8.5, nodes: 100 } },
  ],
  request: (s, sig) => get('/api/bigdata/mr-joins', s, sig, jKeys),
  kpis: d => {
    const n = d.network, m = d.memory, sc = d.scale;
    return [
      { label: 'Joined rows', value: int(d.check.rows), unit: `of ${d.check.expected_rows}`, tone: d.check.ok ? 'good' : 'bad', note: d.check.ok ? `matches the ${d.join} join computed directly` : `${d.check.missing.length} missing · ${d.check.spurious.length} wrong` },
      d.algo === 'broadcast' ? { label: 'Network', value: int(n.broadcast_bytes), unit: 'B', note: `customers table copied to ${d.stages.M} mappers · 0 B shuffled` }
        : { label: 'Shuffled', value: int(n.shuffled_bytes), unit: 'B', note: `${n.shuffled_records} tagged records, both tables` },
      { label: 'Peak memory', value: int(m.peak_records), unit: 'records', note: `${int(m.peak_bytes)} B · ${esc(m.where)}`, tone: d.algo === 'improved' ? 'good' : null },
      { label: 'At the chosen scale', value: sc.winner === 'broadcast' ? 'Broadcast' : 'Repartition', tone: sc.broadcast_fits ? null : 'warn',
        note: `${big(sc.customers)} customers × ${big(sc.orders)} orders: repartition ${bytes(sc.now.repartition_bytes)}, broadcast ${bytes(sc.now.broadcast_bytes)}${sc.broadcast_fits ? '' : ' (does not fit in memory)'}` },
    ];
  },
  illustration: {
    title: 'Where the records travel', height: 240,
    caption: 'Repartition joins shuffle both tables by cid; the broadcast join ships the customers table to every mapper and shuffles nothing.',
    draw(svg, d, s) {
      stagePills(svg, jStages, s.stage);
      const cust = node(svg, { x: 10, y: 52, w: 140, h: 56, title: 'customers.csv', sub: `${J.customers.length + (s.extra ? 1 : 0)} rows (R)`, fill: 0.3, fillCls: 'fill1' });
      const ord = node(svg, { x: 10, y: 160, w: 140, h: 56, title: 'orders.csv', sub: `${J.orders.length + (s.extra ? 1 : 0)} rows (L)`, fill: 0.8, fillCls: 'fill2' });
      if (s.algo === 'broadcast') {
        const maps = [0, 1, 2].map(i => node(svg, { x: 250, y: 46 + i * 62, w: 150, h: 50, title: `Mapper ${i + 1}`, sub: 'hash table HR', hot: s.stage === 'map' }));
        maps.forEach(m => { edge(svg, [cust.r + 4, cust.cy], [m.x - 6, m.cy - 8], { width: 2, cls: 'st1', flow: true }); edge(svg, [ord.r + 4, ord.cy], [m.x - 6, m.cy + 8], { width: 3, cls: 'st2' }); });
        const out = node(svg, { x: 460, y: 100, w: 130, h: 64, title: 'Output', value: d ? String(d.check.rows) : null, sub: 'no reducers', hot: s.stage === 'output' });
        maps.forEach(m => edge(svg, [m.r + 4, m.cy], [out.x - 6, out.cy], { width: 2 }));
        return;
      }
      const R = s.reducers, maps = node(svg, { x: 205, y: 96, w: 120, h: 72, title: 'Mappers', sub: s.algo === 'improved' ? 'key (cid, tag)' : 'key cid + tag', hot: s.stage === 'map' });
      edge(svg, [cust.r + 4, cust.cy], [maps.x - 6, maps.cy - 10], { width: 2, cls: 'st1' }); edge(svg, [ord.r + 4, ord.cy], [maps.x - 6, maps.cy + 10], { width: 3, cls: 'st2' });
      const hR = Math.min(52, 170 / R);
      for (let j = 0; j < R; j++) {
        const rec = d ? d.stages.stats.reducer_records[j] : null;
        const y = 44 + (j + 0.5) * (190 / R) - hR / 2;
        const r = node(svg, { x: 410, y, w: 180, h: hR, title: `Reducer ${j + 1}`, sub: rec != null ? `${rec} rec · ${d.stages.stats.reduce_calls[j]} calls` : '', hot: s.stage === 'shuffle' || s.stage === 'reduce', fill: rec != null ? rec / Math.max(1, d.network.shuffled_records) : 0, fillCls: 'fill4' });
        if (rec !== 0) edge(svg, [maps.r + 4, maps.cy], [r.x - 6, r.cy], { width: rec != null ? widthFor(rec, d.network.shuffled_records) * 0.7 : 2, flow: s.stage === 'shuffle', label: rec != null && R <= 3 ? String(rec) : null });
      }
    },
  },
  charts: [
    { id: 'stage', kind: 'html', title: 'Data at this stage',
      sub: (d, s) => d.algo === 'broadcast' ? ({ map: 'map_init() loads customers into a hash table HR; each mapper then looks up every order in it.', shuffle: 'A broadcast join is map-only: there is no shuffle.', reduce: 'A broadcast join is map-only: there are no reducers.', output: 'Map output is the join result.' }[s.stage])
        : ({ map: d.algo === 'improved' ? 'Each record is tagged and keyed by the composite (cid, tag). C sorts before O.' : 'Each record is tagged C (customer) or O (order) and keyed by cid.',
          shuffle: d.algo === 'improved' ? 'Sorted by (cid, tag), grouped by ' + (d.group_cmp ? 'cid: the customer comes first in every list.' : '(cid, tag): customer and orders are separate calls!') : 'Grouped by cid. The customer record can land anywhere in the list, so the reducer must buffer.',
          reduce: 'One row per reduce call: how much the reducer had to hold in memory, and what it wrote.', output: 'The joined rows, checked against the join computed directly.' }[s.stage]),
      draw(host, d, s) {
        const st = d.stages;
        let html = '';
        if (s.stage === 'output') {
          const miss = new Set(d.check.missing.map(r => r.join('|'))), bad = new Set(d.check.spurious.map(r => r.join('|')));
          html = `<div class="mr-grid wide">${box('This job', `${d.output.length} rows`, table(d.columns.map(c => [c, 'num']), d.output.map(r => ({ cls: bad.has(r.join('|')) ? 'mr-lost' : '', cells: r.map(esc) }))))}`
            + box(`Expected (${d.join} join)`, `${d.expected.length} rows`, table(d.columns.map(c => [c, 'num']), d.expected.map(r => ({ cls: miss.has(r.join('|')) ? 'mr-lost' : '', cells: r.map(esc) })))) + '</div>';
        } else if (d.algo === 'broadcast') {
          if (s.stage === 'map') html = `<div class="mr-grid">${box('Hash table HR (every mapper)', `${st.hash_table.length} customers`, table(['cno → customer'], st.hash_table.map(r => [esc(r)])))}`
            + st.splits.map((sp, m) => box(esc(st.split_names[m]), `${sp.length} orders → ${st.map[m].length} rows`, table(['Order', 'Joined row'], sp.map((o, i) => [esc(o), esc(st.map[m][i]?.[1] ?? '—')])))).join('') + '</div>';
          else html = `<p class="mr-note">Map-only job: ${s.stage === 'shuffle' ? '0 records and 0 bytes are shuffled' : 'no reduce phase runs'}. The cost moved to the ${int(d.network.broadcast_bytes)} B spent copying customers to ${st.M} mappers.</p>`;
        } else if (s.stage === 'map') html = `<div class="mr-grid">${st.map.map((p, m) => box(esc(st.split_names[m]), `${p.length} pairs`, table(['Key', 'Tagged value'], p.map(([k, v]) => [esc(k), tagChip(v)])))).join('')}</div>`;
        else if (s.stage === 'shuffle') html = `<div class="mr-grid wide">${st.shuffle.map((g, r) => box(`Reducer ${r + 1}`, `${esc(st.partitions[r])} · ${g.length} calls`, table(['Group key', 'Values in arrival order'], g.map(x => [esc(x.key), x.values.map(tagChip).join(', ')])))).join('')}</div>`;
        else {
          let ci = 0;
          html = `<div class="mr-grid wide">${st.shuffle.map((g, r) => box(`Reducer ${r + 1}`, `${st.reduce[r].length} rows out`, table(['Reduce call key', ['Records held in memory', 'num']],
            g.map(x => { const c = d.calls[ci++]; return [esc(x.key), c ? String(c.buffered) : '—']; }))
            + kv(st.reduce[r].map(([k, v]) => [k, v]), ['Key', 'Joined row']))).join('')}</div>`;
        }
        host.innerHTML = html + d.notes.map(n => `<p class="mr-note">${esc(n)}</p>`).join('');
      } },
    { id: 'traffic', kind: 'html', title: 'Shuffle traffic matrix', show: d => d.algo !== 'broadcast', sub: 'Tagged records and bytes from each map task to each reducer.',
      draw(host, d) {
        const T = d.stages.traffic, R = d.stages.R;
        const rows = T.map((row, m) => [esc(d.stages.split_names[m]), ...row.map(t => t.records ? `${t.records} · ${t.bytes} B` : '—')]);
        rows.push({ cls: 'mr-total', cells: ['Received', ...d.stages.stats.reducer_records.map((r, j) => `${r} · ${d.stages.stats.reducer_bytes[j]} B`)] });
        host.innerHTML = table(['', ...Array.from({ length: R }, (_, j) => [`R${j + 1}`, 'num'])], rows);
      } },
    { id: 'compare', title: 'Three strategies on the slide data', sub: 'Bytes on the network and peak records held in memory by one task.',
      legend: [['s1', 'box', 'Network bytes'], ['s2', 'box', 'Peak memory of one task (bytes)']],
      draw(host, d) {
        const c = d.compare;
        groupBars(host, c.map(x => x.label), [{ name: 'Network bytes', cls: 'f1', values: c.map(x => x.network_bytes) }, { name: 'Peak memory (bytes)', cls: 'f2', values: c.map(x => x.peak_bytes) }], { h: 240, yl: 'Bytes' });
      } },
    { id: 'scale', title: 'When does broadcasting win?', sub: d => `Network bytes against the number of customers, with ${big(d.scale.orders)} orders on ${d.scale.nodes} nodes. Below the crossing, copy the small table; above it, shuffle.`,
      legend: [['s2', 'line', 'Repartition (shuffle both)'], ['s1', 'line', 'Broadcast customers'], ['s4', 'dot', 'This setting']],
      draw(host, d) {
        const c = d.scale.curve, xs = c.customers;
        const all = c.repartition_gb.concat(c.broadcast_gb).filter(v => v > 0);
        const lo = Math.max(1e-9, Math.min(...all)), hi = Math.max(...all);
        const P = frame(host, { h: 260, x: [xs[0], xs[xs.length - 1]], y: [lo / 2, hi * 2], xlog: true, ylog: true, xl: 'Customers (rows)', yl: 'GB on the network',
          xt: [10, 1e3, 1e5, 1e7].filter(v => v <= xs[xs.length - 1]), xf: big, yt: [1e-6, 1e-4, 1e-2, 1, 100, 1e4].filter(v => v >= lo / 2 && v <= hi * 2), yf: v => v >= 1 ? int(v) : String(v) });
        P.line(xs.map((x, i) => [x, c.repartition_gb[i]]), 's2');
        P.line(xs.map((x, i) => [x, c.broadcast_gb[i]]), 's1');
        const lim = d.scale.max_broadcast_customers;
        if (lim < xs[xs.length - 1]) { P.vline(lim); P.text(P.X(lim) + 4, P.M.t + 12, 'memory limit'); }
        const nowY = d.scale.winner === 'broadcast' ? d.scale.now.broadcast_bytes : d.scale.now.repartition_bytes;
        P.dot(d.scale.customers, nowY / 2 ** 30, 'f4', 5.5);
        hover(host, P, px => { const i = nearest(xs, P.xInv(px)); return { cross: P.X(xs[i]), pts: [[P.X(xs[i]), P.Y(c.repartition_gb[i]), 'f2'], [P.X(xs[i]), P.Y(c.broadcast_gb[i]), 'f1']], html: tip(`${big(xs[i])} customers`, [['Repartition', bytes(c.repartition_gb[i] * 2 ** 30)], ['Broadcast', bytes(c.broadcast_gb[i] * 2 ** 30)]]) }; });
      } },
  ],
  chartLayout: 'two',
  insights: (d, s) => [
    d.algo === 'standard' ? `The reducer cannot know where the customer record sits in its list, so it buffers every order of the key: up to <b>${d.memory.peak_records}</b> records here, and all orders of the biggest customer at scale.`
      : d.algo === 'improved' ? (d.check.ok ? `Sorting on (cid, tag) puts the C record first, so the reducer holds <b>one</b> customer and streams the orders past it.` : `This run is broken: ${d.notes.map(esc).join(' ')}`)
        : `The broadcast join shuffles nothing; it copies the ${J.customers.length + (d.extra ? 1 : 0)}-row customers table to each of ${d.stages.M} mappers (${int(d.network.broadcast_bytes)} B).`,
    `The slide’s reducer (MR 2 p15) emits customers with no orders and orders with no customer, padded with NULLs: it computes a <b>full outer</b> join, although the SQL on p11 is an inner JOIN. ${d.extra ? `With the unmatched rows on, inner gives ${d.compare.find(x => x.algo === d.algo)?.rows_by_join.inner} rows and full gives ${d.compare.find(x => x.algo === d.algo)?.rows_by_join.full}.` : 'Turn on “Add unmatched rows” to see the difference.'}`,
    d.scale.crossover_customers ? `With ${big(d.scale.orders)} orders on ${d.scale.nodes} nodes, broadcasting is cheaper up to about <b>${big(d.scale.crossover_customers)}</b> customers${d.scale.max_broadcast_customers < d.scale.crossover_customers ? `, but memory caps it at ${big(d.scale.max_broadcast_customers)}` : ''}.`
      : `With only ${d.scale.now.maps} map task${d.scale.now.maps > 1 ? 's' : ''}, the customers table is copied ${d.scale.now.copies}×, which is always cheaper than shuffling both tables; scale the orders up to see a crossover.`,
  ],
  maths: { eq: 'repartition network ≈ |R|·b_R + |L|·b_L          (both tables shuffled)\nbroadcast network   ≈ |R|·b_R × min(nodes, map tasks)\nmap tasks           = ⌈|L| bytes / 128 MB⌉\nbroadcast wins when |R|·b_R·(copies − 1) < |L|·b_L, and |R| fits in task memory',
    notes: ['L = orders (many rows per key), R = customers (one row per key). Peak reducer memory: standard = all orders of one key; improved = one customer record; broadcast = the whole of R in every mapper.'] },
};

export default [hdfs, mapreduce, joins];
