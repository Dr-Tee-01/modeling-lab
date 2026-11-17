// Big Data Lab: the IE494 lessons as live simulations. Each topic file exports an array of
// workbench specs with group 'bigdata'; this index gathers them (in lesson order is set by the
// server's lab definition, not here).
const FILES = ['./bigdata/mapreduce.js', './bigdata/rpc.js', './bigdata/spark.js', './bigdata/sparksql.js', './bigdata/warehouse.js'];
const loaded = await Promise.allSettled(FILES.map(f => import(f)));
loaded.forEach((r, i) => { if (r.status === 'rejected' && !/Failed to fetch|Cannot find module|ERR_MODULE_NOT_FOUND|404/.test(String(r.reason))) console.error(FILES[i], r.reason); });
export default loaded.flatMap(r => (r.status === 'fulfilled' ? r.value.default : []));
