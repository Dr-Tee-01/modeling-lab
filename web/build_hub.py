"""Assemble web/index.html from hub_template.html plus the data each tab needs.

Inputs:  results/crossover.json         (python crossover/crossover.py)
         labs/markets/results.json      (stock-lab walk-forward results)
         labs/shape/outlines.json       (ShapeLab Fourier outlines)
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB = ROOT / "web"

cross = json.loads((ROOT / "results/crossover.json").read_text())
mk = json.loads((ROOT / "labs/markets/results.json").read_text())
outl = json.loads((ROOT / "labs/shape/outlines.json").read_text())

ret, vol = mk["returns"]["scores"]["pooled"], mk["volatility"]["scores"]["pooled"]
data = {
    "cross": cross,
    # Influenza in an English boarding school, Jan 1978 (BMJ): boys confined to bed, days 1-14.
    "flu": [3, 8, 26, 76, 225, 298, 258, 233, 189, 128, 68, 29, 14, 4],
    "bird": outl["bird1"]["points"],
    "markets": {
        "ret": [{"k": k, "r2": ret[k]["r2_oos"], "hit": ret[k]["hit_rate"], "dm_p": ret[k]["dm_p"]}
                for k in ["hist_mean", "arima", "ridge", "gbm", "lstm", "gru", "ensemble"]],
        "always_up": ret["_always_up_hit_rate"],
        "vol": [{"k": k, "r2": vol[k]["r2_log"]} for k in ["hist21", "ewma", "garch", "har", "har_x", "gbm"]],
        "trap": mk["trap"]["scores"], "trap_hit": mk["trap"]["direction_hit"],
        "test_days": mk["data"]["test_ticker_days"],
    },
}
blob = json.dumps(data, separators=(",", ":"))
html = (WEB / "hub_template.html").read_text().replace("/*__DATA__*/null", blob)
(WEB / "index.html").write_text(html)
print("web/index.html", len(html) // 1024, "KB")
