"""Copy each lab's standalone demo into web/ with a full document head and a
'back to Modeling Lab' bar, so all three ship as sub-pages of the hub."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB = ROOT / "web"

HEAD = ('<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')

BAR = """<div class="ml-back"><a href="index.html#{tab}">&larr; Modeling Lab</a><span>{label}</span></div>
<style>
.ml-back{{--ml-bg:#14181F;--ml-fg:#E8ECF2;--ml-mute:#9AA3B0;display:flex;gap:12px;align-items:center;flex-wrap:wrap;
  padding:8px 16px;background:var(--ml-bg);color:var(--ml-fg);font:500 13px/1.4 "IBM Plex Sans",system-ui,sans-serif}}
.ml-back a{{color:var(--ml-fg);text-decoration:none;font-weight:600}}
.ml-back a:hover,.ml-back a:focus-visible{{text-decoration:underline}}
.ml-back span{{color:var(--ml-mute)}}
</style>
"""

PAGES = [
    # source, published name, hub tab, label, marker the bar goes before, replacements
    ("labs/spread/spread-lab.html", "spread.html", "spread", "Pillar 1 · Spread (MC312 labs)",
     '<div class="wrap">\n  <header class="top">', []),
    ("labs/shape/shapelab-demo.html", "shape.html", "shape", "Pillar 2 · Shape (ShapeLab)",
     '<div class="preview-banner">',
     [('<div class="preview-banner"><div class="wrap"><b>Concept preview</b><span>A mock-up of the finished project. '
       'Every demo below runs on your real lab data, computed in the browser or precomputed from the lab files.</span></div></div>',
       '<div class="preview-banner"><div class="wrap"><b>Demo</b><span>Every panel below runs on the real lab data, '
       'computed in the browser or precomputed by the Python library.</span></div></div>')]),
    ("labs/markets/markets-demo.html", "markets.html", "markets", "Pillar 3 · Markets (stock-lab)",
     '<header class="bar">', []),
]

for src, dst, tab, label, marker, repl in PAGES:
    html = (ROOT / src).read_text()
    for a, b in repl:
        assert a in html, (src, a[:40])
        html = html.replace(a, b)
    if not html.lstrip().lower().startswith("<!doctype"):
        html = html.replace('<meta charset="utf-8">\n', "", 1)
        html = HEAD + html
    assert html.count(marker) >= 1, (src, marker)
    html = html.replace(marker, BAR.format(tab=tab, label=label) + marker, 1)
    (WEB / dst).write_text(html)
    print(dst, len(html) // 1024, "KB")
