"""Generate the ModeLab AI app icon: an SVG plus PNG and ICO renderings, from one geometry.

    python tools/make_icons.py          # writes src/modlab/server/static/icons/*

The mark: a rounded tile in the brand gradient (blue → violet → cyan) carrying a white
S-shaped growth curve (the logistic/epidemic curve the models produce) ending in a data
point, and a small four-point spark for "AI". Rendering uses NumPy only (supersampled
signed-distance shapes), so no imaging library is needed.
"""
from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "src/modlab/server/static/icons"
U = 64.0                                                # design grid
RADIUS = 15.0                                           # tile corner radius
STOPS = [(0.0, (0x4F, 0x7C, 0xFF)), (0.55, (0x8B, 0x5C, 0xF6)), (1.0, (0x22, 0xD3, 0xEE))]
STROKE = 6.2
DOT = (49.5, 17.5, 5.6)                                 # x, y, r
SPARK = (47.0, 46.0, 8.5)                               # x, y, size


def curve(n: int = 160) -> np.ndarray:
    """A logistic S-curve from bottom left to the data point at top right."""
    x = np.linspace(13.0, DOT[0], n)
    y = 47.0 - (47.0 - DOT[1]) / (1.0 + np.exp(-(x - 30.0) / 5.2))
    y = y - (y[-1] - DOT[1])                            # end exactly on the dot
    return np.stack([x, y], 1)


# ----------------------------------------------------------------------------- SVG
def svg() -> str:
    pts = curve(48)
    d = "M" + " L".join(f"{x:.2f} {y:.2f}" for x, y in pts)
    sx, sy, s = SPARK
    k = s * 0.22                                        # waist of the spark
    spark = (f"M{sx} {sy - s} C{sx + k * .4} {sy - k} {sx + k} {sy - k * .4} {sx + s} {sy} "
             f"C{sx + k} {sy + k * .4} {sx + k * .4} {sy + k} {sx} {sy + s} "
             f"C{sx - k * .4} {sy + k} {sx - k} {sy + k * .4} {sx - s} {sy} "
             f"C{sx - k} {sy - k * .4} {sx - k * .4} {sy - k} {sx} {sy - s}Z")
    stops = "".join(f'<stop offset="{o}" stop-color="#{r:02X}{g:02X}{b:02X}"/>' for o, (r, g, b) in STOPS)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" role="img" aria-label="ModeLab AI">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">{stops}</linearGradient>
    <linearGradient id="sheen" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff" stop-opacity=".22"/><stop offset=".55" stop-color="#fff" stop-opacity="0"/></linearGradient>
  </defs>
  <rect width="64" height="64" rx="{RADIUS}" fill="url(#g)"/>
  <rect width="64" height="64" rx="{RADIUS}" fill="url(#sheen)"/>
  <path d="{d}" fill="none" stroke="#fff" stroke-width="{STROKE}" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="{DOT[0]}" cy="{DOT[1]}" r="{DOT[2]}" fill="#fff"/>
  <path d="{spark}" fill="#fff" fill-opacity=".92"/>
</svg>
"""


# ----------------------------------------------------------------------------- raster
def render(size: int, ss: int = 4) -> np.ndarray:
    """RGBA uint8 image of the icon at `size` px (supersampled `ss`×)."""
    n = size * ss
    c = (np.arange(n) + 0.5) / n * U
    X, Y = np.meshgrid(c, c)
    px = U / n                                          # one subpixel in design units

    # tile: rounded square
    qx, qy = np.abs(X - U / 2) - (U / 2 - RADIUS), np.abs(Y - U / 2) - (U / 2 - RADIUS)
    tile = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - RADIUS
    a_tile = np.clip(0.5 - tile / px, 0, 1)

    # gradient along the diagonal, plus a soft top sheen
    t = np.clip((X + Y) / (2 * U), 0, 1)
    rgb = np.zeros(X.shape + (3,))
    for (t0, c0), (t1, c1) in zip(STOPS, STOPS[1:]):
        m = (t >= t0) & (t <= t1)
        f = ((t - t0) / (t1 - t0))[m][:, None]
        rgb[m] = np.array(c0) * (1 - f) + np.array(c1) * f
    sheen = np.clip(1 - Y / (0.55 * U), 0, 1)[..., None] * 0.22
    rgb = rgb + (255 - rgb) * sheen

    # white marks: curve (distance to polyline), dot, spark
    pts = curve()
    dist = np.full(X.shape, np.inf)
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        vx, vy = x1 - x0, y1 - y0
        h = np.clip(((X - x0) * vx + (Y - y0) * vy) / (vx * vx + vy * vy), 0, 1)
        dist = np.minimum(dist, np.hypot(X - x0 - h * vx, Y - y0 - h * vy))
    a_curve = np.clip(0.5 - (dist - STROKE / 2) / px, 0, 1)
    a_dot = np.clip(0.5 - (np.hypot(X - DOT[0], Y - DOT[1]) - DOT[2]) / px, 0, 1)
    sx, sy, s = SPARK
    star = np.sqrt(np.abs(X - sx) / s) + np.sqrt(np.abs(Y - sy) / s)      # astroid: ≤ 1 inside
    a_spark = np.clip((1 - star) * s / (2 * px), 0, 1) * 0.92
    white = np.maximum.reduce([a_curve, a_dot, a_spark])
    rgb = rgb * (1 - white[..., None]) + 255 * white[..., None]

    img = np.concatenate([rgb, (a_tile * 255)[..., None]], axis=2)
    img = img.reshape(size, ss, size, ss, 4).mean(axis=(1, 3))           # downsample
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def png(img: np.ndarray) -> bytes:
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))
    chunk = lambda tag, data: struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def ico(pngs: dict[int, bytes]) -> bytes:
    """An ICO container holding PNG images (supported by every current browser)."""
    head = struct.pack("<HHH", 0, 1, len(pngs))
    offset, entries, blobs = 6 + 16 * len(pngs), b"", b""
    for size, data in sorted(pngs.items()):
        entries += struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(data), offset)
        blobs += data
        offset += len(data)
    return head + entries + blobs


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "favicon.svg").write_text(svg())
    small = {}
    for size, name in [(16, "favicon-16.png"), (32, "favicon-32.png"), (48, None), (180, "apple-touch-icon.png"),
                       (192, "icon-192.png"), (512, "icon-512.png")]:
        data = png(render(size))
        if size <= 48:
            small[size] = data
        if name:
            (OUT / name).write_bytes(data)
    (OUT / "favicon.ico").write_bytes(ico(small))
    (OUT / "site.webmanifest").write_text(json.dumps({
        "name": "ModeLab AI", "short_name": "ModeLab AI", "start_url": "/", "display": "standalone",
        "background_color": "#050A16", "theme_color": "#050A16",
        "icons": [{"src": "/static/icons/icon-192.png", "sizes": "192x192", "type": "image/png"},
                  {"src": "/static/icons/icon-512.png", "sizes": "512x512", "type": "image/png"},
                  {"src": "/static/icons/favicon.svg", "sizes": "any", "type": "image/svg+xml"}]}, indent=2) + "\n")
    print("wrote", ", ".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
