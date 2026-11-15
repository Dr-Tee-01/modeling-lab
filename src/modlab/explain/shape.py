"""Explainers for the Shape section (IT575 labs 1–5). Figures reuse modlab.shape_service and shapelab.

3D meshes cannot be drawn in an explainer, so the figures use 2D views of them: side-on
projections of the vertices, plane cross-sections, spectra and profiles. Every number in
the prose and the notes is computed here from the files in data/shape.
"""
from __future__ import annotations

import functools
import math

import numpy as np
from scipy.sparse.csgraph import connected_components
import scipy.sparse as sp

from shapelab import curvature as CV
from shapelab import fourier, procrustes, skinning
from shapelab import primitives
from shapelab import spectral as SPC

from .. import shape_service as S
from .registry import explainer
from .registry import figure as _register


def figure(fid: str):
    """Register a figure, computed with numpy's spurious Accelerate matmul warnings silenced (see shape_service.quiet)."""
    def deco(fn):
        @functools.wraps(fn)
        def run(p):
            with S.quiet():
                return fn(p)
        _register(fid)(run)
        return fn
    return deco


# ===================================================================== shared helpers
def _flip(a) -> np.ndarray:
    """Image coordinates (y grows downward) to plot coordinates (y grows upward)."""
    a = np.asarray(a, dtype=float)
    return np.stack([a[..., 0], -a[..., 1]], axis=-1)


def _m(x: float, d: int = 2) -> str:
    """A number with a true minus sign."""
    return f"{x:.{d}f}".replace("-", "−")


def _cap(t: str) -> str:
    return t[:1].upper() + t[1:]


def _g(x: float, d: int = 3) -> str:
    """Significant-figure format with a true minus sign."""
    return f"{x:.{d}g}".replace("-", "−")


def _subs(n: int) -> str:
    return str(n).translate(str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉"))


def _pc(x: float, d: int = 0) -> str:
    return f"{100 * x:.{d}f}%"


def _and(xs) -> str:
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def _slice(V: np.ndarray, F: np.ndarray, axis: int, c: float) -> list[np.ndarray]:
    """Polylines where the plane x[axis] = c cuts a triangle mesh (closed loops where the mesh is closed).

    Each face crossing the plane contributes one segment between two edge points; edge points are
    keyed by their (sorted) vertex pair, so segments chain into loops by walking shared keys.
    """
    d = V[:, axis] - c
    d = np.where(d == 0, 1e-12, d)
    pos = d[F] > 0
    cut = F[pos.sum(1) % 3 != 0]
    if not len(cut):
        return []
    links: dict = {}
    for f in cut:
        keys = []
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            if (d[a] > 0) != (d[b] > 0):
                keys.append((min(a, b), max(a, b)))
        if len(keys) == 2:
            links.setdefault(keys[0], []).append(keys[1])
            links.setdefault(keys[1], []).append(keys[0])

    def point(k):
        a, b = k
        t = d[a] / (d[a] - d[b])
        return V[a] + t * (V[b] - V[a])

    seen, loops = set(), []
    for start in links:
        if start in seen:
            continue
        chain, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = [k for k in links[cur] if k != prev and k not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            seen.add(cur)
            chain.append(cur)
        if len(chain) > 4:
            loops.append(np.array([point(k) for k in chain]))
    return loops


def _sub(n: int, every: int) -> np.ndarray:
    return np.arange(0, n, every)


# ===================================================================== Fourier: figures
BIRDS = tuple(S.outline_names())


@functools.lru_cache(maxsize=8)
def _bird(name: str):
    s, _ = S.fourier_summary(name)
    b = np.asarray(s.points, dtype=float)
    return s, b, fourier.descriptors(b)


def _rebuild(name: str, k: int) -> np.ndarray:
    (br, _), _ = S.fourier_reconstruction(name, int(k))
    return np.asarray(br)


def _turn_deg(b: np.ndarray, i: int, w: int = 4) -> float:
    """How sharply the outline turns at point i: angle between the chords to points i-w and i+w."""
    n = len(b)
    u, v = b[i] - b[(i - w) % n], b[(i + w) % n] - b[i]
    c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


@figure("fourier-flow")
def _f_flow(p):
    s, b, _ = _bird("bird1")
    N, K = len(b), s.coefficients_for_10pct
    err = float(s.error_curve[K])
    return {
        "kind": "flow", "height": 210,
        "nodes": [
            {"id": "o", "x": 5, "y": 60, "w": 104, "h": 84, "title": "Outline", "value": f"{N}", "sub": "(x, y) points"},
            {"id": "z", "x": 126, "y": 60, "w": 104, "h": 84, "title": "Complex", "value": "z = x + iy", "sub": "one per point"},
            {"id": "f", "x": 247, "y": 60, "w": 104, "h": 84, "title": "FFT", "value": f"{N}", "sub": "coefficients Zₖ"},
            {"id": "k", "x": 368, "y": 60, "w": 104, "h": 84, "title": "Keep lowest", "value": f"K = {K}", "sub": "rest set to 0", "hot": True},
            {"id": "r", "x": 489, "y": 60, "w": 104, "h": 84, "title": "Inverse FFT", "value": _pc(err, 1), "sub": "relative error"},
        ],
        "edges": [{"from": "o", "to": "z"}, {"from": "z", "to": "f"}, {"from": "f", "to": "k", "flow": True},
                  {"from": "k", "to": "r", "flow": True}],
        "texts": [{"x": 300, "y": 30, "text": f"{S.outline_label('bird1')}: {N} points in, {K} complex numbers kept", "cls": "n-title"},
                  {"x": 300, "y": 180, "text": "The FFT and its inverse are exact; only the 'keep' step loses information.", "cls": "n-sub"}],
        "notes": [f"{N} points become {N} coefficients, of which {K} are kept: {N / K:.0f}× fewer numbers for {_pc(err, 1)} error."],
    }


@figure("fourier-ladder")
def _f_ladder(p):
    s, b, Z = _bird("bird1")
    c10 = s.coefficients_for_10pct
    ks = [2, 3, c10, 4 * c10]
    P = _flip(b)
    w = np.ptp(P[:, 0]) * 1.25
    paths, dots, errs = [], [], []
    top = P[:, 1].max() + 0.12 * np.ptp(P[:, 1])
    for j, k in enumerate(ks):
        R = _flip(_rebuild("bird1", k))
        off = np.array([j * w, 0.0])
        paths.append({"points": P + off, "color": "neutral", "closed": True, "width": 1, "thin": True,
                      **({"name": "Original outline"} if j == 0 else {})})
        paths.append({"points": R + off, "color": "s2", "closed": True, "width": 2.2,
                      **({"name": "Rebuilt from K coefficients"} if j == 0 else {})})
        e = float(s.error_curve[k])
        errs.append(e)
        dots.append({"x": float(P[:, 0].min() + j * w), "y": float(top), "color": "accent", "label": f"K = {k}: {_pc(e, 0)}"})
    return {
        "kind": "paths", "equal": True, "axes": False, "paths": paths, "dots": dots,
        "notes": [f"K = 2 adds the f = +1 circle, radius {abs(Z[1]) / len(b):.1f} px: small, because this outline is traced in the "
                  f"f = −1 direction. K = 3 adds that circle (radius {abs(Z[-1]) / len(b):.1f} px) and the pair draws an ellipse "
                  f"({_pc(errs[1], 0)} error).",
                  f"With K = {ks[2]} the bird is recognisable ({_pc(errs[2], 1)}); K = {ks[3]} brings the error to {_pc(errs[3], 1)}."],
    }


@figure("fourier-outline")
def _f_outline(p):
    name = p.choice("outline", "bird1", BIRDS)
    s, b, _ = _bird(name)
    N = len(b)
    K = p.num("keep", s.coefficients_for_10pct, 1, min(N, 200), integer=True)
    R = _rebuild(name, K)
    gaps = np.linalg.norm(b - R, axis=1)
    i = int(np.argmax(gaps))
    err = float(s.error_curve[K])
    return {
        "kind": "paths", "equal": True, "axes": False,
        "paths": [{"points": _flip(b), "color": "neutral", "closed": True, "width": 1.4, "thin": True, "name": "Original"},
                  {"points": _flip(R), "color": "s2", "closed": True, "width": 2.2, "name": f"K = {K}"}],
        "dots": [{"x": float(b[i, 0]), "y": float(-b[i, 1]), "color": "bad", "label": "largest gap"}],
        "notes": [f"K = {K} of {N} coefficients: relative error {_pc(err, 1)}, and {N} points are stored as {K} complex "
                  f"number{'s' if K > 1 else ''} ({N / K:.0f}× fewer)." if K < N else f"All {N} coefficients kept: the outline is rebuilt exactly.",
                  f"The largest gap is {gaps[i]:.1f} px at point {i + 1}, against a mean gap of {gaps.mean():.1f} px."],
    }


@figure("fourier-spectrum")
def _f_spectrum(p):
    fmax = 40
    f = np.r_[np.arange(-fmax, 0), np.arange(1, fmax + 1)]
    series, notes = [], []
    for j, name in enumerate(BIRDS):
        s, b, Z = _bird(name)
        N = len(b)
        mag = np.abs(Z)
        shape = mag.copy()
        shape[0] = 0
        y = mag[f % N] / shape.max()
        series.append({"name": S.outline_label(name), "color": ("s1", "s2", "s3", "s4")[j % 4], "y": y})
        E = mag ** 2
        freq = np.minimum(np.arange(N), N - np.arange(N))  # |frequency| of each FFT index
        rest = E[1:].sum()
        notes.append(f"{S.outline_label(name)}: the two coefficients at f = ±1 carry {_pc(E[freq == 1].sum() / rest, 0)} of the energy "
                     f"left after the centre; |f| ≤ 5 carries {_pc(E[(freq >= 1) & (freq <= 5)].sum() / rest, 1)}.")
    return {"kind": "lines", "x": f, "ylog": True, "xlabel": "Frequency f (turns around the outline)",
            "ylabel": "|Z_f| relative to the largest", "series": series, "markers": [{"x": 0, "label": "f = 0 (centre) left out"}],
            "notes": notes}


@figure("fourier-error")
def _f_error(p):
    tol = p.num("tol", 10, 1, 30) / 100
    K = np.arange(1, 201)
    series, markers, need = [], [], []
    for j, name in enumerate(BIRDS):
        s, _, _ = _bird(name)
        curve = np.asarray(s.error_curve)
        series.append({"name": S.outline_label(name), "color": ("s1", "s2", "s3", "s4")[j % 4], "y": curve[1:201]})
        k = int(np.argmax(curve <= tol))
        need.append((S.outline_label(name), k, len(curve) - 1))
        markers.append({"x": k, "label": f"{S.outline_label(name)}: {k}"})
    series.append({"name": f"Target {_pc(tol)}", "color": "neutral", "y": np.full(len(K), tol), "dash": True, "thin": True})
    tol2 = tol / 2
    k2 = [int(np.argmax(np.asarray(_bird(n)[0].error_curve) <= tol2)) for n in BIRDS]
    return {
        "kind": "lines", "x": K, "xlog": True, "ymin": 0, "xlabel": "Coefficients kept K (log scale)", "ylabel": "Relative error",
        "series": series, "markers": markers,
        "notes": [f"For {_pc(tol)} error: " + _and([f"{lab} needs K = {k} of {n}" for lab, k, n in need]) + ".",
                  f"Halving the target to {_pc(tol2, 1)} needs K = " + _and([str(k) for k in k2])
                  + f", {np.mean(k2) / np.mean([k for _, k, _ in need]):.1f}× as many on average."],
    }


@functools.lru_cache(maxsize=8)
def _turns(name: str) -> np.ndarray:
    _, b, _ = _bird(name)
    return np.array([_turn_deg(b, j) for j in range(len(b))])


@figure("fourier-gaps")
def _f_gaps(p):
    name = p.choice("outline", "bird1", BIRDS)
    s, b, _ = _bird(name)
    N = len(b)
    K = p.num("keep", s.coefficients_for_10pct, 1, 100, integer=True)
    g = np.linalg.norm(b - _rebuild(name, K), axis=1)
    i = int(np.argmax(g))
    top = np.sort(g ** 2)[::-1]
    share = top[: max(1, N // 10)].sum() / max(top.sum(), 1e-12)
    turn = _turns(name)
    return {
        "kind": "lines", "x": np.arange(1, N + 1), "ymin": 0, "xlabel": "Point along the outline", "ylabel": "Gap to the rebuilt outline (px)",
        "series": [{"name": f"Gap at K = {K}", "color": "s2", "y": g, "area": True}],
        "markers": [{"x": i + 1, "label": f"max {g[i]:.1f} px"}],
        "notes": [f"{S.outline_label(name)}: the worst {N // 10} points (10%) hold {_pc(share, 0)} of the squared error, {share / 0.1:.1f}× their even share.",
                  f"At the worst point the outline turns by {turn[i]:.0f}°, against a median turn of {np.median(turn):.0f}°"
                  + (": the remaining error sits at sharp corners, which need high frequencies." if K >= s.coefficients_for_10pct else
                     ". At this K the gaps still mostly reflect the missing overall shape.")],
    }


@functools.lru_cache(maxsize=1)
def _moved_bird():
    _, b, _ = _bird("bird1")
    th = np.radians(40)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return np.roll(1.7 * b @ R.T + np.array([50.0, -20.0]), len(b) // 3, axis=0)


def _invariants(Z: np.ndarray, fs) -> np.ndarray:
    return np.abs(Z[np.asarray(fs) % len(Z)]) / np.abs(Z[1])


@figure("fourier-invariance")
def _f_invariance(p):
    fs = [-3, -2, -1, 2, 3, 4, 5]
    _, _, Z1 = _bird("bird1")
    Zm = fourier.descriptors(_moved_bird())
    _, _, Z2 = _bird(BIRDS[1])
    a, m, c = _invariants(Z1, fs), _invariants(Zm, fs), _invariants(Z2, fs)
    return {
        "kind": "bars", "labels": [f"f = {f:+d}" for f in fs], "ylabel": "|Z_f| / |Z_1|",
        "series": [{"name": S.outline_label("bird1"), "color": "s1", "values": a},
                   {"name": f"{S.outline_label('bird1')} rotated 40°, scaled 1.7×, moved, new start point", "color": "s3", "values": m},
                   {"name": S.outline_label(BIRDS[1]), "color": "s2", "values": c}],
        "notes": [f"After the move the normalised magnitudes differ by at most {np.max(np.abs(a - m)):.1e}: the same shape gives the same numbers.",
                  f"{S.outline_label(BIRDS[1])} differs from {S.outline_label('bird1')} by up to {np.max(np.abs(a - c)):.2f}, so the descriptors tell the birds apart."],
    }


# ===================================================================== Fourier: text
_B1, _B1b, _ = _bird("bird1")
_B2 = _bird(BIRDS[1])[0]
_N1, _C1 = len(_B1b), _B1.coefficients_for_10pct
_cent = _B1b - _B1b.mean(0)
_CENTRED_ERR = float(np.linalg.norm(_B1b - _rebuild("bird1", _C1)) / np.linalg.norm(_cent))
_AXES = abs(abs(_bird("bird1")[2][1]) - abs(_bird("bird1")[2][-1])) / (abs(_bird("bird1")[2][1]) + abs(_bird("bird1")[2][-1]))
_CENTRED_10 = int(np.argmax([np.linalg.norm(_B1b - _rebuild("bird1", k)) / np.linalg.norm(_cent) <= 0.1 for k in range(1, 200)])) + 1

_OUTLINE_CHOICE = {"k": "outline", "label": "Outline", "options": [[n, S.outline_label(n)] for n in BIRDS], "value": "bird1"}

explainer("shape/fourier", {
    "title": "Fourier descriptors: drawing an outline with a few rotating circles",
    "kicker": "Geometry Lab · IT575 L1",
    "summary": f"{S.outline_label('bird1')} is traced with {_N1} points, yet {_C1} complex numbers redraw it to within 10%. "
               "Treat the outline as a sequence of complex numbers, take its Fourier transform, and keep only the slowest "
               "frequencies: the result is a compact, smooth description that can also ignore position, size and rotation.",
    "sections": [
        {"id": "question", "title": "The question", "blocks": [
            {"type": "p", "text": f"The two bird silhouettes are stored as closed chains of pixel positions: {S.outline_label('bird1')} "
                                  f"has {_N1} points and {S.outline_label(BIRDS[1])} has {len(_B2.points)}. That is a faithful record, "
                                  "but it is not a description: two tracings of the same bird starting at different pixels share "
                                  "no numbers at all."},
            {"type": "p", "text": "The question this lab asks: <i>how few numbers capture the shape, and can those numbers be "
                                  "made independent of where the bird sits, how big it is and which way it faces?</i>"},
            {"type": "cards", "items": [
                {"title": "Compact", "text": "A handful of coefficients instead of hundreds of points."},
                {"title": "Smooth", "text": "Dropping high frequencies removes the staircase of the pixel tracing."},
                {"title": "Comparable", "text": "After normalisation, the same shape gives the same numbers in any pose."},
            ]},
        ]},
        {"id": "idea", "title": "An outline as a list of complex numbers", "blocks": [
            {"type": "p", "text": "Write each point as one complex number, <b>z = x + iy</b>. Walking round the outline gives a "
                                  "sequence z₀, z₁, …, z_{N−1} that repeats after N steps, which is exactly what the discrete "
                                  "Fourier transform is built for. The transform rewrites the sequence as a sum of circles, each "
                                  "turning a whole number of times as you go round once."},
            {"type": "figure", "figure": "fourier-flow", "title": "From points to coefficients and back",
             "caption": "Each stage is computed from the bird's real outline. Only the 'keep' step throws information away."},
            {"type": "callout", "tone": "info", "title": "Why complex numbers?",
             "text": "Using x + iy lets one transform handle both coordinates at once. A single coefficient Z_f is a circle: "
                     "its size is the radius, its phase the starting angle, and f says how many turns it makes."},
        ]},
        {"id": "equations", "title": "The equations, term by term", "blocks": [
            {"type": "eq", "text": "Zₖ = Σₙ zₙ e^(−2πi kn/N)          (forward transform)\n"
                                   "ẑₙ = (1/N) Σ_{k kept} Zₖ e^(2πi kn/N)   (rebuild from the kept terms)\n"
                                   "‖z − ẑ‖² = (1/N) Σ_{k dropped} |Zₖ|²     (Parseval)",
             "where": [["zₙ", "the n-th outline point as x + iy, n = 0 … N−1"],
                       ["Zₖ", "the k-th Fourier coefficient; index k above N/2 stands for the negative frequency k − N"],
                       ["e^(2πi kn/N)", "a point going round the unit circle k times while n runs once round the outline"],
                       ["K", "the number of coefficients kept, taken in the order f = 0, +1, −1, +2, −2, …"],
                       ["‖z − ẑ‖", "the reconstruction error over all points; the relative error divides it by ‖z‖"]]},
            {"type": "p", "text": "Read the rebuild line as a machine of linked arms. Z₀/N is the centre of the outline. "
                                  "Z₊₁ is an arm turning once anticlockwise, Z₋₁ an arm turning once clockwise; together they trace "
                                  "an ellipse. Every further pair adds a smaller, faster arm that bends the ellipse into detail."},
            {"type": "figure", "figure": "fourier-ladder", "title": "Adding frequencies one at a time",
             "caption": "Thin line: the traced bird. Thick line: the outline rebuilt from the first K coefficients. Each label gives K and the error."},
            {"type": "quiz", "q": "What does the reconstruction look like with K = 1?",
             "options": ["A circle through the outline", "Every point at the outline's centre", "A straight line"], "answer": 1,
             "why": "K = 1 keeps only Z₀, the average of all the points. Every rebuilt point lands on that average: the centroid."},
        ]},
        {"id": "spectrum", "title": "Where the energy is", "blocks": [
            {"type": "p", "text": "Parseval's theorem says the squared size of the outline equals the squared sizes of its "
                                  "coefficients, shared out by frequency. So the error of dropping coefficients is simply the "
                                  "energy they carried, and the error for every K comes from one FFT."},
            {"type": "figure", "figure": "fourier-spectrum", "title": "Coefficient sizes by frequency",
             "caption": "Log scale. Low frequencies (near the middle) are orders of magnitude larger than high ones on either side."},
            {"type": "p", "text": "The spectrum is not symmetric: positive and negative frequencies are different circles, one "
                                  "turning each way. The pair at f = ±1 sets the basic ellipse: its half-axes are "
                                  "(|Z₊₁| + |Z₋₁|)/N and ||Z₊₁| − |Z₋₁||/N, so the closer the two are in size, the thinner the ellipse. "
                                  f"For {S.outline_label('bird1')} the short axis is {_AXES:.2f} of the long one."},
        ]},
        {"id": "choose", "title": "Choosing K", "blocks": [
            {"type": "figure", "figure": "fourier-error", "title": "Error against coefficients kept",
             "caption": "Move the target error. Markers show the smallest K that reaches it for each bird.",
             "controls": [{"k": "tol", "label": "Target error", "min": 1, "max": 30, "step": 1, "value": 10, "unit": "%"}]},
            {"type": "figure", "figure": "fourier-outline", "title": "A bird rebuilt from K coefficients",
             "caption": "Pick a bird and move K. The dot marks the point that is furthest from where it should be.",
             "controls": [_OUTLINE_CHOICE,
                          {"k": "keep", "label": "Coefficients kept K", "min": 1, "max": 100, "step": 1, "value": _C1}]},
            {"type": "try", "label": "Open the live model with a rough sketch", "params": {"outline": "bird1", "keep": 3}},
            {"type": "try", "label": f"Open {S.outline_label(BIRDS[1])} at 10% error",
             "params": {"outline": BIRDS[1], "keep": _B2.coefficients_for_10pct}},
        ]},
        {"id": "gaps", "title": "Where the error lives", "blocks": [
            {"type": "p", "text": "A truncated Fourier series is smooth by construction, so it cannot follow a sharp corner. "
                                  "The leftover error is not spread evenly round the outline; it piles up where the outline "
                                  "turns sharply: corners, notches and the tips of the feathers."},
            {"type": "figure", "figure": "fourier-gaps", "title": "Gap at each point of the outline",
             "caption": "Distance from each traced point to its rebuilt position. Peaks sit at corners.",
             "controls": [_OUTLINE_CHOICE, {"k": "keep", "label": "Coefficients kept K", "min": 1, "max": 100, "step": 1, "value": _C1}]},
            {"type": "callout", "tone": "warn", "title": "Two limits of the method",
             "text": "Truncating a Fourier series overshoots near sharp corners (the Gibbs effect), so small K rounds corners off "
                     "and wiggles beside them. And the method needs a single closed outline: a shape with holes or "
                     "several parts needs one series per contour."},
        ]},
        {"id": "invariance", "title": "Descriptors that ignore pose", "blocks": [
            {"type": "p", "text": "Fourier coefficients change in simple ways when the outline is moved, which makes it easy to "
                                  "strip those changes out."},
            {"type": "steps", "title": "Normalising the coefficients", "items": [
                "Translation adds a constant to every zₙ, which only changes Z₀. Drop Z₀.",
                "Scaling multiplies every coefficient by the same factor. Divide by |Z₁|.",
                "Rotation multiplies every coefficient by e^(iθ). Keep magnitudes |Z_f| and the phase is gone.",
                "Starting the tracing at a different point multiplies Z_f by e^(2πi f s/N): again a phase, removed by taking magnitudes.",
            ]},
            {"type": "figure", "figure": "fourier-invariance", "title": "Normalised magnitudes |Z_f| / |Z₁|",
             "caption": "The moved copy of the bird gives the same bars as the original; the other bird does not."},
            {"type": "quiz", "q": "You rotate an outline by 90° and trace it from a different starting pixel. Which numbers change?",
             "options": ["All the magnitudes |Z_f|", "Only the phases of the coefficients", "Only Z₀"], "answer": 1,
             "why": "Both rotation and a new starting point multiply each coefficient by a unit complex number, which changes its "
                    "phase but not its size."},
            {"type": "callout", "tone": "tip", "title": "The price of invariance",
             "text": "Throwing away phases also throws away information: different shapes can share the same magnitudes. "
                     "For recognition that is usually acceptable; for rebuilding the outline, keep the full coefficients."},
        ]},
        {"id": "server", "title": "How the server computes it", "blocks": [
            {"type": "steps", "items": [
                "Read the outline CSV as N rows of (x, y) and form z = x + iy.",
                "Take one FFT (numpy) to get all N coefficients.",
                "Sort the indices by |frequency|: 0, 1, N−1, 2, N−2, … and keep the first K.",
                "Zero the rest and take the inverse FFT to rebuild the points.",
                "For the error curve, sum the dropped energy from the top down (Parseval), giving the error for every K at once.",
            ]},
            {"type": "callout", "tone": "warn", "title": "What the 10% figure measures",
             "text": f"The relative error divides by ‖z‖, which includes the outline's position in the image. Measured against the "
                     f"centred outline instead, K = {_C1} leaves {_pc(_CENTRED_ERR, 0)} error, and reaching 10% needs K = {_CENTRED_10}."},
            {"type": "compare", "items": [
                {"title": "Fourier descriptors (this lab)", "points": ["One closed curve", "No correspondence needed between shapes",
                                                                       "Basis fixed in advance (sines and cosines)"]},
                {"title": "Statistical shape model (next lab)", "points": ["Labelled landmarks on every shape", "Needs a training set",
                                                                          "Basis learned from data (PCA modes)"]},
            ]},
        ]},
    ],
    "takeaways": [
        "An outline of N points becomes N complex numbers z = x + iy, and its FFT rewrites them as N rotating circles.",
        f"Keeping the K lowest frequencies smooths the outline; {S.outline_label('bird1')} needs {_C1} of {_N1} for 10% error.",
        "Parseval's theorem turns the error of any K into a sum of the dropped coefficients' energy.",
        "Error concentrates at sharp corners, which need high frequencies.",
        "Dropping Z₀, dividing by |Z₁| and keeping magnitudes gives descriptors that ignore position, size, rotation and starting point.",
    ],
    "glossary": [
        ["Fourier descriptor", "A Fourier coefficient of a closed outline, used to describe its shape."],
        ["DFT / FFT", "The discrete Fourier transform, and the fast algorithm that computes it in N log N steps."],
        ["Frequency f", "How many times a coefficient's circle turns while the outline is traversed once."],
        ["Parseval's theorem", "The energy of a signal equals the energy of its Fourier coefficients (up to a factor N)."],
        ["Relative error", "‖z − ẑ‖ / ‖z‖ over all points."],
        ["Gibbs effect", "Overshoot of a truncated Fourier series near a jump or corner."],
        ["Invariant", "A quantity that does not change under a transformation, such as rotation."],
    ],
}, order=1)


# ===================================================================== Jaw shape model: figures
@functools.lru_cache(maxsize=1)
def _jaws():
    """Raw jaws, all jaws aligned to the model mean, and every jaw's mode coordinates (in sd)."""
    jm, _ = S.jaw_model()
    raw, _ = S.jaw_landmarks()
    raw = np.asarray(raw, dtype=float)
    m = jm.model
    k = len(m.variances)
    with S.quiet():
        aligned = procrustes.align(m.mean, raw)
        dev = (aligned - m.mean).reshape(len(raw), -1)
        scores = dev @ m.modes.reshape(k, -1).T / np.sqrt(m.variances)
    if not np.all(np.isfinite(scores)):
        raise FloatingPointError("jaw scores are not finite")
    idx = np.linspace(0, len(raw) - 1, S.JAW_SAMPLE_COUNT).round().astype(int)
    return jm, raw, aligned, scores, idx


def _spread(shapes: np.ndarray) -> float:
    """RMS landmark distance from the collection's average shape, relative to that shape's RMS size."""
    avg = shapes.mean(0)
    size = np.sqrt(np.mean(np.sum((avg - avg.mean(0)) ** 2, axis=1)))
    return float(np.sqrt(np.mean(np.sum((shapes - avg) ** 2, axis=2))) / size)


JAW_STAGES = ("as drawn", "centred", "centred and scaled", "centred, scaled and rotated")


@functools.lru_cache(maxsize=4)
def _jaw_stage(stage: int) -> np.ndarray:
    jm, raw, aligned, _, idx = _jaws()
    if stage == 0:
        return raw
    if stage == 1:
        return raw - raw.mean(axis=1, keepdims=True)
    if stage == 2:
        return procrustes.normalize(raw)
    return aligned


@figure("jaw-align")
def _j_align(p):
    stage = p.num("stage", 0, 0, 3, integer=True)
    _, raw, _, _, idx = _jaws()
    X = _jaw_stage(stage)
    paths = [{"points": _flip(X[i]), "color": "neutral", "width": 1, "thin": True} for i in idx[1:]]
    paths.append({"points": _flip(X[idx[0]]), "color": "s2", "width": 2.4, "name": "One jaw, followed through the steps"})
    sp = [_spread(_jaw_stage(s)) for s in range(4)]
    return {
        "kind": "paths", "equal": True, "axes": stage == 0, "paths": paths,
        "notes": [f"Step {stage} of 3, jaws {JAW_STAGES[stage]}: they differ from their average by {_pc(sp[stage], 0)} of its size "
                  f"(all {len(raw)} jaws).",
                  "Spread at each step: " + ", ".join(f"{_pc(v, 0)}" for v in sp) + ". What is left after the last step is shape."],
    }


@functools.lru_cache(maxsize=1)
def _gpa_history():
    raw = _jaws()[1]
    shapes = procrustes.normalize(raw)
    mean, moves = shapes[0], []
    for _ in range(200):
        new = procrustes.normalize(procrustes.align(mean, shapes).mean(axis=0))
        new = new @ procrustes.optimal_rotation(mean, new)
        moved = float(np.sum((new - mean) ** 2))
        moves.append(moved)
        mean = new
        if moved < 1e-12:
            break
    # the same loop started from another jaw lands on the same mean, up to a rotation
    other = procrustes.procrustes_mean(raw, init=len(raw) // 2)
    other = other @ procrustes.optimal_rotation(mean, other)
    return np.array(moves), float(np.sqrt(np.sum((mean - other) ** 2)))


@figure("jaw-gpa")
def _j_gpa(p):
    moves, gap = _gpa_history()
    step = np.sqrt(moves)
    return {
        "kind": "lines", "x": np.arange(1, len(moves) + 1), "ylog": True, "xlabel": "Iteration",
        "ylabel": "How far the mean moved", "series": [{"name": "Change in the mean shape", "color": "s1", "y": step}],
        "markers": [{"x": len(moves), "label": "stop"}],
        "notes": [f"The mean moves {step[0]:.2g} in the first iteration and {step[-1]:.1g} in the last: the loop stops after "
                  f"{len(moves)} iterations, when the squared change falls below 10⁻¹².",
                  f"Started from jaw {len(_jaws()[1]) // 2 + 1} instead of jaw 1, it reaches the same mean, up to a rotation, "
                  f"to within {gap:.0e} (the mean has size 1)."],
    }


@figure("jaw-variance")
def _j_variance(p):
    jm = _jaws()[0]
    v = np.asarray(jm.model.explained)
    cum = np.cumsum(v)
    k95 = int(np.argmax(cum >= 0.95)) + 1 if cum[-1] >= 0.95 else None
    return {
        "kind": "bars", "labels": [str(i) for i in range(1, len(v) + 1)], "xlabel": "Mode", "ylabel": "Share of all variation (%)",
        "series": [{"name": "This mode", "color": "s1", "values": 100 * v},
                   {"name": "Running total", "color": "s2", "values": 100 * cum}],
        "notes": [f"Mode 1 alone explains {_pc(v[0], 1)}; three modes explain {_pc(cum[2], 1)}; all {len(v)} kept modes {_pc(cum[-1], 1)}.",
                  (f"{k95} modes pass 95%, " if k95 else "") + f"and from mode 4 on each mode adds less than {_pc(v[3:].max(), 1)}."],
    }


@figure("jaw-mode")
def _j_mode(p):
    mode = p.num("mode", 1, 1, 6, integer=True)
    b = p.num("b", 2, 0.5, 3)
    jm, _, aligned, _, idx = _jaws()
    m = jm.model
    sd = math.sqrt(float(m.variances[mode - 1]))
    plus, minus = m.mean + b * sd * m.modes[mode - 1], m.mean - b * sd * m.modes[mode - 1]
    w0, h0 = np.ptp(m.mean[:, 0]), np.ptp(m.mean[:, 1])
    shift = np.linalg.norm(plus - m.mean, axis=1)
    j = int(np.argmax(shift))
    paths = [{"points": _flip(aligned[i]), "color": "neutral", "width": 1, "thin": True} for i in idx]
    paths += [{"points": _flip(m.mean), "color": "neutral", "width": 2, "dash": True, "name": "Mean jaw"},
              {"points": _flip(plus), "color": "s2", "width": 2.4, "name": f"+{b:g} sd"},
              {"points": _flip(minus), "color": "s1", "width": 2.4, "name": f"−{b:g} sd"}]
    where = "at an end of the jawline" if j < 5 or j >= m.mean.shape[0] - 5 else "in the middle stretch of the jawline"

    def change(r, base, more, less):
        return f"about as {base}" if abs(r - 1) < 0.005 else f"{abs(r - 1):.0%} {more if r > 1 else less}"

    def peak(x):
        # sideways position of the top of the arch (mean x of the five highest landmarks), from the ends' midpoint
        top = np.argsort(x[:, 1])[:5]
        return (x[top, 0].mean() - (x[0, 0] + x[-1, 0]) / 2) / w0

    def desc(x):
        d = peak(x) - peak(m.mean)
        side = "about where it was" if abs(d) < 0.005 else f"{abs(d):.0%} of the width towards landmark {1 if d * (x[0, 0] - x[-1, 0]) > 0 else len(x)}"
        return (f"{change(np.ptp(x[:, 0]) / w0, 'wide', 'wider', 'narrower')} and {change(np.ptp(x[:, 1]) / h0, 'tall', 'taller', 'shorter')}, "
                f"with the top of the arch {side}")
    return {
        "kind": "paths", "equal": True, "axes": False, "paths": paths,
        "notes": [f"Mode {mode} explains {_pc(m.explained[mode - 1], 1)} of the variation. Against the mean, the jaw at +{b:g} sd is "
                  f"{desc(plus)}; at −{b:g} sd it is {desc(minus)}.",
                  f"The landmark that moves most is number {j + 1} of {m.mean.shape[0]}, {where}, by {shift[j] / w0:.1%} of the jaw's width."],
    }


@figure("jaw-scores")
def _j_scores(p):
    _, _, _, B, _ = _jaws()
    d = np.hypot(B[:, 0], B[:, 1])
    pts = [{"x": float(x), "y": float(y), "color": "warn" if r > 3 else "s1", "r": 3.2} for x, y, r in zip(B[:, 0], B[:, 1], d)]
    t = np.linspace(0, 2 * np.pi, 97)
    lines = [{"x": r * np.cos(t), "y": r * np.sin(t), "color": "neutral", "dash": r != 2, "thin": True,
              **({"name": "2 sd"} if r == 2 else {})} for r in (1, 2, 3)]
    within = float(np.mean(d <= 2))
    corr = float(np.corrcoef(B[:, 0], B[:, 1])[0, 1])
    return {
        "kind": "scatter", "equal": True, "points": pts, "lines": lines, "xlabel": "b₁ (sd along mode 1)", "ylabel": "b₂ (sd along mode 2)",
        "legend": [{"name": "Jaw within 3 sd", "color": "s1"}, {"name": "Jaw beyond 3 sd", "color": "warn"},
                   {"name": "1, 2 and 3 sd circles", "color": "neutral", "dash": True}],
        "notes": [f"{_pc(within, 1)} of the {len(B)} jaws lie within 2 sd in this plane; a Gaussian cloud would put {_pc(1 - math.exp(-2), 1)} there. "
                  f"{int(np.sum(d > 3))} jaws (highlighted) lie beyond 3 sd.",
                  f"The correlation between b₁ and b₂ is zero to within {max(abs(corr), 1e-16):.0e}: PCA scores are uncorrelated by construction."],
    }


@figure("jaw-rebuild")
def _j_rebuild(p):
    jm, _, aligned, B, _ = _jaws()
    m = jm.model
    n = len(aligned)
    i = p.num("jaw", 1, 1, n, integer=True) - 1
    k = p.num("modes", 3, 0, len(m.variances), integer=True)
    rec = m.sample(B[i, :k]) if k else m.mean
    dev = np.linalg.norm(aligned[i] - m.mean)
    err = np.linalg.norm(aligned[i] - rec) / dev
    # the same measure for every jaw, to put this one in context
    allrec = m.mean + np.tensordot(B[:, :k] * np.sqrt(m.variances[:k]), m.modes[:k], axes=1) if k else np.broadcast_to(m.mean, aligned.shape)
    allerr = np.linalg.norm((aligned - allrec).reshape(n, -1), axis=1) / np.linalg.norm((aligned - m.mean).reshape(n, -1), axis=1)
    return {
        "kind": "paths", "equal": True, "axes": False,
        "paths": [{"points": _flip(m.mean), "color": "neutral", "width": 1.6, "dash": True, "name": "Mean jaw"},
                  {"points": _flip(aligned[i]), "color": "s1", "width": 2.4, "name": f"Jaw {i + 1}, aligned"},
                  {"points": _flip(rec), "color": "s2", "width": 2.4, "name": f"Rebuilt from {k} mode{'s' if k != 1 else ''}"}],
        "notes": [f"Jaw {i + 1} sits at b = (" + ", ".join(f"{v:+.1f}".replace("-", "−") for v in B[i, :3]) + ") sd on modes 1–3. "
                  f"With {k} mode{'s' if k != 1 else ''}, {_pc(err, 0)} of its difference from the mean is left unexplained.",
                  f"Across all {n} jaws the median is {_pc(np.median(allerr), 0)}."],
    }


@figure("jaw-flow")
def _j_flow(p):
    jm = _jaws()[0]
    moves, _ = _gpa_history()
    cum = np.cumsum(jm.model.explained)
    nc = jm.model.mean.size
    return {
        "kind": "flow", "height": 200,
        "nodes": [
            {"id": "l", "x": 5, "y": 50, "w": 104, "h": 90, "title": "Landmarks", "value": f"{jm.n_shapes}", "sub": f"jaws × {jm.n_landmarks} points"},
            {"id": "a", "x": 126, "y": 50, "w": 104, "h": 90, "title": "Procrustes", "value": f"{len(moves)} its", "sub": "pose removed"},
            {"id": "c", "x": 247, "y": 50, "w": 104, "h": 90, "title": "Covariance", "value": f"{nc} × {nc}", "sub": "coordinates"},
            {"id": "e", "x": 368, "y": 50, "w": 104, "h": 90, "title": "PCA", "value": f"{len(cum)} modes", "sub": "largest kept"},
            {"id": "m", "x": 489, "y": 50, "w": 104, "h": 90, "title": "Model", "value": "3 numbers", "sub": f"{_pc(cum[2], 1)} of variation", "hot": True},
        ],
        "edges": [{"from": "l", "to": "a"}, {"from": "a", "to": "c"}, {"from": "c", "to": "e"}, {"from": "e", "to": "m", "flow": True}],
        "texts": [{"x": 300, "y": 175, "text": f"Each jaw: {nc} numbers in, b₁, b₂, b₃ out", "cls": "n-sub"}],
        "notes": [f"{jm.n_shapes} jaws of {jm.n_landmarks} landmarks each give {nc} coordinates per jaw; three PCA modes keep {_pc(cum[2], 1)} of their variation."],
    }


# ===================================================================== Jaw shape model: text
_JM = _jaws()[0]
_JV = np.asarray(_JM.model.explained)
_JC = np.cumsum(_JV)
_JNC = int(_JM.model.mean.size)

explainer("shape/jaw", {
    "title": f"A statistical shape model: how {_JM.n_shapes} jawlines vary",
    "kicker": "Geometry Lab · IT575 L2",
    "summary": f"Each of {_JM.n_shapes} jawlines is marked with the same {_JM.n_landmarks} landmarks, {_JNC} numbers per jaw. "
               "Procrustes alignment removes position, size and rotation; principal component analysis then finds the few "
               f"directions in which the jaws really differ. Three of them carry {_pc(_JC[2], 1)} of the variation.",
    "sections": [
        {"id": "question", "title": "The question", "blocks": [
            {"type": "p", "text": f"A jawline here is {_JM.n_landmarks} points, placed at corresponding positions on every jaw: "
                                  "landmark 1 is always the same end, and so on along the curve. That gives each jaw "
                                  f"{_JNC} coordinates. But jaws are not {_JNC} independent numbers; neighbouring landmarks move "
                                  "together, and most jaws differ in a few typical ways."},
            {"type": "p", "text": "The question: <i>what are those typical ways, how many are needed, and how do we generate a "
                                  "new, plausible jaw from them?</i>"},
            {"type": "cards", "items": [
                {"title": "Landmarks", "text": "Points that mean the same thing on every shape, so jaws can be compared point by point."},
                {"title": "Alignment", "text": "Differences in where and how large a jaw was drawn are not differences in shape."},
                {"title": "Modes", "text": "Directions of variation learned from the data, ordered by how much they explain."},
            ]},
        ]},
        {"id": "procrustes", "title": "Removing pose: Procrustes alignment", "blocks": [
            {"type": "p", "text": "Before comparing shapes, take out everything that is not shape. Three steps do it: move each "
                                  "jaw so its centroid is at the origin, scale it so its total size is 1, and rotate it to best "
                                  "match a reference."},
            {"type": "figure", "figure": "jaw-align", "title": "Jaws through the three alignment steps",
             "caption": f"{S.JAW_SAMPLE_COUNT} of the jaws, with one highlighted. Step 0 is the raw data; step 3 is what PCA sees.",
             "controls": [{"k": "stage", "label": "Step", "min": 0, "max": 3, "step": 1, "value": 0}]},
            {"type": "eq", "text": "x ← (x − x̄) / ‖x − x̄‖\nR* = argmin_R ‖x R − y‖  =  U Vᵀ   where  xᵀ y = U Σ Vᵀ",
             "where": [["x, y", f"a jaw and the reference, each {_JM.n_landmarks} × 2"],
                       ["x̄", "the centroid of the jaw's landmarks"],
                       ["‖·‖", "Frobenius norm: the square root of the sum of squared coordinates"],
                       ["U Σ Vᵀ", "singular value decomposition of the 2 × 2 matrix xᵀ y; the sign of U's last column is "
                                  "flipped if needed so R is a rotation, not a reflection"]]},
            {"type": "p", "text": "But which reference? Aligning to any one jaw would bias the result towards it. "
                                  "<b>Generalised Procrustes analysis</b> iterates instead: align every jaw to the current mean, "
                                  "average them, normalise, and repeat until the mean stops moving."},
            {"type": "figure", "figure": "jaw-gpa", "title": "Generalised Procrustes converges",
             "caption": "Log scale: the change in the mean shape at each iteration."},
        ]},
        {"id": "pca", "title": "Principal components: the typical ways jaws differ", "blocks": [
            {"type": "p", "text": f"After alignment each jaw is a point in a space of {_JNC} dimensions. PCA finds the directions in "
                                  "which that cloud of points is most spread out. They are the eigenvectors of the covariance "
                                  "matrix, and each eigenvalue is the variance along its direction."},
            {"type": "eq", "text": "C = (1/(n−1)) Σⱼ (xⱼ − x̄)(xⱼ − x̄)ᵀ = Φ Λ Φᵀ\nx ≈ x̄ + Σᵢ bᵢ √λᵢ φᵢ\nshare of mode i = λᵢ / Σ λ",
             "where": [["xⱼ", f"aligned jaw j as a vector of {_JNC} numbers"],
                       ["x̄", "the Procrustes mean jaw"],
                       ["φᵢ, λᵢ", "the i-th mode (unit eigenvector) and its variance"],
                       ["bᵢ", "how far along mode i, in standard deviations"],
                       ["n", f"number of jaws, {_JM.n_shapes}"]]},
            {"type": "figure", "figure": "jaw-variance", "title": "Variation explained by each mode",
             "caption": "Bars: each mode's share. The running total climbs quickly and then flattens."},
            {"type": "quiz", "q": "Why does PCA come after Procrustes alignment, not before?",
             "options": ["PCA needs the landmarks to be in order", "Otherwise the first modes would describe position, size and rotation",
                         "Alignment makes the covariance matrix smaller"], "answer": 1,
             "why": "Unaligned, the biggest differences between jaws are where they were drawn and how large. PCA would spend its "
                    "first modes on those and mix them with real shape differences."},
        ]},
        {"id": "modes", "title": "Walking along a mode", "blocks": [
            {"type": "p", "text": "A mode is easiest to understand by moving along it. Start at the mean jaw and add bᵢ standard "
                                  "deviations of mode i. Because each mode is one fixed vector, every landmark moves in a straight "
                                  "line and all of them move together."},
            {"type": "figure", "figure": "jaw-mode", "title": "The mean jaw pushed both ways along one mode",
             "caption": "Thin lines: aligned training jaws. Dashed: the mean. The two thick lines are the mean plus and minus b standard deviations.",
             "controls": [{"k": "mode", "label": "Mode", "min": 1, "max": 6, "step": 1, "value": 1},
                          {"k": "b", "label": "Distance b", "min": 0.5, "max": 3, "step": 0.5, "value": 2, "unit": "sd"}]},
            {"type": "try", "label": "Open the live model at mode 1, +2 sd", "params": {"b1": 2, "b2": 0, "b3": 0}},
            {"type": "try", "label": "Combine modes 2 and 3", "params": {"b1": 0, "b2": 1.5, "b3": -1.5},
             "note": "Modes are independent, so their effects add."},
        ]},
        {"id": "scores", "title": "Every jaw as a point", "blocks": [
            {"type": "p", "text": "Projecting each aligned jaw on the modes gives its coordinates b. Plotting b₁ against b₂ shows "
                                  "the whole training set at once, and what 'typical' means."},
            {"type": "figure", "figure": "jaw-scores", "title": "The training jaws in mode space",
             "caption": "Each dot is one jaw. Circles mark 1, 2 and 3 standard deviations from the mean jaw."},
            {"type": "callout", "tone": "info", "title": "Plausible and implausible jaws",
             "text": "The model can generate any b, but jaws far outside the cloud were never seen. Keeping each bᵢ within about "
                     "±3 sd, or the total distance √(Σ bᵢ²) small, keeps generated jaws realistic."},
        ]},
        {"id": "rebuild", "title": "Rebuilding a jaw from a few numbers", "blocks": [
            {"type": "p", "text": "The model compresses a jaw to its first few b values. Rebuild it as mean plus modes, and "
                                  "compare with the real jaw: the gap is what the dropped modes would have added."},
            {"type": "figure", "figure": "jaw-rebuild", "title": "One jaw and its reconstruction",
             "caption": "Pick a jaw and the number of modes. Zero modes gives the mean jaw.",
             "controls": [{"k": "jaw", "label": "Jaw", "min": 1, "max": _JM.n_shapes, "step": 1, "value": 1},
                          {"k": "modes", "label": "Modes used", "min": 0, "max": len(_JV), "step": 1, "value": 3}]},
            {"type": "quiz", "q": "A colleague adds 300 more jaws drawn by a different person, slightly larger and shifted. "
                                  "What happens to the modes?",
             "options": ["Nothing, as long as Procrustes runs first", "Mode 1 becomes 'drawn by the second person'",
                         "PCA fails"], "answer": 0,
             "why": "Size and position are removed by alignment. Only a systematic difference in shape, for example a different "
                    "habit of placing landmarks, would show up as a mode."},
        ]},
        {"id": "server", "title": "How the server computes it", "blocks": [
            {"type": "figure", "figure": "jaw-flow", "title": "The model, built once and kept",
             "caption": "Computed when the model is first requested, then reused for every slider move."},
            {"type": "steps", "items": [
                f"Load the x and y landmark arrays: {_JM.n_shapes} jaws × {_JM.n_landmarks} landmarks.",
                "Normalise every jaw (centroid to the origin, unit size) and run generalised Procrustes from the first jaw.",
                f"Stack the aligned jaws minus the mean into a {_JM.n_shapes} × {_JNC} matrix and take its SVD; "
                "the right singular vectors are the modes, and squared singular values / (n − 1) are the variances.",
                f"Keep the first {len(_JV)} modes. A slider move is one sum: mean + Σ bᵢ √λᵢ φᵢ.",
            ]},
            {"type": "callout", "tone": "warn", "title": "What the model assumes",
             "text": "Landmarks must correspond exactly across jaws; a misplaced landmark looks like shape variation. The model is "
                     "linear, so curved patterns of variation are approximated by straight lines through shape space. And the "
                     "±3 sd plausibility rule assumes the cloud of jaws is roughly Gaussian."},
            {"type": "compare", "items": [
                {"title": "Fourier descriptors", "points": ["Fixed basis: sines and cosines", "Works on one outline alone",
                                                            "Low frequencies are smooth, not necessarily typical"]},
                {"title": "Statistical shape model", "points": ["Basis learned from the training jaws", "Needs many shapes with landmarks",
                                                                "First modes are the most common variations"]},
            ]},
        ]},
    ],
    "takeaways": [
        f"Each jaw is {_JM.n_landmarks} corresponding landmarks, {_JNC} numbers.",
        "Procrustes alignment removes position, size and rotation; generalised Procrustes averages to a mean without favouring one jaw.",
        "PCA on the aligned jaws gives modes ordered by variance; a shape is mean + Σ bᵢ √λᵢ φᵢ.",
        f"Three modes explain {_pc(_JC[2], 1)} of the variation in the {_JM.n_shapes} jaws.",
        "Keeping b within about ±3 sd generates jaws that look like the training set.",
    ],
    "glossary": [
        ["Landmark", "A point placed at the same anatomical or geometric position on every shape."],
        ["Procrustes alignment", "Removing translation, scale and rotation by a best-fit similarity transform."],
        ["Generalised Procrustes", "Aligning a whole collection to its own mean, iterated until the mean settles."],
        ["PCA", "Principal component analysis: the eigenvectors of the covariance matrix, ordered by variance."],
        ["Mode", "One principal direction of variation, a pattern in which all landmarks move together."],
        ["Shape space", "The space of aligned shapes, one point per shape."],
        ["Standard deviation (sd)", "The square root of a mode's variance; b is measured in these units."],
    ],
}, order=2)


# ===================================================================== Spectral: figures
SPEC_MESH = "cat4"
PART_COLOURS = ("s1", "s2", "s3", "s4", "accent", "good", "warn", "bad", "neutral")


def _basis(name: str = SPEC_MESH):
    sp_, _ = S.eigenbasis(name, S.spectral_basis(name))
    return sp_


@functools.lru_cache(maxsize=8)
def _edges(name: str) -> np.ndarray:
    (V, F), _ = S.load_mesh(name)
    return np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)


def _nodal_domains(name: str, phi: np.ndarray) -> int:
    """Connected regions of the mesh where the function keeps one sign."""
    E = _edges(name)
    s = phi > 0
    keep = s[E[:, 0]] == s[E[:, 1]]
    n = len(phi)
    G = sp.coo_matrix((np.ones(int(keep.sum())), (E[keep, 0], E[keep, 1])), shape=(n, n))
    return int(connected_components(G, directed=False)[0])


# Side-on views: the cat and the cactus stand along z with their profile in the (y, z) plane; the bunny sits
# along y with its profile in the (x, y) plane.
SIDES = {"cat4": (1, 2), "cactus": (1, 2), "bunny": (0, 1)}
AXIS_NAMES = "xyz"
VIEW_POINTS = 1200  # about this many vertices per scatter, so a 14k-vertex mesh is no heavier than a small one


def _view(name: str):
    (V, _), _ = S.load_mesh(name)
    a, b = SIDES[name]
    return V, _sub(len(V), max(1, math.ceil(len(V) / VIEW_POINTS))), a, b


def _mesh_choice(meshes, value):
    return {"k": "mesh", "label": "Mesh", "options": [[n, S.mesh_label(n)] for n in meshes], "value": value}


@functools.lru_cache(maxsize=1)
def _ring():
    """Laplacian eigenpairs of the bird outline as a closed polygon (1D cotangent weights: 1 / edge length)."""
    from scipy.sparse.linalg import eigsh
    _, b, _ = _bird("bird1")
    b = b[np.linalg.norm(np.roll(b, -1, axis=0) - b, axis=1) > 0]  # the tracing repeats a few pixels; drop the repeats
    n = len(b)
    nxt = np.roll(np.arange(n), -1)
    ln = np.linalg.norm(b[nxt] - b, axis=1)
    w = 1 / ln
    Wm = sp.coo_matrix((np.r_[w, w], (np.r_[np.arange(n), nxt], np.r_[nxt, np.arange(n)])), shape=(n, n)).tocsr()
    L = sp.diags(np.asarray(Wm.sum(1)).ravel()) - Wm
    mass = 0.5 * (ln + np.roll(ln, 1))
    M = sp.diags(mass)
    with S.quiet():
        vals, vecs = eigsh(L.tocsc(), k=17, M=M.tocsc(), sigma=-1e-6, which="LM")
    o = np.argsort(vals)
    s = np.r_[0, np.cumsum(ln)[:-1]]
    return np.maximum(vals[o], 0), vecs[:, o], s, float(ln.sum()), mass


@figure("spectral-ring")
def _s_ring(p):
    i = p.num("i", 1, 1, 12, integer=True)
    vals, vecs, s, L, mass = _ring()
    phi = vecs[:, i]
    phi = phi / np.sqrt(np.sum(mass * phi ** 2) / L)  # unit mean square, so a cosine has amplitude √2
    f = (i + 1) // 2
    # the best-matching wave at the same frequency: project on cos and sin of 2π f s / L
    c, sn = np.cos(2 * np.pi * f * s / L), np.sin(2 * np.pi * f * s / L)
    A = np.stack([c, sn], 1)
    coef, *_ = np.linalg.lstsq(A * np.sqrt(mass)[:, None], phi * np.sqrt(mass), rcond=None)
    wave = A @ coef
    fit = 1 - np.sum(mass * (phi - wave) ** 2) / np.sum(mass * phi ** 2)
    pred = (2 * np.pi * f / L) ** 2
    return {
        "kind": "lines", "x": s, "xlabel": "Distance along the bird outline (px)", "ylabel": f"φ{_subs(i)}",
        "series": [{"name": f"Eigenfunction φ{_subs(i)} of the outline's Laplacian", "color": "s1", "y": phi},
                   {"name": f"Best wave with {f} turn{'s' if f > 1 else ''} round the outline", "color": "s2", "y": wave, "dash": True}],
        "notes": [f"φ{_subs(i)} is a sine wave with {f} full period{'s' if f > 1 else ''} round the outline: the best-fitting wave "
                  f"matches it to within {max(1 - fit, 1e-15):.0e} of its energy.",
                  f"Its eigenvalue λ{_subs(i)} = {vals[i]:.3g}; a perfect wave of that frequency on a loop of length {L:.0f} px has "
                  f"(2π·{f}/{L:.0f})² = {pred:.3g}."],
    }


@figure("spectral-nodal")
def _s_nodal(p):
    name = p.choice("mesh", SPEC_MESH, S.SPECTRAL_MESHES)
    i = p.num("i", 2, 1, 12, integer=True)
    sp_ = _basis(name)
    V, idx, a, b = _view(name)
    phi = sp_.basis.vectors[:, i]
    pts = [{"x": float(V[j, a]), "y": float(V[j, b]), "color": "s2" if phi[j] > 0 else "s1", "r": 2} for j in idx]
    nd = _nodal_domains(name, phi)
    lab = S.mesh_label(name).lower()
    return {
        "kind": "scatter", "equal": True, "points": pts, "xlabel": AXIS_NAMES[a], "ylabel": AXIS_NAMES[b],
        "legend": [{"name": f"φ{_subs(i)} > 0", "color": "s2"}, {"name": f"φ{_subs(i)} < 0", "color": "s1"}],
        "notes": [f"On the {lab}, φ{_subs(i)} has eigenvalue λ{_subs(i)} = {sp_.basis.values[i]:.3g} and is positive on {_pc(np.mean(phi > 0), 0)} of the vertices.",
                  f"Its sign splits the {lab} into {nd} connected region{'s' if nd != 1 else ''}; Courant's theorem allows at most {i + 1} for φ{_subs(i)}."],
    }


@figure("spectral-weyl")
def _s_weyl(p):
    sp_ = _basis()
    lam = np.asarray(sp_.basis.values)
    i = np.arange(len(lam))
    area = float(sp_.basis.mass.sum())
    slope = np.polyfit(i[len(i) // 4:], lam[len(i) // 4:], 1)[0]
    return {
        "kind": "lines", "x": i, "xlabel": "Index i", "ylabel": "Eigenvalue λᵢ", "ymin": 0,
        "series": [{"name": "Cat eigenvalues", "color": "s1", "y": lam},
                   {"name": "Weyl's law 4π i / area", "color": "neutral", "y": 4 * np.pi * i / area, "dash": True}],
        "notes": [f"The cat's surface area is {area:,.0f} square units, so Weyl's law predicts eigenvalues growing by 4π/area = {4 * np.pi / area:.3g} per step.",
                  f"A straight-line fit to the upper three quarters grows by {slope:.3g} per step, {abs(slope / (4 * np.pi / area) - 1):.0%} off."],
    }


REBUILD_MAX = max(S.spectral_basis(n) for n in S.SPECTRAL_MESHES)


@figure("spectral-rebuild")
def _s_rebuild(p):
    name = p.choice("mesh", SPEC_MESH, S.SPECTRAL_MESHES)
    sp_ = _basis(name)
    K = sp_.basis.k
    asked = p.num("k", 30, 1, REBUILD_MAX, integer=True)
    k = min(asked, K)
    V, idx, a, b = _view(name)
    R = S.spectral_reconstruct(sp_, k)
    shift = 1.25 * np.ptp(V[:, a])
    pts = [{"x": float(V[j, a]), "y": float(V[j, b]), "color": "neutral", "r": 1.4} for j in idx]
    pts += [{"x": float(R[j, a] + shift), "y": float(R[j, b]), "color": "s2", "r": 1.6} for j in idx]
    err = float(sp_.errors[k - 1])
    lab = S.mesh_label(name).lower()
    return {
        "kind": "scatter", "equal": True, "points": pts, "xlabel": f"{AXIS_NAMES[a]} (left: original, right: rebuilt)", "ylabel": AXIS_NAMES[b],
        "legend": [{"name": "Original vertices", "color": "neutral"}, {"name": f"Rebuilt from {k} eigenfunctions", "color": "s2"}],
        "notes": [f"k = {k} of the {lab}'s {K} eigenfunctions" + (f" (its basis stops at {K})" if asked > K else "")
                  + f": relative error {_pc(err, 1)}.",
                  f"The rebuilt {lab} is stored as {k} × 3 coefficients instead of {len(V):,} × 3 coordinates ({_pc(k / len(V), 2)} of the numbers)."],
    }


@figure("spectral-error")
def _s_error(p):
    name = p.choice("mesh", SPEC_MESH, S.SPECTRAL_MESHES)
    sp_ = _basis(name)
    e = np.asarray(sp_.errors)
    k = np.arange(1, len(e) + 1)
    c10 = int(np.argmax(e <= 0.1)) + 1 if np.any(e <= 0.1) else None
    en = np.sum(np.asarray(sp_.coeffs) ** 2, axis=1)[1:]  # leave out φ0, which only carries the centroid
    lab = S.mesh_label(name)
    return {
        "kind": "lines", "x": k, "ylog": True, "xlabel": "Eigenfunctions used k", "ylabel": "Relative error",
        "series": [{"name": f"{lab}: error of the k-term rebuild", "color": "s1", "y": e}],
        "hlines": [{"y": 0.1, "label": "10%"}],
        "markers": [{"x": c10, "label": f"k = {c10}"}] if c10 else [],
        "notes": [(f"The {lab.lower()}'s error falls below 10% at k = {c10} of {S.mesh_info(name)[0]['n_vertices']:,} possible, and reaches {_pc(e[-1], 1)} at k = {len(e)}." if c10 else
                   f"Even k = {len(e)} leaves {_pc(e[-1], 1)}."),
                  f"Leaving out φ₀ (the centroid), φ₁ to φ₁₀ carry {_pc(en[:10].sum() / en.sum(), 1)} of the coordinate energy captured by the basis."],
    }


@figure("spectral-segment")
def _s_segment(p):
    name = p.choice("mesh", SPEC_MESH, S.SPECTRAL_MESHES)
    parts = p.num("parts", 5, 2, len(PART_COLOURS), integer=True)
    labels, _ = S.segmentation(name, S.spectral_basis(name), parts)
    labels = np.asarray(labels)
    V, idx, a, b = _view(name)
    pts = [{"x": float(V[j, a]), "y": float(V[j, b]), "color": PART_COLOURS[labels[j]], "r": 1.8} for j in idx]
    sizes = np.bincount(labels, minlength=parts) / len(labels)
    comps = [_nodal_like(name, labels == c) for c in range(parts)]
    return {
        "kind": "scatter", "equal": True, "points": pts, "xlabel": AXIS_NAMES[a], "ylabel": AXIS_NAMES[b],
        "legend": [{"name": f"Part {c + 1} ({_pc(sizes[c], 0)})", "color": PART_COLOURS[c]} for c in range(parts)],
        "notes": [f"{S.mesh_label(name)}, {parts} parts, largest first: " + ", ".join(_pc(v, 0) for v in sizes) + " of the vertices.",
                  (f"Every part is one connected piece of surface" if max(comps) == 1 else
                   f"{sum(c > 1 for c in comps)} of the parts fall into more than one piece")
                  + ", although k-means never sees which vertices are neighbours."],
    }


def _nodal_like(name: str, mask: np.ndarray) -> int:
    """Number of connected pieces of the mesh restricted to the vertices in ``mask``."""
    E = _edges(name)
    keep = mask[E[:, 0]] & mask[E[:, 1]]
    n = len(mask)
    G = sp.coo_matrix((np.ones(int(keep.sum())), (E[keep, 0], E[keep, 1])), shape=(n, n))
    labels = connected_components(G, directed=False)[1]
    return int(len(np.unique(labels[mask])))


@figure("spectral-flow")
def _s_flow(p):
    sp_ = _basis()
    (V, F), _ = S.load_mesh(SPEC_MESH)
    K = sp_.basis.k
    return {
        "kind": "flow", "height": 210,
        "nodes": [
            {"id": "m", "x": 5, "y": 55, "w": 104, "h": 90, "title": S.mesh_label(SPEC_MESH), "value": f"{len(V):,}", "sub": "vertices"},
            {"id": "o", "x": 126, "y": 55, "w": 104, "h": 90, "title": "Operators", "value": "L, M", "sub": "cotan, area"},
            {"id": "e", "x": 247, "y": 55, "w": 104, "h": 90, "title": "Eigenpairs", "value": f"{K}", "sub": "−Lφ = λMφ", "hot": True},
            {"id": "p", "x": 368, "y": 55, "w": 104, "h": 90, "title": "Project", "value": "k = 30", "sub": "first k"},
            {"id": "r", "x": 489, "y": 55, "w": 104, "h": 90, "title": "Rebuilt", "value": _pc(float(sp_.errors[29]), 1), "sub": "error"},
        ],
        "edges": [{"from": "m", "to": "o"}, {"from": "o", "to": "e"}, {"from": "e", "to": "p", "flow": True}, {"from": "p", "to": "r", "flow": True}],
        "texts": [{"x": 299, "y": 30, "text": "solved once per mesh", "cls": "n-sub"},
                  {"x": 300, "y": 185, "text": f"{len(F):,} triangles; changing k or the number of parts reuses the {K} eigenpairs", "cls": "n-sub"}],
        "notes": [f"The {S.mesh_label(SPEC_MESH).lower()} has {len(V):,} vertices; the server keeps its lowest {K} eigenpairs."],
    }


# ===================================================================== Spectral: text
# No eigensolve at import: every number that needs one is computed inside a figure (memoised) on first request.
_SINFO = {n: S.mesh_info(n)[0] for n in S.SPECTRAL_MESHES}

_SPEC_CHOICE = _mesh_choice(S.SPECTRAL_MESHES, SPEC_MESH)

explainer("shape/spectral", {
    "title": "The Laplace–Beltrami spectrum: sines and cosines for a surface",
    "kicker": "Geometry Lab · IT575 L4",
    "summary": "On a closed curve, sines and cosines are the natural vibrations. On a surface the same role is played by the "
               "eigenfunctions of the Laplace–Beltrami operator. They give a smooth-to-detailed basis for any function on the mesh: "
               "a small fraction of them rebuilds a mesh closely, and clustering their values splits it into parts.",
    "sections": [
        {"id": "question", "title": "The question", "blocks": [
            {"type": "p", "text": "Fourier descriptors work because an outline is a loop, and a loop has an obvious set of waves: "
                                  "those that fit a whole number of times round it. A surface such as the "
                                  f"{S.mesh_label(SPEC_MESH).lower()} ({_SINFO[SPEC_MESH]['n_vertices']:,} vertices, {_SINFO[SPEC_MESH]['n_faces']:,} triangles) has no such "
                                  "parameter to run along."},
            {"type": "p", "text": "The question: <i>what are the waves of a surface, and can they do for meshes what Fourier "
                                  "coefficients did for outlines: compress, smooth and describe?</i>"},
            {"type": "cards", "items": [
                {"title": "Smoothing", "text": "Keep the slow waves, drop the fast ones, and fine detail disappears."},
                {"title": "Compression", "text": "A few dozen coefficients per coordinate instead of thousands of vertices."},
                {"title": "Segmentation", "text": "Vertices on the same limb take similar values on the slow waves."},
            ]},
        ]},
        {"id": "waves", "title": "Waves are eigenfunctions of the Laplacian", "blocks": [
            {"type": "p", "text": "The link is the <b>Laplacian</b>, the second derivative. A sine wave is special because taking "
                                  "its second derivative gives the same wave back, scaled: d²/ds² sin(ωs) = −ω² sin(ωs). Waves are "
                                  "the <b>eigenfunctions</b> of the Laplacian, and ω² is the eigenvalue."},
            {"type": "p", "text": "That gives a recipe that works on any shape: build the Laplacian of the shape, then find its "
                                  "eigenfunctions. As a check, do it on the bird outline from the first lab, treated as a loop of "
                                  "points joined by springs."},
            {"type": "figure", "figure": "spectral-ring", "title": "Eigenfunctions of an outline are sine waves",
             "caption": "Pick an eigenfunction. Pairs (1–2, 3–4, …) share a frequency: a sine and a cosine, shifted along the loop.",
             "controls": [{"k": "i", "label": "Eigenfunction i", "min": 1, "max": 12, "step": 1, "value": 1}]},
            {"type": "callout", "tone": "info", "title": "Why this matters",
             "text": "Nothing in the computation mentioned sines. They came out of the Laplacian of the loop. On a surface the "
                     "same computation gives the surface's own waves."},
        ]},
        {"id": "operator", "title": "The Laplacian of a triangle mesh", "blocks": [
            {"type": "p", "text": "On a mesh the Laplacian of a function f at a vertex compares f there with its neighbours, "
                                  "weighted by how the triangles meet. The standard choice is the <b>cotangent Laplacian</b>."},
            {"type": "eq", "text": "(L f)ᵢ = Σⱼ ½ (cot αᵢⱼ + cot βᵢⱼ) (fⱼ − fᵢ)\nM = diag(A₁, …, Aₙ)\n−L φ = λ M φ,   φᵢᵀ M φⱼ = δᵢⱼ",
             "where": [["αᵢⱼ, βᵢⱼ", "the two angles opposite edge ij, one in each triangle that shares it"],
                       ["Aᵢ", "the area belonging to vertex i (mixed Voronoi area)"],
                       ["L", "cotangent Laplacian: sparse, one row per vertex"],
                       ["M", "mass matrix: turns sums over vertices into integrals over the surface"],
                       ["φ, λ", "eigenfunction (one value per vertex) and eigenvalue: how fast φ oscillates"]]},
            {"type": "steps", "title": "Building the operators", "items": [
                "For every triangle, compute the cotangent of each corner angle.",
                "Give each edge half the sum of the cotangents of the two angles facing it; these are the off-diagonal entries of L.",
                "Set each diagonal entry so every row of L sums to zero: a constant function has zero Laplacian.",
                "Share each triangle's area among its corners to get the vertex areas in M.",
                "Solve the generalised eigenproblem for the smallest eigenvalues.",
            ]},
            {"type": "quiz", "q": "What is the eigenfunction with eigenvalue λ₀ = 0?",
             "options": ["A constant: the same value at every vertex", "The height of each vertex", "Zero everywhere"], "answer": 0,
             "why": "Every row of L sums to zero, so L applied to a constant is zero. Projecting the coordinates on it gives the "
                    "surface's centroid, just as Z₀ gave the outline's centre."},
        ]},
        {"id": "look", "title": "What the eigenfunctions look like", "blocks": [
            {"type": "p", "text": "Low eigenfunctions vary slowly over the whole surface; higher ones oscillate faster. The places "
                                  "where an eigenfunction changes sign are its <b>nodal lines</b>, and they cut the surface into "
                                  "regions. The first non-constant one usually splits the shape along its longest direction."},
            {"type": "figure", "figure": "spectral-nodal", "title": "Sign of an eigenfunction (side view)",
             "caption": f"About {VIEW_POINTS:,} vertices seen from the side, marked by the sign of φᵢ.",
             "controls": [_SPEC_CHOICE, {"k": "i", "label": "Eigenfunction i", "min": 1, "max": 12, "step": 1, "value": 2}]},
            {"type": "p", "text": "Eigenvalues measure frequency, and on large surfaces they come closer together. <b>Weyl's law</b> "
                                  "says that for a surface of area A the i-th eigenvalue grows like 4π i / A: the bigger the "
                                  "surface, the more slow waves fit on it."},
            {"type": "figure", "figure": "spectral-weyl", "title": "Eigenvalues against Weyl's law",
             "caption": "The eigenvalues climb roughly along a straight line whose slope depends only on the surface area."},
        ]},
        {"id": "smooth", "title": "Smoothing and compression", "blocks": [
            {"type": "p", "text": "The vertex coordinates x, y and z are three functions on the mesh. Project each onto the first k "
                                  "eigenfunctions and add the pieces back up: the result is the shape with only its slow "
                                  "variations kept."},
            {"type": "eq", "text": "cᵢ = φᵢᵀ M x        (one coefficient per eigenfunction and coordinate)\nx̂ = Σ_{i<k} φᵢ cᵢ\n"
                                   "error = ‖x − x̂‖_M / ‖x − x̄‖_M",
             "where": [["x", "the n × 3 vertex positions"], ["cᵢ", "the three spectral coefficients of eigenfunction i"],
                       ["‖·‖_M", "area-weighted norm, so dense patches of vertices do not count twice"],
                       ["x̄", "the area-weighted centroid; the error is relative to the centred shape"]]},
            {"type": "figure", "figure": "spectral-rebuild", "title": "A mesh rebuilt from k eigenfunctions",
             "caption": "Side views: the original vertices, then the same vertices after projection on the first k eigenfunctions.",
             "controls": [_SPEC_CHOICE, {"k": "k", "label": "Eigenfunctions k", "min": 1, "max": REBUILD_MAX, "step": 1, "value": 30}]},
            {"type": "figure", "figure": "spectral-error", "title": "Error against eigenfunctions used",
             "caption": "Log scale. Because the eigenfunctions are M-orthonormal, every point of this curve comes from one projection.",
             "controls": [_SPEC_CHOICE]},
            {"type": "try", "label": "Open the live model as a blob (k = 5)", "params": {"mesh": SPEC_MESH, "view": "smooth", "k": 5}},
            {"type": "try", "label": "Open it with detail (k = 60)", "params": {"mesh": SPEC_MESH, "view": "smooth", "k": 60}},
        ]},
        {"id": "segment", "title": "Segmentation from the spectrum", "blocks": [
            {"type": "p", "text": "Give each vertex the list of its values on the first eigenfunctions, scaled down by frequency. "
                                  "This <b>global point signature</b> places each vertex in a {0}-dimensional space where vertices on "
                                  "the same limb sit close together, because slow waves barely change along a limb.".format(S.GPS_DIMS)},
            {"type": "eq", "text": f"GPS(v) = ( φ₁(v)/√λ₁, φ₂(v)/√λ₂, …, φ{_subs(S.GPS_DIMS)}(v)/√λ{_subs(S.GPS_DIMS)} )",
             "where": [["φᵢ(v)", "value of eigenfunction i at vertex v"],
                       ["1/√λᵢ", "weight that lets slow eigenfunctions dominate"]]},
            {"type": "steps", "items": [
                f"Compute GPS(v) for every vertex from φ₁ … φ{_subs(S.GPS_DIMS)}.",
                "Run k-means with k-means++ starting points on those vectors.",
                f"Repeat with {S.SEGMENT_RESTARTS} different starts and keep the clustering with the smallest total distance to the centres.",
                "Number the parts by size so colours are stable.",
            ]},
            {"type": "figure", "figure": "spectral-segment", "title": "Parts found by k-means on the spectral embedding",
             "caption": f"About {VIEW_POINTS:,} vertices seen from the side, marked by part. No labels or training data are used.",
             "controls": [_SPEC_CHOICE, {"k": "parts", "label": "Parts", "min": 2, "max": len(PART_COLOURS), "step": 1, "value": 5}]},
            {"type": "quiz", "q": "Why divide each eigenfunction by √λ before clustering?",
             "options": ["To make every coordinate the same size", "So slow, large-scale eigenfunctions count most",
                         "Because λ is the area of the part"], "answer": 1,
             "why": "High eigenfunctions oscillate within a single limb. Dividing by √λ shrinks them, so the clustering follows the "
                    "large-scale structure that separates limbs."},
            {"type": "try", "label": "Segment the cat into 5 parts in the live model", "params": {"mesh": SPEC_MESH, "view": "segment", "parts": 5}},
        ]},
        {"id": "server", "title": "How the server computes it", "blocks": [
            {"type": "figure", "figure": "spectral-flow", "title": "Solve once, reuse for every slider move",
             "caption": "The eigenproblem is the only expensive step and is cached per mesh."},
            {"type": "p", "text": "The server builds L and M with numpy and scipy, then asks for the smallest eigenpairs with "
                                  "<b>shift-invert Lanczos</b> (scipy's eigsh with a shift just below zero). Basis sizes are set per "
                                  "mesh to keep the first solve near a second: " +
                                  _and([f"{_SINFO[n]['label'].lower()} {S.spectral_basis(n)} eigenpairs for {_SINFO[n]['n_vertices']:,} vertices"
                                        for n in S.SPECTRAL_MESHES]) + "."},
            {"type": "callout", "tone": "warn", "title": "Limits",
             "text": "Eigenfunctions are only defined up to sign, and when two eigenvalues are nearly equal any mix of the two is "
                     "also valid, so individual φᵢ are unstable between similar meshes; the server fixes signs by making the largest "
                     "entry positive. Badly shaped triangles make cotangents blow up. And a basis of a hundred-odd functions cannot "
                     "represent detail finer than its fastest wave."},
            {"type": "compare", "items": [
                {"title": "Fourier descriptors", "points": ["A loop: waves known in closed form", "One FFT, N log N",
                                                            "Coefficients comparable between shapes"]},
                {"title": "Laplace–Beltrami spectrum", "points": ["Any surface: waves computed from the mesh", "Sparse eigensolve, the costly step",
                                                                  "Bases differ from mesh to mesh"]},
            ]},
        ]},
    ],
    "takeaways": [
        "Sines and cosines are the eigenfunctions of the Laplacian on a loop; the Laplace–Beltrami operator gives a surface's own waves.",
        "On a mesh, the cotangent Laplacian L and the area matrix M define −Lφ = λMφ; φ₀ is constant and λ grows with frequency.",
        "Projecting the coordinates on the first k eigenfunctions smooths and compresses the shape: the error falls fast over "
        "the first eigenfunctions and slowly after.",
        "Eigenvalues grow roughly like 4π i / area (Weyl's law).",
        "Clustering the scaled eigenfunction values (the global point signature) segments a mesh into parts with no training data.",
    ],
    "glossary": [
        ["Laplace–Beltrami operator", "The Laplacian generalised to curved surfaces: how a function differs from its local average."],
        ["Cotangent Laplacian", "The standard discrete Laplace–Beltrami operator on triangle meshes, weighted by cotangents of opposite angles."],
        ["Mass matrix", "Diagonal matrix of vertex areas used to integrate over the surface."],
        ["Eigenfunction", "A function that the operator only rescales; here, a vibration mode of the surface."],
        ["Nodal line", "Where an eigenfunction is zero; it separates positive and negative regions."],
        ["Weyl's law", "The i-th Laplace eigenvalue of a surface of area A grows like 4π i / A."],
        ["Global point signature", "A vertex's eigenfunction values divided by √λ, used as coordinates for clustering."],
        ["Shift-invert Lanczos", "An iterative method that finds the eigenvalues nearest a chosen shift."],
    ],
}, order=3)


# ===================================================================== Curvature: figures
TYPE_WORDS = {"elliptic": "a dome or bowl (elliptic)", "hyperbolic": "a saddle (hyperbolic)",
              "parabolic": "curved one way only (parabolic)", "planar": "flat to second order (planar)"}
PATCH_CONTROLS = [{"k": "a3", "label": "a₃ (u²)", "min": -2, "max": 2, "step": 0.1, "value": 1},
                  {"k": "a4", "label": "a₄ (uv)", "min": -2, "max": 2, "step": 0.1, "value": 0.6},
                  {"k": "a5", "label": "a₅ (v²)", "min": -2, "max": 2, "step": 0.1, "value": -0.5}]


def _patch(p):
    a = (0.0, 0.0, 0.0, p.num("a3", 1, -2, 2), p.num("a4", 0.6, -2, 2), p.num("a5", -0.5, -2, 2))
    surf = CV.PolynomialSurface.from_lab(a)
    fm = surf.forms(0.0, 0.0)
    I = np.array([[float(fm.E), float(fm.F)], [float(fm.F), float(fm.G)]])
    II = np.array([[float(fm.L), float(fm.M)], [float(fm.M), float(fm.N)]])
    from scipy.linalg import eigh
    k, d = eigh(II, I)  # principal curvatures (ascending) and directions in the (u, v) plane
    c = surf.curvature(0.0, 0.0)
    return a, surf, I, II, k, d, c


def _formula(a) -> str:
    terms = [(a[3], "u²"), (a[4], "uv"), (a[5], "v²")]
    out = ""
    for v, t in terms:
        if abs(v) < 1e-12:
            continue
        sign = (" − " if out else "−") if v < 0 else (" + " if out else "")
        out += sign + (f"{abs(v):g} " if abs(abs(v) - 1) > 1e-12 else "") + t
    return "z = " + (out if out else "0")


@figure("curv-normal")
def _c_normal(p):
    a, surf, I, II, k, d, c = _patch(p)
    th = np.linspace(0, 180, 181)
    u, v = np.cos(np.radians(th)), np.sin(np.radians(th))
    kn = (II[0, 0] * u * u + 2 * II[0, 1] * u * v + II[1, 1] * v * v) / (I[0, 0] * u * u + 2 * I[0, 1] * u * v + I[1, 1] * v * v)
    ang = [float(np.degrees(np.arctan2(d[1, j], d[0, j])) % 180) for j in (1, 0)]
    typ = surf.classify(0.0, 0.0)
    umbilic = abs(k[1] - k[0]) < 1e-9
    first = (f"{_formula(a)}: every direction has the same normal curvature κ = {_m(k[1])}. This is an umbilic point, "
             "where the principal directions are undefined." if umbilic else
             f"{_formula(a)}: the largest and smallest normal curvatures are κ₁ = {_m(k[1])} and κ₂ = {_m(k[0])}, "
             f"at {ang[0]:.0f}° and {ang[1]:.0f}°, exactly 90° apart.")
    return {
        "kind": "lines", "x": th, "xlabel": "Direction in the tangent plane (degrees from the u axis)", "ylabel": "Normal curvature κ",
        "series": [{"name": "κ(θ) at the origin", "color": "s1", "y": kn, "area": False},
                   {"name": "κ = 0", "color": "neutral", "y": np.zeros(len(th)), "dash": True, "thin": True}],
        "markers": [] if umbilic else [{"x": ang[0], "label": f"κ₁ = {_m(k[1])}"}, {"x": ang[1], "label": f"κ₂ = {_m(k[0])}"}],
        "notes": [first, f"K = κ₁κ₂ = {_m(float(c.K))} and H = (κ₁ + κ₂)/2 = {_m(float(c.H))}: the origin is {TYPE_WORDS[typ]}."],
    }


@figure("curv-sections")
def _c_sections(p):
    a, surf, I, II, k, d, c = _patch(p)
    t = np.linspace(-1, 1, 81)
    paths, notes = [], []
    for j, (kk, col, nm) in enumerate(((k[1], "s2", "κ₁"), (k[0], "s1", "κ₂"))):
        dirv = d[:, 1 - j] / np.linalg.norm(d[:, 1 - j])
        z = surf.f(t * dirv[0], t * dirv[1])
        paths.append({"points": np.stack([t, z], 1), "color": col, "width": 2.4, "name": f"Section along the {nm} direction"})
        if abs(kk) > 1e-3:
            r = 1 / abs(kk)
            x = np.linspace(-min(r, 1.0), min(r, 1.0), 81)
            zc = (1 / kk) - np.sign(kk) * np.sqrt(np.maximum(r * r - x * x, 0))
            paths.append({"points": np.stack([x, zc], 1), "color": col, "width": 1.2, "dash": True, "thin": True,
                          "name": f"Osculating circle, radius {r:.2f}"})
    zmax = max(np.max(np.abs(q["points"][:, 1])) for q in paths)
    return {
        "kind": "paths", "equal": True, "axes": True, "xlabel": "distance from the origin along the direction", "ylabel": "height z",
        "paths": paths, "dots": [{"x": 0.0, "y": 0.0, "color": "neutral", "label": "origin"}],
        "notes": [f"Along the κ₁ direction the surface bends {'up' if k[1] > 0 else 'down' if k[1] < 0 else 'neither way'}; "
                  f"along κ₂ it bends {'up' if k[0] > 0 else 'down' if k[0] < 0 else 'neither way'}"
                  + (": opposite ways, so the origin is a saddle." if k[0] * k[1] < -1e-12 else
                     ": the same way, so the origin is a dome or bowl." if k[0] * k[1] > 1e-12 else "."),
                  f"Each dashed circle has radius 1/|κ| and hugs its section at the origin; the sections reach heights up to {zmax:.2f} over this window."],
    }


MONKEY_N = 41


@figure("curv-monkey")
def _c_monkey(p):
    out, _ = S.surface_curvature("monkey", (), MONKEY_N, 1.0)
    K = np.asarray(out["K"])
    lim = float(np.max(np.abs(K)))
    cells = (K.T[::-1] / lim).ravel()  # rows from v = +1 (top) to v = −1, columns u = −1 … +1
    o = out["at_origin"]
    ring = np.hypot(*np.meshgrid(out["u"], out["v"], indexing="ij"))
    rpk = float(ring.ravel()[np.argmin(K)])
    return {
        "kind": "grid", "cols": K.shape[0], "rows": K.shape[1], "cells": cells, "palette": "div", "height": 330,
        "xlabel": "u", "ylabel": "v",
        "xticks": [[0, "−1"], [(K.shape[0] - 1) // 2, "0"], [K.shape[0] - 1, "1"]],
        "yticks": [[0, "1"], [(K.shape[1] - 1) // 2, "0"], [K.shape[1] - 1, "−1"]],
        # the div palette draws negative cells in s1 and positive ones in bad; this surface has no positive cells
        "legend": [{"name": "K < 0, stronger where darker", "color": "s1"}] + ([{"name": "K > 0", "color": "bad"}] if np.any(K > 1e-12) else []),
        "notes": [f"{out['formula'].replace('^3', '³').replace('^2', '²').replace(' - ', ' − ')}: K = {abs(o['K']):.0f} and "
                  f"H = {abs(o['H']):.0f} at the centre, yet the surface is not a plane. "
                  f"K is negative everywhere else, strongest ({_m(-lim)}) at distance {rpk:.2f} from the centre.",
                  f"The fundamental-form result matches the closed form K = −36 r² / (1 + 9 r⁴)² to {out['closed_form']['max_abs_diff']:.0e}."],
    }


def _fan(V: np.ndarray, F: np.ndarray, v: int):
    """The triangles around vertex v, in order: neighbour chain, corner angles at v and spoke lengths."""
    faces = F[np.any(F == v, axis=1)]
    nxt = {}
    for f in faces:
        r = int(np.flatnonzero(f == v)[0])
        a, b = int(f[(r + 1) % 3]), int(f[(r + 2) % 3])
        nxt[a] = b
    start = next(iter(nxt))
    chain = [start]
    while nxt[chain[-1]] != start and len(chain) <= len(nxt):
        chain.append(nxt[chain[-1]])
    chain.append(start)
    vec = V[chain] - V[v]
    ln = np.linalg.norm(vec, axis=1)
    cosang = np.einsum("ij,ij->i", vec[:-1], vec[1:]) / (ln[:-1] * ln[1:])
    return chain, np.arccos(np.clip(cosang, -1, 1)), ln


@functools.lru_cache(maxsize=4)
def _interior_defects(name: str):
    (V, F), _ = S.load_mesh(name)
    c, _ = S.mesh_curvature(name)
    A = SPC.mass_matrix(V, F).diagonal()
    inner = np.flatnonzero(~CV.boundary_vertices(F, len(V)))
    defect = np.asarray(c.gaussian)[inner] * A[inner]
    o = np.argsort(defect)
    return inner[o], defect[o], A


DEFECT_MESH = "cat4"


@figure("curv-defect")
def _c_defect(p):
    name = p.choice("mesh", DEFECT_MESH, S.CURVATURE_MESHES)
    pct = p.num("pct", 98, 0, 100)
    (V, F), _ = S.load_mesh(name)
    order, defects, A = _interior_defects(name)
    j = int(round(pct / 100 * (len(order) - 1)))
    v = int(order[j])
    chain, ang, ln = _fan(V, F, v)
    phi = np.r_[0, np.cumsum(ang)]
    P = np.stack([ln * np.cos(phi), ln * np.sin(phi)], 1)
    paths = [{"points": np.array([[0, 0], P[i], P[i + 1]]), "color": "s1", "closed": True, "width": 1.2, "fill": True, "thin": True}
             for i in range(len(ang))]
    paths.append({"points": np.array([[0, 0], P[0]]), "color": "s2", "width": 2.6, "name": "First edge"})
    paths.append({"points": np.array([[0, 0], P[-1]]), "color": "bad", "width": 2.6, "dash": True, "name": "Same edge after going round"})
    total = float(np.degrees(ang.sum()))
    d = 360 - total
    K = np.radians(d) / A[v]
    return {
        "kind": "paths", "equal": True, "axes": False, "paths": paths, "dots": [{"x": 0.0, "y": 0.0, "color": "neutral", "label": f"vertex {v}"}],
        "notes": [f"The {len(ang)} corner angles at this vertex add up to {total:.1f}°, so its angle defect is 360° − {total:.1f}° = {_m(d, 1)}°"
                  + (": laid flat, the triangles leave a gap, as round a dome." if d > 0.5 else
                     ": laid flat, the triangles overlap, as round a saddle." if d < -0.5 else
                     ": the triangles almost exactly fill the plane, so K is close to zero here (flat, or curved in one direction only)."),
                  f"Dividing by its area {A[v]:.3g} gives K = {_g(K)}. This vertex is at the {pct:.0f}th percentile of the "
                  f"{len(order):,} interior vertices of the {S.mesh_label(name).lower()}."],
    }


@functools.lru_cache(maxsize=8)
def _totals(name: str):
    (V, F), _ = S.load_mesh(name)
    c, _ = S.mesh_curvature(name)
    KA = np.asarray(c.gaussian) * SPC.mass_matrix(V, F).diagonal()
    return float(KA[KA > 0].sum()), float(KA[KA < 0].sum()), float(KA.sum()), c.euler_characteristic


@figure("curv-bonnet")
def _c_bonnet(p):
    names = S.mesh_names()
    t = [_totals(n) for n in names]
    worst = max(abs(net - 2 * np.pi * chi) for _, _, net, chi in t)
    return {
        "kind": "bars", "labels": [S.mesh_label(n) for n in names], "ylabel": "Total Gaussian curvature (multiples of π)",
        "series": [{"name": "Domes: Σ K·A where K > 0", "color": "s2", "values": [pos / np.pi for pos, _, _, _ in t]},
                   {"name": "Saddles: Σ K·A where K < 0", "color": "s1", "values": [neg / np.pi for _, neg, _, _ in t]},
                   {"name": "Net Σ K·A", "color": "s3", "values": [net / np.pi for _, _, net, _ in t]},
                   {"name": "2πχ (Gauss–Bonnet)", "color": "neutral", "values": [2 * chi for _, _, _, chi in t]}],
        "notes": [_cap(_and([f"the {S.mesh_label(n).lower()} has χ = {chi}" for n, (_, _, _, chi) in zip(names, t)]))
                  + f"; every net total equals 2πχ to within {worst:.0e}.",
                  "Dome against saddle: " + "; ".join(f"{S.mesh_label(n).lower()} {pos / np.pi:.1f}π against {-neg / np.pi:.1f}π"
                                                     for n, (pos, neg, _, _) in zip(names, t)) + "."],
    }


@figure("curv-twist")
def _c_twist(p):
    ang = p.num("angle", 180, 0, 360)
    (V, F, W, handles, rest), _ = S.skin_rig("cylinder")
    rows = [("Rest pose", V, 1.0)]
    for m, lab in (("dqs", "DQS twist"), ("lbs", "LBS twist")):
        out, _ = S.skin("cylinder", m, float(ang))
        rows.append((lab, np.asarray(out["vertices"]), out["volume_ratio"]))
    net, absol = [], []
    for _, X, _ in rows:
        c = CV.mesh_curvature(X, F)
        KA = c.K * SPC.mass_matrix(X, F).diagonal()
        net.append(float(np.nansum(KA)) / np.pi)
        absol.append(float(np.nansum(np.abs(KA))) / np.pi)
    return {
        "kind": "bars", "labels": [r[0] for r in rows], "ylabel": "Multiples of π",
        "series": [{"name": "Net Σ K·A", "color": "s3", "values": net},
                   {"name": "Total |K|·A", "color": "s4", "values": absol}],
        "notes": [f"Twisted by {ang:.0f}°, the net total is {net[0]:.4f}π in all three, to within {max(abs(v - net[0]) for v in net):.0e}π: "
                  "the tube is still a closed surface without holes, χ = 2.",
                  f"What can change is the total amount of curvature: |K|·A is {absol[0]:.3f}π at rest, {absol[1]:.3f}π with DQS and "
                  f"{absol[2]:.3f}π with LBS, which keeps {_pc(rows[2][2], 0)} of the volume."],
    }


@functools.lru_cache(maxsize=1)
def _sphere_errors():
    rows = []
    for sub in (2, 3, 4, 5):
        V, F = primitives.icosphere(sub, 1.0)
        c = CV.mesh_curvature(V, F)
        rows.append((len(V), float(np.max(np.abs(c.K - 1))), float(np.max(np.abs(c.H - 1)))))
    return rows


@figure("curv-sphere")
def _c_sphere(p):
    rows = _sphere_errors()
    n = [r[0] for r in rows]
    eK, eH = [r[1] for r in rows], [r[2] for r in rows]
    return {
        "kind": "lines", "x": n, "xlog": True, "ylog": True, "xlabel": "Vertices on the unit sphere", "ylabel": "Largest error",
        "series": [{"name": "Gaussian K (exact: 1)", "color": "s1", "y": eK}, {"name": "Mean H (exact: 1)", "color": "s2", "y": eH}],
        "notes": [f"Each subdivision quadruples the vertices and cuts the K error by {eK[-2] / eK[-1]:.1f}× and the H error by {eH[-2] / eH[-1]:.1f}×: "
                  "both errors shrink like the square of the edge length.",
                  f"At {n[-1]:,} vertices the worst vertex is off by {eK[-1]:.1e} in K and {eH[-1]:.1e} in H."],
    }


MAP_MESH = "cactus"


@figure("curv-map")
def _c_map(p):
    name = p.choice("mesh", MAP_MESH, S.CURVATURE_MESHES)
    V, idx, a, b = _view(name)
    c, _ = S.mesh_curvature(name)
    K = np.asarray(c.gaussian)
    lim = float(np.percentile(np.abs(K), 98))
    thr = 0.1 * lim
    col = np.where(K > thr, "bad", np.where(K < -thr, "s1", "neutral"))
    pts = [{"x": float(V[j, a]), "y": float(V[j, b]), "color": str(col[j]), "r": 1.6} for j in idx]
    return {
        "kind": "scatter", "equal": True, "points": pts, "xlabel": AXIS_NAMES[a], "ylabel": AXIS_NAMES[b],
        "legend": [{"name": "Dome-like, K > 0", "color": "bad"}, {"name": "Saddle-like, K < 0", "color": "s1"},
                   {"name": "Small |K|", "color": "neutral"}],
        "notes": [f"{_pc(np.mean(K > thr), 0)} of the {S.mesh_label(name).lower()}'s vertices are clearly dome-like (K > {thr:.3g}), "
                  f"{_pc(np.mean(K < -thr), 0)} clearly saddle-like (K < −{thr:.3g}); the rest have small |K|.",
                  f"The threshold is a tenth of the 98th percentile of |K|; a few spiky vertices reach |K| = {np.max(np.abs(K)):.3g}."],
    }


# ===================================================================== Curvature: text
_CT_MAP = _totals(MAP_MESH)  # only the cheapest closed mesh at import; the bars figure computes the rest on request
_CI = {n: S.mesh_info(n)[0] for n in S.CURVATURE_MESHES}
_open = [n for n in S.CURVATURE_MESHES if not _CI[n]["closed"]]

explainer("shape/curvature", {
    "title": "Curvature: domes, saddles and why they always balance",
    "kicker": "Geometry Lab · IT575 L3",
    "summary": "At every point a surface bends most in one direction and least in the perpendicular one. The product of those two "
               "curvatures, Gaussian curvature, says whether the point is a dome or a saddle. On a mesh it becomes an angle "
               "deficit at each vertex, and its total over a closed surface is fixed by topology alone: the Gauss–Bonnet theorem.",
    "sections": [
        {"id": "question", "title": "The question", "blocks": [
            {"type": "p", "text": "Look at the cactus: the top of each arm is rounded like a dome, while the place where an arm "
                                  "leaves the trunk is shaped like a saddle. Curvature turns that impression into numbers at "
                                  "every point, and those numbers obey a surprising rule."},
            {"type": "p", "text": "The question: <i>how do we measure bending at a point of a surface, how do we do it on a "
                                  "triangle mesh, and why is the total amount of bending of a closed surface always the same?</i>"},
            {"type": "cards", "items": [
                {"title": "K > 0", "text": "Dome or bowl: the surface bends the same way in every direction."},
                {"title": "K < 0", "text": "Saddle: it bends up in some directions and down in others."},
                {"title": "K = 0", "text": "Cylinder or plane: flat in at least one direction."},
            ]},
        ]},
        {"id": "normal", "title": "Bending in every direction: normal curvature", "blocks": [
            {"type": "p", "text": "Stand at a point and slice the surface with a vertical plane through the normal. The slice is a "
                                  "curve, and its curvature (one over the radius of the best-fitting circle) is the "
                                  "<b>normal curvature</b> in that direction. Turn the plane, and the curvature changes."},
            {"type": "figure", "figure": "curv-normal", "title": "Normal curvature in every direction at the origin",
             "caption": "The patch is z = a₃u² + a₄uv + a₅v². The markers show the principal directions.",
             "controls": PATCH_CONTROLS},
            {"type": "eq", "text": "κ(θ) = κ₁ cos²(θ − θ₁) + κ₂ sin²(θ − θ₁)        (Euler)\nK = κ₁ κ₂        H = (κ₁ + κ₂) / 2",
             "where": [["κ(θ)", "normal curvature in the direction at angle θ"],
                       ["κ₁, κ₂", "principal curvatures: the largest and smallest normal curvatures"],
                       ["θ₁", "the principal direction of κ₁; the other is always 90° away"],
                       ["K", "Gaussian curvature"], ["H", "mean curvature"]]},
            {"type": "figure", "figure": "curv-sections", "title": "The two principal sections and their circles",
             "caption": "The surface cut along each principal direction, with the circle that best fits it at the origin.",
             "controls": PATCH_CONTROLS},
            {"type": "quiz", "q": "What is the Gaussian curvature of a cylinder of radius 2?",
             "options": ["1/4", "1/2", "0"], "answer": 2,
             "why": "Around the cylinder κ₁ = 1/2, but along its axis it is straight, κ₂ = 0. The product is 0: a cylinder can be "
                    "unrolled flat without stretching."},
        ]},
        {"id": "forms", "title": "On a smooth patch: the fundamental forms", "blocks": [
            {"type": "p", "text": "For a surface written as z = f(u, v), derivatives give everything. The <b>first fundamental "
                                  "form</b> measures lengths on the surface; the <b>second</b> measures how fast the normal turns."},
            {"type": "eq", "text": "E = 1 + f_u²,   F = f_u f_v,   G = 1 + f_v²\nL = f_uu / W,   M = f_uv / W,   N = f_vv / W,   W = √(1 + f_u² + f_v²)\n"
                                   "K = (LN − M²) / (EG − F²)\nH = (EN − 2FM + GL) / (2 (EG − F²))",
             "where": [["f_u, f_uv, …", "partial derivatives of the height function"],
                       ["E, F, G", "first fundamental form: the metric"],
                       ["L, M, N", "second fundamental form: the bending"],
                       ["W", "length of the unnormalised normal (−f_u, −f_v, 1)"]]},
            {"type": "figure", "figure": "curv-monkey", "title": "Gaussian curvature over the monkey saddle",
             "caption": f"z = u³ − 3uv² on [−1, 1]², {MONKEY_N} × {MONKEY_N} samples, u across and v up. "
                        "Intensity shows |K| relative to its largest value."},
            {"type": "callout", "tone": "tip", "title": "A saddle with three valleys",
             "text": "The monkey saddle has room for two legs and a tail. At its centre both principal curvatures are zero, so "
                     "second-order tests call it flat, yet it is not a plane: the classification only sees the quadratic part."},
            {"type": "try", "label": "Open the monkey saddle in the live model", "params": {"surface": "monkey"}},
        ]},
        {"id": "mesh", "title": "On a mesh: the angle defect", "blocks": [
            {"type": "p", "text": "A triangle mesh has no derivatives, but it has angles. Around a vertex on a flat sheet the corner "
                                  "angles add up to exactly 360°. On a dome they add up to less, and laid flat the triangles leave a "
                                  "gap; on a saddle they add up to more and overlap. The shortfall is the <b>angle defect</b>."},
            {"type": "eq", "text": "K(v) = (2π − Σⱼ θⱼ) / A(v)        (π instead of 2π on the boundary)\nH(v) = −½ (Δx)(v) · n(v)",
             "where": [["θⱼ", "the corner angles at vertex v of the triangles around it"],
                       ["A(v)", "the vertex's share of the surrounding area (mixed Voronoi area)"],
                       ["Δx", "the cotangent Laplacian applied to the vertex positions"],
                       ["n(v)", "the unit normal at v"]]},
            {"type": "figure", "figure": "curv-defect", "title": "The triangles round one vertex, laid flat",
             "caption": "Pick a mesh and a vertex by its rank: low percentiles are the strongest saddles, high ones the strongest "
                        "domes. The two marked edges are the same mesh edge before and after going round.",
             "controls": [_mesh_choice(S.CURVATURE_MESHES, DEFECT_MESH), {"k": "pct", "label": "Percentile of angle defect", "min": 0, "max": 100, "step": 1, "value": 98, "unit": "%"}]},
            {"type": "steps", "title": "Discrete curvature at every vertex", "items": [
                "For each triangle, compute its three corner angles and add each to its vertex's angle sum.",
                "Subtract each sum from 2π (or π for boundary vertices): the angle defect.",
                "Divide by the vertex's Voronoi area to get K.",
                "Apply the cotangent Laplacian to the coordinates; half its component along the normal is H.",
            ]},
        ]},
        {"id": "bonnet", "title": "Gauss–Bonnet: the total is fixed", "blocks": [
            {"type": "p", "text": "Add up K times area over a whole closed surface and the answer depends only on how many holes it "
                                  "has, through the <b>Euler characteristic</b> χ = V − E + F. Every closed surface without holes, "
                                  "such as the bunny or the cactus, gives exactly 4π. Domes and saddles must balance to make it so."},
            {"type": "eq", "text": "Σᵥ K(v) A(v) = 2π χ        χ = V − E + F",
             "where": [["V, E, F", "number of vertices, edges and faces"],
                       ["χ", "2 for a closed surface without holes, 0 for a torus; a boundary lowers it"]]},
            {"type": "figure", "figure": "curv-bonnet", "title": "Domes, saddles and their total on every mesh",
             "caption": "Positive and negative parts can be large, but their sum is always 2πχ."},
            {"type": "p", "text": "On a mesh this holds exactly, not approximately: the angle sums of all triangles are π each, and "
                                  "counting carefully turns Σ(2π − angles) into 2π(V − E + F). So even a badly deformed mesh keeps "
                                  "the same total, as long as no triangles are torn or added."},
            {"type": "figure", "figure": "curv-twist", "title": "Twist the cylinder: the total does not move",
             "caption": "The closed cylinder from the skinning lab, twisted by the two blending methods.",
             "controls": [{"k": "angle", "label": "Twist", "min": 0, "max": 360, "step": 15, "value": 180, "unit": "°"}]},
            {"type": "quiz", "q": "You push a dent into the side of a closed, hole-free mesh. What happens to Σ K·A?",
             "options": ["It decreases", "It stays 4π", "It depends on how deep the dent is"], "answer": 1,
             "why": "The dent adds a ring of saddle-like vertices around a bowl, and the extra positive and negative curvature cancel. "
                    "Only a change of topology, such as punching a hole, changes the total."},
            {"type": "try", "label": "Check Gauss–Bonnet on the bunny", "params": {"mesh": "bunny", "field": "gaussian"}},
        ]},
        {"id": "where", "title": "Where the saddles are", "blocks": [
            {"type": "figure", "figure": "curv-map", "title": "Sign of K (side view)",
             "caption": f"About {VIEW_POINTS:,} vertices seen from the side, marked by the sign of K. Vertices with |K| below a tenth "
                        "of the 98th percentile count as small.",
             "controls": [_mesh_choice(S.CURVATURE_MESHES, MAP_MESH)]},
            {"type": "p", "text": f"On the {S.mesh_label(MAP_MESH).lower()}, the rounded tips carry positive curvature and the junctions negative. The trunk's sides are "
                                  "nearly cylindrical, so K there is close to zero even though the surface clearly bends: K only "
                                  "notices bending in two directions at once. Mean curvature H does see it."},
            {"type": "try", "label": f"Colour the {S.mesh_label(MAP_MESH).lower()} by mean curvature", "params": {"mesh": MAP_MESH, "field": "mean"}},
        ]},
        {"id": "server", "title": "How the server computes it, and how accurate it is", "blocks": [
            {"type": "p", "text": "The server computes K by angle defect and H by the cotangent Laplacian for every vertex of the "
                                  f"selected mesh. " + _cap(_and([f"the {_CI[n]['label'].lower()} ({_CI[n]['n_vertices']:,} vertices, χ = {_CI[n]['euler_characteristic']})"
                                                               for n in S.CURVATURE_MESHES])) + " are offered. "
                                  "To check accuracy, the same code runs on spheres of radius 1, where K = H = 1 everywhere."},
            {"type": "figure", "figure": "curv-sphere", "title": "Accuracy on refined spheres",
             "caption": "Log–log axes. Both estimates converge as the mesh is refined."},
            {"type": "callout", "tone": "warn", "title": "Limits",
             "text": "Discrete curvature is noisy: a single badly shaped vertex can produce a huge value, which is why the live "
                     "model's colour scale clips at the 2nd and 98th percentiles. The estimates assume reasonably even triangles; "
                     + ("and on an open mesh such as the " + _and([_CI[n]["label"].lower() for n in _open]) +
                        " the boundary vertices measure their defect from π instead of 2π; those defects stand in for the "
                        "turning of the boundary curve, which keeps the discrete total at exactly 2πχ."
                        if _open else "and boundaries need special treatment.")},
            {"type": "compare", "items": [
                {"title": "Gaussian curvature K", "points": ["Product κ₁κ₂", "Intrinsic: unchanged by bending without stretching",
                                                            "Total fixed by topology"]},
                {"title": "Mean curvature H", "points": ["Average (κ₁ + κ₂)/2", "Depends on how the surface sits in space",
                                                        "Zero on soap films (minimal surfaces)"]},
            ]},
        ]},
    ],
    "takeaways": [
        "Normal curvature varies with direction; its extremes κ₁ and κ₂ are the principal curvatures, always 90° apart.",
        "Gaussian curvature K = κ₁κ₂ is positive at domes, negative at saddles and zero on cylinders and planes.",
        "On a mesh, K is the angle defect 2π − Σθ divided by the vertex area; H comes from the cotangent Laplacian.",
        f"Gauss–Bonnet: Σ K·A = 2πχ on a closed mesh, exactly. The {S.mesh_label(MAP_MESH).lower()} (χ = {_CT_MAP[3]}) "
        f"gives {_CT_MAP[2] / np.pi:.4f}π from {_CT_MAP[0] / np.pi:.1f}π of dome and {-_CT_MAP[1] / np.pi:.1f}π of saddle.",
        "Discrete estimates converge on refined meshes but are noisy at badly shaped vertices.",
    ],
    "glossary": [
        ["Normal curvature", "Curvature of the curve cut from the surface by a plane containing the normal."],
        ["Principal curvatures", "The largest and smallest normal curvatures at a point."],
        ["Gaussian curvature K", "κ₁κ₂: sign tells dome (+), saddle (−) or cylinder-like (0)."],
        ["Mean curvature H", "(κ₁ + κ₂)/2, the average bending."],
        ["Fundamental forms", "The first measures lengths on the surface; the second measures how the normal turns."],
        ["Angle defect", "2π minus the sum of corner angles at a vertex."],
        ["Euler characteristic χ", "V − E + F; 2 for a sphere-like closed surface."],
        ["Gauss–Bonnet theorem", "The integral of K over a closed surface equals 2πχ."],
    ],
}, order=4)


# ===================================================================== Skinning: figures
METHOD_NAMES = {"lbs": "LBS", "nlerp": "NLERP", "slerp": "SLERP", "dqs": "DQS"}
METHOD_COLOURS = {"lbs": "bad", "nlerp": "s3", "slerp": "s4", "dqs": "s2"}


def _cyl():
    (V, F, W, handles, rest), _ = S.skin_rig("cylinder")
    return V, F, W, handles, rest


def _top_handle(V, handles) -> int:
    return int(np.argmax(V[handles, 2]))


@figure("skin-weights")
def _k_weights(p):
    V, F, W, handles, rest = _cyl()
    top = _top_handle(V, handles)
    z = np.asarray(rest["ring_z"])
    wt = np.array([W[i, top].mean() for i in rest["rings"]])
    half = float(np.interp(0.5, wt, z))
    return {
        "kind": "lines", "x": z, "xlabel": "Height along the cylinder", "ylabel": "Skinning weight", "ymin": 0, "ymax": 1,
        "series": [{"name": "Weight of the top (twisted) handle", "color": "s2", "y": wt, "area": True},
                   {"name": "Weight of the bottom (fixed) handle", "color": "s1", "y": 1 - wt}],
        "markers": [{"x": half, "label": "w = ½"}],
        "notes": [f"The weights pass ½ at height {half:.2f} of {z.max() - z.min():.0f}; each ring's two weights add up to 1.",
                  f"Near the caps the far handle still has weight {wt[0]:.3f} at the bottom and {1 - wt[-1]:.3f} at the top: "
                  "bounded biharmonic weights spread smoothly over the whole tube."],
    }


@figure("skin-quat")
def _k_quat(p):
    ang = p.num("angle", 180, 0, 360)
    w = p.num("w", 0.5, 0, 1)
    th = math.radians(ang)
    q0 = skinning.quat_from_axis_angle([0, 0, 1], 0.0)
    q1 = skinning.quat_from_axis_angle([0, 0, 1], th)
    if q0 @ q1 < 0:  # the shorter arc, as the blending code does
        q1 = -q1
    lin = (1 - w) * q0 + w * q1
    nl = lin / np.linalg.norm(lin)
    sl = skinning.slerp(q0, q1, w)
    xy = lambda q: [float(q[0]), float(q[3])]  # rotations about z only use the w and z components
    t = np.linspace(0, 2 * np.pi, 121)
    a0, a1 = math.atan2(q0[3], q0[0]), math.atan2(q1[3], q1[0])
    arc = np.linspace(a0, a1, 40)
    # the matrix LBS blends, applied to a unit point at right angles to the axis
    R = np.stack([skinning.rodrigues(0.0, [0, 0, 1]), skinning.rodrigues(th, [0, 0, 1])])
    x = skinning.lbs(np.array([[1.0, 0, 0]]), np.array([[1 - w, w]]), R)[0]
    nl_ang = round(math.degrees(2 * math.atan2(nl[3], nl[0])), 1) + 0.0  # + 0.0 turns −0.0 into 0.0
    sl_ang = round(math.degrees(2 * math.atan2(sl[3], sl[0])), 1) + 0.0
    return {
        "kind": "paths", "equal": True, "axes": True, "xlabel": "w component (cos θ/2)", "ylabel": "z component (sin θ/2)",
        "paths": [{"points": np.stack([np.cos(t), np.sin(t)], 1), "color": "neutral", "width": 1, "thin": True, "name": "Unit quaternions"},
                  {"points": np.stack([np.cos(arc), np.sin(arc)], 1), "color": "s1", "width": 2.6, "name": "SLERP path (the arc)"},
                  {"points": np.array([xy(q0), xy(q1)]), "color": "warn", "width": 2, "dash": True, "name": "Linear blend (the chord)"},
                  {"points": np.array([[0, 0], xy(nl)]), "color": "s2", "width": 1.4, "thin": True, "name": "NLERP: push the chord point out"}],
        "dots": [{"x": xy(q0)[0], "y": xy(q0)[1], "color": "neutral", "label": "q₀ (bottom)"},
                 {"x": xy(q1)[0], "y": xy(q1)[1], "color": "neutral", "label": "q₁ (top)"},
                 {"x": xy(lin)[0], "y": xy(lin)[1], "color": "warn", "label": f"blend, length {np.linalg.norm(lin):.2f}"},
                 {"x": xy(nl)[0], "y": xy(nl)[1], "color": "s2", "label": "NLERP"},
                 {"x": xy(sl)[0], "y": xy(sl)[1], "color": "s1", "label": "SLERP"}],
        "notes": [f"With weight {w:.2f} on a {ang:.0f}° twist, LBS's averaged matrix moves a point at radius 1 to radius {np.linalg.norm(x[:2]):.3f}; "
                  f"the quaternion chord point has length {np.linalg.norm(lin):.3f} and is rescaled to 1.",
                  f"NLERP turns the vertex by {nl_ang:.1f}° and SLERP by {sl_ang:.1f}°: both are true rotations, "
                  + ("and they agree at this weight." if abs(nl_ang - sl_ang) < 0.05 else
                     f"but NLERP is {abs(nl_ang - sl_ang):.1f}° off the constant-speed path.")],
    }


@figure("skin-ring")
def _k_ring(p):
    ang = p.num("angle", 180, 0, 360)
    h = p.num("height", 0.5, 0, 1)
    V, F, W, handles, rest = _cyl()
    z = np.asarray(rest["ring_z"])
    j = int(np.argmin(np.abs(z - (z.min() + h * (z.max() - z.min())))))
    ring = rest["rings"][j]
    order = ring[np.argsort(np.arctan2(V[ring, 1], V[ring, 0]))]
    r0 = float(rest["ring_radius"][j])
    paths = [{"points": V[order, :2], "color": "neutral", "width": 1.4, "dash": True, "closed": True, "name": "Rest"}]
    dots, rad, turn = [], {}, {}
    for m in ("lbs", "dqs"):
        D = np.asarray(S.skin("cylinder", m, float(ang))[0]["vertices"])
        paths.append({"points": D[order, :2], "color": METHOD_COLOURS[m], "width": 2.4, "closed": True, "name": METHOD_NAMES[m]})
        paths.append({"points": np.array([[0, 0], D[order[0], :2]]), "color": METHOD_COLOURS[m], "width": 1.2, "thin": True})
        rad[m] = float(np.linalg.norm(D[order, :2], axis=1).mean() / r0)
        turn[m] = math.degrees(math.atan2(D[order[0], 1], D[order[0], 0]) - math.atan2(V[order[0], 1], V[order[0], 0])) % 360
    wt = float(W[ring, _top_handle(V, handles)].mean())
    dots.append({"x": float(V[order[0], 0]), "y": float(V[order[0], 1]), "color": "neutral", "label": "a vertex at rest"})
    return {
        "kind": "paths", "equal": True, "axes": False, "paths": paths, "dots": dots,
        "notes": [f"The ring at height {z[j]:.1f} has weight {wt:.2f} on the twisted handle. LBS keeps {_pc(rad['lbs'], 1)} of its radius; "
                  f"DQS keeps {_pc(rad['dqs'], 1)}.",
                  ("A 360° turn is the same rotation as no turn, so every method leaves the ring where it was." if ang % 360 == 0 and ang > 0 else
                   f"DQS turns the ring by {turn['dqs']:.0f}° (the weight times the twist would be {wt * ang:.0f}°); "
                   + (f"LBS turns it by {turn['lbs']:.0f}°" + (" while shrinking it." if rad["lbs"] < 0.995 else ".") if rad["lbs"] > 0.05 else
                      "LBS has crushed it almost to the axis."))],
    }


@figure("skin-profile")
def _k_profile(p):
    ang = p.num("angle", 180, 0, 360)
    V, F, W, handles, rest = _cyl()
    z = np.asarray(rest["ring_z"])
    series, low = [], {}
    for m in S.SKIN_METHODS:
        prof = S.skin("cylinder", m, float(ang))[0]["cross_section_profile"]
        series.append({"name": METHOD_NAMES[m], "color": METHOD_COLOURS[m], "y": prof["ratio"], "dash": m in ("nlerp", "slerp")})
        low[m] = float(np.min(prof["ratio"]))
    return {
        "kind": "lines", "x": z, "xlabel": "Height along the cylinder", "ylabel": "Radius kept", "ymin": 0, "ymax": 1.1,
        "series": series,
        "notes": [f"At {ang:.0f}° the thinnest ring keeps " + ", ".join(f"{_pc(v, 1)} ({METHOD_NAMES[m]})" for m, v in low.items()) + ".",
                  "The quaternion methods keep every ring round; LBS pinches the tube where the two weights are most mixed."
                  if low["lbs"] < 0.95 else "At this angle even LBS keeps the tube nearly round."],
    }


@functools.lru_cache(maxsize=1)
def _scale_curve():
    V, F, W, handles, rest = _cyl()
    ring = rest["rings"][rest["mid"]]
    angs = np.arange(-360, 361, 15)
    out = {m: [] for m in ("lbs", "dqs")}
    for a in angs:
        R, t = S.skin_pose("cylinder", V, handles, float(a))
        for m in out:
            D = skinning.deform(V[ring], W[ring], R, t, m)
            out[m].append(float(np.linalg.norm(D[:, :2], axis=1).mean() / rest["ring_radius"][rest["mid"]]))
    w = float(W[ring, _top_handle(V, handles)].mean())
    return angs, out, w


@figure("skin-scale")
def _k_scale(p):
    angs, out, w = _scale_curve()
    th = np.radians(angs)
    theory = np.sqrt((1 - w) ** 2 + w ** 2 + 2 * w * (1 - w) * np.cos(th))
    i90 = int(np.argmin(np.abs(angs - 90)))
    return {
        "kind": "lines", "x": angs, "xlabel": "Twist angle (degrees)", "ylabel": "Middle ring radius kept", "ymin": 0, "ymax": 1.1,
        "series": [{"name": "LBS, measured on the mesh", "color": "bad", "y": out["lbs"]},
                   {"name": f"Formula |(1−w) + w e^(iθ)|, w = {w:.3f}", "color": "neutral", "y": theory, "dash": True},
                   {"name": "DQS, measured", "color": "s2", "y": out["dqs"]}],
        "notes": [f"The middle ring has weight w = {w:.3f}, so LBS scales it by |(1−w) + w e^(iθ)|, close to |cos(θ/2)|: "
                  f"{_pc(out['lbs'][i90], 0)} at 90°, {_pc(min(out['lbs']), 1)} at its worst.",
                  f"DQS keeps {_pc(np.mean(out['dqs']), 1)} at every angle, to within {max(abs(v - 1) for v in out['dqs']):.0e}. At ±360° LBS is back to "
                  f"{_pc(out['lbs'][-1], 0)}: a full turn is the same matrix as no turn."],
    }


CACTUS_AXIS = 0  # the cactus rig bends about x, so a slice at constant x shows the whole bend


@figure("skin-cactus")
def _k_cactus(p):
    bend = p.num("bend", 18, -60, 60)
    (V, F, W, handles, rest), _ = S.skin_rig("cactus")
    cx = float(np.median(V[:, CACTUS_AXIS]))
    paths = [{"points": l[:, 1:], "color": "neutral", "width": 1.4, "dash": True, "closed": True, **({"name": "Rest"} if i == 0 else {})}
             for i, l in enumerate(_slice(V, F, CACTUS_AXIS, cx))]
    D = {}
    for m in ("lbs", "dqs"):
        D[m] = np.asarray(S.skin("cactus", m, float(bend))[0]["vertices"])
        paths += [{"points": l[:, 1:], "color": METHOD_COLOURS[m], "width": 2.2, "closed": True, **({"name": METHOD_NAMES[m]} if i == 0 else {})}
                  for i, l in enumerate(_slice(D[m], F, CACTUS_AXIS, cx))]
    ratios = S.skin_pose_ratios("cactus")
    moving = [j for j, r in enumerate(S.CACTUS_PROPORTIONS) if r != 0]
    dots = [{"x": float(V[handles[j], 1]), "y": float(V[handles[j], 2]), "color": "accent" if j in moving else "neutral",
             "label": f"turns {_m(S.CACTUS_PROPORTIONS[j] * bend, 0)}°" if j in moving else "fixed"} for j in range(len(handles))]
    gap = np.linalg.norm(D["lbs"] - D["dqs"], axis=1)
    vol = {m: S.skin("cactus", m, float(bend))[0]["volume_ratio"] for m in ("lbs", "dqs")}
    return {
        "kind": "paths", "equal": True, "axes": False, "paths": paths, "dots": dots,
        "notes": [f"A cut through the middle of the cactus. With a bend of {_m(bend, 0)}° (the second moving handle turns {ratios[1]:.2f} as much), "
                  f"the volume ends at {_pc(vol['lbs'], 1)} of the rest volume with LBS and {_pc(vol['dqs'], 1)} with DQS.",
                  f"The two methods put a vertex up to {gap.max() / np.ptp(V[:, 2]):.1%} of the cactus height apart; "
                  f"half of the vertices differ by less than {np.median(gap) / np.ptp(V[:, 2]):.2%}."],
    }


@figure("skin-volume")
def _k_volume(p):
    ang = p.num("angle", 180, 0, 360)
    bend = p.num("bend", 18, -60, 60)
    cyl = [S.skin("cylinder", m, float(ang))[0]["volume_ratio"] for m in S.SKIN_METHODS]
    cac = [S.skin("cactus", m, float(bend))[0]["volume_ratio"] for m in S.SKIN_METHODS]
    return {
        "kind": "bars", "labels": [METHOD_NAMES[m] for m in S.SKIN_METHODS], "ylabel": "Volume kept (%)",
        "series": [{"name": f"Cylinder, {ang:.0f}° twist", "color": "s1", "values": 100 * np.array(cyl)},
                   {"name": f"Cactus, {_m(bend, 0)}° bend", "color": "s2", "values": 100 * np.array(cac)}],
        "notes": ["Cylinder: " + ", ".join(f"{METHOD_NAMES[m]} {_pc(v, 1)}" for m, v in zip(S.SKIN_METHODS, cyl)) + ".",
                  "Cactus: " + ", ".join(f"{METHOD_NAMES[m]} {_pc(v, 1)}" for m, v in zip(S.SKIN_METHODS, cac))
                  + f". Even the quaternion methods change the cactus's volume by up to {max(abs(v - 1) for m, v in zip(S.SKIN_METHODS, cac) if m != 'lbs'):.1%}: "
                  "they prevent collapse, not every change of volume."],
    }


@figure("skin-flow")
def _k_flow(p):
    V, F, W, handles, rest = _cyl()
    out = {m: S.skin("cylinder", m, 180.0)[0] for m in ("lbs", "dqs")}
    return {
        "kind": "flow", "height": 230,
        "nodes": [
            {"id": "r", "x": 5, "y": 70, "w": 104, "h": 86, "title": "Rest mesh", "value": f"{len(V):,}", "sub": "vertices"},
            {"id": "h", "x": 126, "y": 70, "w": 104, "h": 86, "title": "Handles", "value": f"{len(handles)}", "sub": "R, t each"},
            {"id": "w", "x": 247, "y": 70, "w": 104, "h": 86, "title": "Weights", "value": f"{len(V):,} × {len(handles)}", "sub": "rows sum to 1"},
            {"id": "l", "x": 380, "y": 12, "w": 96, "h": 76, "title": "LBS", "value": _pc(out["lbs"]["cross_section_ratio"], 0), "sub": "thinnest ring"},
            {"id": "d", "x": 380, "y": 138, "w": 96, "h": 76, "title": "DQS", "value": _pc(out["dqs"]["cross_section_ratio"], 0), "sub": "thinnest ring", "hot": True},
            {"id": "o", "x": 500, "y": 70, "w": 94, "h": 86, "title": "Deformed", "value": "per vertex", "sub": "x′ = T(x) x"},
        ],
        "edges": [{"from": "r", "to": "h"}, {"from": "h", "to": "w"}, {"from": "w", "to": "l", "label": "blend matrices"},
                  {"from": "w", "to": "d", "label": "blend dual quats", "below": True, "flow": True},
                  {"from": "l", "to": "o"}, {"from": "d", "to": "o", "flow": True}],
        "notes": [f"The cylinder has {len(V):,} vertices and {len(handles)} handles; at a 180° twist the thinnest ring keeps "
                  f"{_pc(out['lbs']['cross_section_ratio'], 1)} of its radius with LBS and {_pc(out['dqs']['cross_section_ratio'], 1)} with DQS."],
    }


# ===================================================================== Skinning: text
_KV, _KF, _KW, _KH, _KR = _cyl()
_K180 = {m: S.skin("cylinder", m, 180.0)[0] for m in S.SKIN_METHODS}
_CAC = S.skin_rig("cactus")[0]

explainer("shape/skinning", {
    "title": "Skinning: why dual quaternions keep a twisted arm round",
    "kicker": "Geometry Lab · IT575 L5",
    "summary": "Skinning moves every vertex of a character by a weighted mix of a few handle transforms. The standard method, "
               "linear blend skinning, averages rotation matrices, and an average of rotations is not a rotation: twist a tube by "
               f"180° and its middle collapses to {_pc(_K180['lbs']['cross_section_ratio'], 0)} of its radius. Blending quaternions, "
               "and dual quaternions for rotation and translation together, keeps every vertex rigid.",
    "sections": [
        {"id": "question", "title": "The question", "blocks": [
            {"type": "p", "text": f"The test rig is a closed cylinder of {len(_KV):,} vertices with {len(_KH)} handles, one at the "
                                  "centre of each cap. Hold the bottom cap still and turn the top cap about the tube's axis, as you "
                                  "would twist a forearm. Every vertex in between should turn part of the way and stay at the "
                                  "same distance from the axis."},
            {"type": "p", "text": "The question: <i>how should a vertex combine the motions of several handles, and why does the "
                                  "obvious answer fail so badly?</i>"},
            {"type": "cards", "items": [
                {"title": "Handles", "text": "A few control points, each with a rigid motion: a rotation R and a translation t."},
                {"title": "Weights", "text": "For every vertex, how much each handle pulls it. Each vertex's weights add up to 1."},
                {"title": "Blending", "text": "Combining the handle motions into one motion per vertex: the step that can go wrong."},
            ]},
        ]},
        {"id": "weights", "title": "Handles and weights", "blocks": [
            {"type": "p", "text": "The weights are precomputed and stored with each rig as <b>bounded biharmonic weights</b>: "
                                  "smooth functions over the mesh, 1 at their own handle, 0 at the others, never negative. Along "
                                  "the cylinder they fade smoothly from one cap to the other."},
            {"type": "figure", "figure": "skin-weights", "title": "Weights along the cylinder",
             "caption": "The average weight of each ring of vertices, from the bottom cap to the top."},
            {"type": "eq", "text": "Tⱼ(x) = Rⱼ (x − pⱼ) + pⱼ = Rⱼ x + tⱼ,     tⱼ = pⱼ − Rⱼ pⱼ\nΣⱼ wᵢⱼ = 1 for every vertex i",
             "where": [["Rⱼ", "rotation of handle j (here about the tube's axis)"], ["pⱼ", "position of handle j: it turns about itself"],
                       ["tⱼ", "the matching translation"], ["wᵢⱼ", "weight of handle j at vertex i"]]},
        ]},
        {"id": "lbs", "title": "Linear blend skinning and the candy wrapper", "blocks": [
            {"type": "p", "text": "The simplest blend applies every handle's motion to the vertex and averages the results with the "
                                  "weights. That is the same as averaging the matrices first."},
            {"type": "eq", "text": "x′ = Σⱼ wᵢⱼ (Rⱼ x + tⱼ) = (Σⱼ wᵢⱼ Rⱼ) x + Σⱼ wᵢⱼ tⱼ\n½ R(0) + ½ R(θ) = cos(θ/2) · R(θ/2)",
             "where": [["Σⱼ wᵢⱼ Rⱼ", "the blended matrix; generally not a rotation"],
                       ["R(θ)", "rotation by θ about the axis"],
                       ["cos(θ/2)", "the factor by which the half-and-half blend shrinks everything perpendicular to the axis"]]},
            {"type": "p", "text": "The second line is the whole problem. Halfway along the tube, where the weights are ½ and ½, LBS "
                                  "does turn the ring by θ/2, but it also scales it by cos(θ/2). At 180° that factor is zero: the "
                                  "ring collapses onto the axis, the 'candy wrapper'."},
            {"type": "figure", "figure": "skin-ring", "title": "One ring of the tube, seen along the axis",
             "caption": "Choose the twist and which ring to look at. The thin spokes follow one vertex of the ring.",
             "controls": [{"k": "angle", "label": "Twist", "min": 0, "max": 360, "step": 15, "value": 180, "unit": "°"},
                          {"k": "height", "label": "Ring position (0 bottom, 1 top)", "min": 0, "max": 1, "step": 0.05, "value": 0.5}]},
            {"type": "figure", "figure": "skin-profile", "title": "Radius kept along the whole tube",
             "caption": "All four blending methods. NLERP and SLERP are dashed; they lie on top of DQS here.",
             "controls": [{"k": "angle", "label": "Twist", "min": 0, "max": 360, "step": 15, "value": 180, "unit": "°"}]},
            {"type": "figure", "figure": "skin-scale", "title": "How much the middle ring keeps, at every angle",
             "caption": "Measured on the mesh, against the formula for the blended matrix."},
            {"type": "quiz", "q": "With LBS, which twist angle leaves the middle ring at about 71% of its radius?",
             "options": ["45°", "90°", "135°"], "answer": 1,
             "why": "The half-and-half blend scales by |cos(θ/2)|, and cos 45° ≈ 0.71. The weights at the middle ring are not exactly "
                    "½, so the measured value is close to, not exactly, that."},
            {"type": "try", "label": "See the candy wrapper in 3D (LBS, 180°)", "params": {"mesh": "cylinder", "method": "lbs", "angle": 180}},
        ]},
        {"id": "quaternions", "title": "Blending rotations as quaternions", "blocks": [
            {"type": "p", "text": "A rotation by θ about a unit axis n can be written as a unit <b>quaternion</b>, four numbers on "
                                  "the unit sphere in 4D. Blending quaternions and then rescaling to length 1 always gives a "
                                  "rotation, so the vertex can only turn, never shrink."},
            {"type": "eq", "text": "q = ( cos(θ/2), sin(θ/2) n )\nNLERP:  q = Σⱼ wⱼ qⱼ / ‖Σⱼ wⱼ qⱼ‖\nSLERP:  q(w) = sin((1−w)Ω)/sin Ω · q₀ + sin(wΩ)/sin Ω · q₁,   cos Ω = q₀·q₁",
             "where": [["q", "unit quaternion (scalar part first)"], ["θ, n", "rotation angle and axis"],
                       ["NLERP", "normalised linear blend: average, then rescale to length 1"],
                       ["SLERP", "spherical linear interpolation: move along the great circle at constant speed"],
                       ["Ω", "the angle between the two quaternions"]]},
            {"type": "figure", "figure": "skin-quat", "title": "Two rotations about the same axis as points on a circle",
             "caption": "For rotations about one axis only two quaternion components change, so the unit quaternions form a circle. "
                        "The chord point is the plain average; NLERP pushes it back out to the circle.",
             "controls": [{"k": "angle", "label": "Twist", "min": 0, "max": 360, "step": 5, "value": 180, "unit": "°"},
                          {"k": "w", "label": "Weight of the top handle", "min": 0, "max": 1, "step": 0.05, "value": 0.5}]},
            {"type": "callout", "tone": "info", "title": "Why θ/2?",
             "text": "q and −q are the same rotation, so the quaternions cover rotations twice. Blending code first flips quaternions "
                     "into the same half of the sphere, otherwise it would interpolate the long way round."},
            {"type": "quiz", "q": "You twist the top cap by exactly 360°. What does each method produce?",
             "options": ["LBS collapses, DQS keeps a full twist", "Every method returns the rest shape", "Every method collapses"],
             "answer": 1,
             "why": "A 360° rotation has the same matrix as no rotation, and the blending sees only the handle transforms. To get a "
                    "full twist the rig would need intermediate handles."},
        ]},
        {"id": "dqs", "title": "Dual quaternions: rotation and translation together", "blocks": [
            {"type": "p", "text": "NLERP and SLERP fix the rotation but still average the translations separately. When handles "
                                  "rotate about different centres that is not a rigid motion. A <b>dual quaternion</b> stores rotation "
                                  "and translation in one object, q + εd, so both are blended together."},
            {"type": "eq", "text": "q̂ⱼ = qⱼ + ε dⱼ,   dⱼ = ½ (0, tⱼ) qⱼ,   ε² = 0\nq̂ = Σⱼ wⱼ q̂ⱼ / ‖Σⱼ wⱼ qⱼ‖\nx′ = R(q) x + 2 vec(d q*)",
             "where": [["q̂ⱼ", "dual quaternion of handle j"], ["ε", "the dual unit: a number whose square is zero"],
                       ["dⱼ", "dual part, encoding the translation"], ["q*", "quaternion conjugate"],
                       ["vec(·)", "the vector part of a quaternion"]]},
            {"type": "steps", "title": "Dual-quaternion skinning, per vertex", "items": [
                "Convert each handle's (R, t) to a unit dual quaternion; flip signs so all real parts share a hemisphere.",
                "Blend the handles' dual quaternions with the vertex's weights.",
                "Divide both parts by the length of the real part.",
                "Read off the rotation from the real part and the translation from 2 d q*, and apply them to the vertex.",
            ]},
            {"type": "try", "label": "Compare: DQS at 180°", "params": {"mesh": "cylinder", "method": "dqs", "angle": 180}},
        ]},
        {"id": "cactus", "title": "Beyond twisting: the cactus", "blocks": [
            {"type": "p", "text": f"The cactus rig has {len(_CAC[3])} handles and bends rather than twists: two handles turn about "
                                  "horizontal axes through themselves, in a fixed ratio, while the others stay still. The rotation "
                                  "axes are perpendicular to the cut below, so the cut shows the whole bend."},
            {"type": "figure", "figure": "skin-cactus", "title": "A cut through the cactus, rest and bent",
             "caption": "Dashed: rest pose. The dots mark the handles (projected onto the cut).",
             "controls": [{"k": "bend", "label": "Bend", "min": -60, "max": 60, "step": 2, "value": 18, "unit": "°"}]},
            {"type": "figure", "figure": "skin-volume", "title": "Volume kept by each method",
             "caption": "Signed volume of the deformed mesh divided by the rest volume.",
             "controls": [{"k": "angle", "label": "Cylinder twist", "min": 0, "max": 360, "step": 15, "value": 180, "unit": "°"},
                          {"k": "bend", "label": "Cactus bend", "min": -60, "max": 60, "step": 2, "value": 18, "unit": "°"}]},
            {"type": "compare", "items": [
                {"title": "LBS", "points": ["One matrix average per vertex, the cheapest", "Shrinks joints under twist and bend",
                                            "Still the default in many engines"]},
                {"title": "NLERP / SLERP", "points": ["Rotations stay rigid", "Translations averaged separately",
                                                      "SLERP is exact only for two handles"]},
                {"title": "DQS", "points": ["Rotation and translation blended together", "No collapse",
                                            "Can bulge slightly at bent joints"]},
            ]},
            {"type": "try", "label": "Open the cactus bend in the live model", "params": {"mesh": "cactus", "method": "dqs", "bend": 18}},
        ]},
        {"id": "server", "title": "How the server computes it", "blocks": [
            {"type": "figure", "figure": "skin-flow", "title": "From handles to a deformed mesh",
             "caption": "Both blending routes, with the thinnest ring of the cylinder at a 180° twist."},
            {"type": "steps", "items": [
                "Load the rig's mesh and its bounded biharmonic weights (a MATLAB file) and check one weight row per vertex.",
                "Build each handle's rotation with Rodrigues' formula about an axis through the handle, and the matching translation.",
                "Deform all vertices at once with the chosen method (numpy, vectorised over vertices).",
                "Measure the signed volume (divergence theorem) and, for the cylinder, the mean radius of every ring of vertices.",
            ]},
            {"type": "callout", "tone": "warn", "title": "Limits",
             "text": "Skinning only knows the handle transforms, so twists of 360° or more cannot be told apart from smaller ones. "
                     "Results depend on the weights as much as on the blend: poor weights crease or tear any method. And no "
                     "blend preserves volume exactly; DQS avoids collapse but can bulge at joints."},
        ]},
    ],
    "takeaways": [
        "Skinning moves each vertex by a weighted blend of a few handle transforms; each vertex's weights add up to 1.",
        "LBS averages matrices, and ½R(0) + ½R(θ) = cos(θ/2)·R(θ/2): a scaled rotation, which collapses at 180°.",
        f"On the cylinder at 180°, LBS keeps {_pc(_K180['lbs']['cross_section_ratio'], 1)} of the thinnest ring's radius and "
        f"{_pc(_K180['lbs']['volume_ratio'], 0)} of the volume; DQS keeps {_pc(_K180['dqs']['cross_section_ratio'], 1)} and "
        f"{_pc(_K180['dqs']['volume_ratio'], 1)}.",
        "Blending unit quaternions and renormalising always gives a rotation; SLERP follows the arc at constant speed.",
        "Dual quaternions blend rotation and translation together, which matters when handles turn about different centres.",
    ],
    "glossary": [
        ["Handle", "A control point whose rigid motion drives part of the mesh."],
        ["Skinning weight", "How strongly a handle moves a given vertex."],
        ["Bounded biharmonic weights", "Smooth, non-negative weights computed by minimising a biharmonic energy."],
        ["LBS", "Linear blend skinning: average the handle matrices."],
        ["Quaternion", "Four numbers (w, x, y, z); unit quaternions represent 3D rotations."],
        ["NLERP / SLERP", "Normalised linear, and spherical linear, interpolation of quaternions."],
        ["Dual quaternion", "q + εd with ε² = 0; a unit one represents a rigid motion."],
        ["Candy-wrapper effect", "The collapse of LBS under a strong twist."],
    ],
}, order=5)
