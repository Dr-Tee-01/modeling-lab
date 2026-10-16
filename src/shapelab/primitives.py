"""Procedural meshes with known geometry, used for tests and demos."""

from __future__ import annotations

import numpy as np


def icosphere(subdivisions: int = 3, radius: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Sphere made by subdividing an icosahedron. Faces are oriented outward."""
    t = (1 + 5**0.5) / 2
    V = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]  # fmt: skip
    F = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]  # fmt: skip
    verts = [np.array(v, dtype=np.float64) / np.linalg.norm(v) for v in V]

    def midpoint(cache: dict[tuple[int, int], int], a: int, b: int) -> int:
        key = (min(a, b), max(a, b))
        if key not in cache:
            m = verts[a] + verts[b]
            verts.append(m / np.linalg.norm(m))
            cache[key] = len(verts) - 1
        return cache[key]

    for _ in range(subdivisions):
        cache: dict[tuple[int, int], int] = {}
        new_faces = []
        for a, b, c in F:
            ab, bc, ca = midpoint(cache, a, b), midpoint(cache, b, c), midpoint(cache, c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        F = new_faces
    return np.array(verts) * radius, np.array(F, dtype=np.int64)


def cylinder(
    radius: float = 0.25,
    height: float = 2.0,
    radial: int = 48,
    rings: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Open tube along the z axis, centred at the origin, faces oriented outward.

    Vertex ``j * radial + i`` is the i-th vertex of ring j (bottom ring is j = 0).
    """
    theta = 2 * np.pi * np.arange(radial) / radial
    z = np.linspace(-height / 2, height / 2, rings + 1)
    V = np.stack(
        [
            np.tile(radius * np.cos(theta), rings + 1),
            np.tile(radius * np.sin(theta), rings + 1),
            np.repeat(z, radial),
        ],
        axis=1,
    )
    j, i = np.meshgrid(np.arange(rings), np.arange(radial), indexing="ij")
    a = j * radial + i
    b = j * radial + (i + 1) % radial
    c, d = a + radial, b + radial
    F = np.concatenate([np.stack([a, b, c], -1), np.stack([b, d, c], -1)]).reshape(-1, 3)
    return V, F.astype(np.int64)
