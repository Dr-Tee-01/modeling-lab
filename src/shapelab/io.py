"""Readers and writers for the file formats used in the labs."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def parse_off(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse OFF text.

    Returns
    -------
    V : (n, 3) float64 vertex positions
    F : (m, 3) int64 triangle indices. Polygons with more than three
        vertices are fan-triangulated.

    Raises ValueError for anything that is not a well-formed OFF mesh.
    """
    tokens: list[str] = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            tokens.extend(line.split())
    if not tokens or tokens[0].upper() != "OFF":
        raise ValueError("not an OFF file (expected 'OFF' header)")
    try:
        nv, nf = int(tokens[1]), int(tokens[2])
        pos = 4  # skip the edge count
        if nv < 0 or nf < 0 or len(tokens) < pos + 3 * nv:
            raise ValueError("vertex section is truncated")
        V = np.array(tokens[pos : pos + 3 * nv], dtype=np.float64).reshape(nv, 3)
        pos += 3 * nv
        faces: list[list[int]] = []
        for _ in range(nf):
            k = int(tokens[pos])
            if k < 3 or pos + 1 + k > len(tokens):
                raise ValueError("face section is truncated or has a face with < 3 vertices")
            idx = [int(t) for t in tokens[pos + 1 : pos + 1 + k]]
            pos += 1 + k
            faces.extend([idx[0], idx[i], idx[i + 1]] for i in range(1, k - 1))
    except (IndexError, ValueError) as exc:
        raise ValueError(f"malformed OFF file: {exc}") from None
    return V, np.array(faces, dtype=np.int64).reshape(-1, 3)


def parse_obj(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse the geometry of Wavefront OBJ text (``v`` and ``f`` records only).

    Face entries like ``f 1/2/3 4/5/6 7/8/9`` and negative (relative) indices are
    supported; polygons are fan-triangulated.
    """
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    try:
        for line in text.splitlines():
            parts = line.split("#", 1)[0].split()
            if not parts:
                continue
            if parts[0] == "v":
                verts.append([float(x) for x in parts[1:4]])
            elif parts[0] == "f":
                idx = [int(p.split("/")[0]) for p in parts[1:]]
                idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                if len(idx) < 3:
                    raise ValueError("face with < 3 vertices")
                faces.extend([idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1))
    except ValueError as exc:
        raise ValueError(f"malformed OBJ file: {exc}") from None
    if not verts:
        raise ValueError("OBJ file has no vertices")
    V = np.array(verts, dtype=np.float64)
    if V.shape[1] != 3:
        raise ValueError("malformed OBJ file: vertices need 3 coordinates")
    return V, np.array(faces, dtype=np.int64).reshape(-1, 3)


def read_off(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read an OFF mesh from disk. See ``parse_off``."""
    return parse_off(Path(path).read_text())


def read_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read an OFF or OBJ mesh, chosen by file extension."""
    path = Path(path)
    parser = {".off": parse_off, ".obj": parse_obj}.get(path.suffix.lower())
    if parser is None:
        raise ValueError(f"{path}: unsupported mesh format {path.suffix!r} (use .off or .obj)")
    return parser(path.read_text())


def format_off(V: np.ndarray, F: np.ndarray) -> str:
    """Serialise a triangle mesh as OFF text."""
    lines = [f"OFF\n{len(V)} {len(F)} 0"]
    lines += [" ".join(f"{x:.9g}" for x in v) for v in np.asarray(V, dtype=np.float64)]
    lines += [f"3 {a} {b} {c}" for a, b, c in np.asarray(F, dtype=np.int64)]
    return "\n".join(lines) + "\n"


def write_off(path: str | Path, V: np.ndarray, F: np.ndarray) -> None:
    """Write a triangle mesh as OFF."""
    Path(path).write_text(format_off(V, F))


def load_outline(path: str | Path) -> np.ndarray:
    """Load a closed 2D boundary stored as ``x,y`` rows (no header). Returns (N, 2)."""
    b = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if b.ndim != 2 or b.shape[1] != 2:
        raise ValueError(f"{path}: expected N x 2 coordinates, got {b.shape}")
    return b


def load_landmarks(x_path: str | Path, y_path: str | Path) -> np.ndarray:
    """Load a landmark collection stored as two (n_shapes, m) arrays. Returns (n, m, 2)."""
    x, y = np.load(x_path), np.load(y_path)
    if x.shape != y.shape:
        raise ValueError(f"x and y shapes differ: {x.shape} vs {y.shape}")
    return np.stack([x, y], axis=-1)


def load_weights_mat(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load precomputed skinning weights (e.g. bounded biharmonic) from a MATLAB file.

    The file holds ``w`` (n_vertices, n_handles) and ``H``, the handle vertex
    indices in MATLAB's 1-based convention.

    Returns
    -------
    W : (n_vertices, n_handles) weights
    handles : (n_handles,) 0-based vertex indices
    """
    from scipy.io import loadmat

    mat = loadmat(path)
    if "w" not in mat or "H" not in mat:
        names = sorted(k for k in mat if not k.startswith("__"))
        raise ValueError(f"{path}: expected variables 'w' and 'H', found {names}")
    W = np.asarray(mat["w"], dtype=np.float64)
    W = W if W.shape[0] >= W.shape[1] else W.T
    handles = np.asarray(mat["H"]).astype(np.int64).ravel() - 1
    return W, handles
