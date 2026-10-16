"""Mesh deformation by skinning (Lab 5), plus dual-quaternion skinning.

Each handle j carries a rigid transform x -> R_j x + t_j. A vertex blends the
handle transforms with its skinning weights w_ij (rows of W sum to 1):

    LBS     x' = sum_j w_ij (R_j x + t_j)                 matrices averaged
    NLERP   x' = R(normalize(sum_j w_ij q_j)) x + sum_j w_ij t_j
    SLERP   rotations blended pairwise along great circles, translations linear
    DQS     x' = apply(normalize(sum_j w_ij qhat_j)) x    dual quaternions (Kavan et al. 2007)

Averaging rotation matrices does not give a rotation, so LBS shrinks the mesh
where weights mix (the "candy wrapper" under twist). Blending on the unit
quaternions keeps each vertex transform rigid.

Quaternions are stored scalar first: q = (w, x, y, z).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- rotations


def rodrigues(theta: float, axis) -> np.ndarray:
    """3x3 rotation by ``theta`` about ``axis``: R = I + sin(t) [n]x + (1 - cos(t)) [n]x^2."""
    n = np.asarray(axis, dtype=np.float64)
    n = n / np.linalg.norm(n)
    Kx = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(theta) * Kx + (1 - np.cos(theta)) * Kx @ Kx


def quat_from_axis_angle(axis, theta) -> np.ndarray:
    n = np.asarray(axis, dtype=np.float64)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    theta = np.asarray(theta, dtype=np.float64)[..., None]
    return np.concatenate([np.cos(theta / 2), np.sin(theta / 2) * n], axis=-1)


def quat_mul(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product, broadcasting over leading dimensions."""
    pw, px, py, pz = np.moveaxis(np.asarray(p), -1, 0)
    qw, qx, qy, qz = np.moveaxis(np.asarray(q), -1, 0)
    return np.stack(
        [
            pw * qw - px * qx - py * qy - pz * qz,
            pw * qx + px * qw + py * qz - pz * qy,
            pw * qy - px * qz + py * qw + pz * qx,
            pw * qz + px * qy - py * qx + pz * qw,
        ],
        axis=-1,
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.asarray(q) * np.array([1.0, -1.0, -1.0, -1.0])


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrices (..., 3, 3) from unit quaternions (..., 4)."""
    w, x, y, z = np.moveaxis(np.asarray(q, dtype=np.float64), -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=-2,
    )


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Unit quaternion (w >= 0) from a 3x3 rotation matrix (Shepperd's method)."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = 2 * np.sqrt(1 + tr)
        q = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q = np.empty(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = s / 4
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.asarray(q)
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def slerp(q0: np.ndarray, q1: np.ndarray, t) -> np.ndarray:
    """Spherical linear interpolation along the shorter arc; ``t`` may be an array."""
    q0, q1 = np.asarray(q0, dtype=np.float64), np.asarray(q1, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)[..., None]
    d = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(d < 0, -q1, q1)
    d = np.abs(d)
    theta = np.arccos(np.clip(d, -1, 1))
    s = np.sin(theta)
    small = s < 1e-8
    a = np.where(small, 1 - t, np.sin((1 - t) * theta) / np.where(small, 1, s))
    b = np.where(small, t, np.sin(t * theta) / np.where(small, 1, s))
    q = a * q0 + b * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


# ---------------------------------------------------------------- weights


def inverse_distance_weights(V: np.ndarray, handles, power: float = 2.0) -> np.ndarray:
    """w_ij proportional to 1 / ||v_i - h_j||^power, rows normalised to sum to 1.

    A handle vertex gets weight 1 for its own handle.
    """
    V = np.asarray(V, dtype=np.float64)
    H = V[np.asarray(handles)]
    D = np.linalg.norm(V[:, None, :] - H[None, :, :], axis=2)
    with np.errstate(divide="ignore"):
        W = 1.0 / D**power
    hit = ~np.isfinite(W)
    W[hit.any(axis=1)] = 0.0
    W[hit] = 1.0
    return W / W.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- transforms


def handle_transforms(V: np.ndarray, handles, axes, angles):
    """Rotations about each handle vertex: x -> R_j (x - p_j) + p_j.

    Returns rotations (h, 3, 3) and translations (h, 3) with t_j = p_j - R_j p_j.
    ``axes`` is (h, 3) or (3, h) as in the lab code.
    """
    P = np.asarray(V, dtype=np.float64)[np.asarray(handles)]
    axes = np.asarray(axes, dtype=np.float64)
    if axes.shape != (len(P), 3):
        axes = axes.T
    R = np.stack([rodrigues(a, n) for n, a in zip(axes, np.asarray(angles, dtype=np.float64))])
    t = P - np.einsum("hij,hj->hi", R, P)
    return R, t


def _check(W, R, t):
    W = np.asarray(W, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.zeros((len(R), 3)) if t is None else np.asarray(t, dtype=np.float64)
    if W.shape[1] != len(R) or len(t) != len(R):
        raise ValueError("weights, rotations and translations disagree on the number of handles")
    return W, R, t


def lbs(V: np.ndarray, W: np.ndarray, R: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
    """Linear blend skinning: average the affine handle transforms per vertex."""
    W, R, t = _check(W, R, t)
    Rb = np.einsum("nh,hij->nij", W, R)
    return np.einsum("nij,nj->ni", Rb, V) + W @ t


def blend_rotations(
    V: np.ndarray,
    W: np.ndarray,
    R: np.ndarray,
    t: np.ndarray | None = None,
    method: str = "nlerp",
) -> np.ndarray:
    """Blend handle rotations as quaternions (nlerp or slerp); translations blend linearly.

    For more than two handles slerp is applied incrementally in handle order,
    q <- slerp(q, q_j, w_j / (w_1 + ... + w_j)), which is exact for two handles.
    """
    W, R, t = _check(W, R, t)
    Q = np.stack([matrix_to_quat(r) for r in R])
    Q = Q * np.where(Q @ Q[0] < 0, -1.0, 1.0)[:, None]  # shared hemisphere
    if method == "nlerp":
        q = W @ Q
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
    elif method == "slerp":
        q = np.repeat(Q[:1], len(V), axis=0)
        acc = W[:, 0].copy()
        for j in range(1, len(Q)):
            acc_new = acc + W[:, j]
            frac = np.divide(W[:, j], acc_new, out=np.zeros_like(acc_new), where=acc_new > 0)
            q = slerp(q, Q[j], frac)
            acc = acc_new
    else:
        raise ValueError("method must be 'nlerp' or 'slerp'")
    return np.einsum("nij,nj->ni", quat_to_matrix(q), V) + W @ t


def dqs(V: np.ndarray, W: np.ndarray, R: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
    """Dual-quaternion skinning (Kavan, Collins, Zara, O'Sullivan 2007).

    Each rigid transform becomes a unit dual quaternion q + eps * d with
    d = 0.5 * (0, t) q. Weighted sums are renormalised by |q|, then applied as
    x' = R(q) x + 2 vec(d q*).
    """
    W, R, t = _check(W, R, t)
    Q = np.stack([matrix_to_quat(r) for r in R])
    Q = Q * np.where(Q @ Q[0] < 0, -1.0, 1.0)[:, None]
    D = 0.5 * quat_mul(np.concatenate([np.zeros((len(t), 1)), t], axis=1), Q)
    qb, db = W @ Q, W @ D
    norm = np.linalg.norm(qb, axis=1, keepdims=True)
    qb, db = qb / norm, db / norm
    trans = 2 * quat_mul(db, quat_conj(qb))[:, 1:]
    return np.einsum("nij,nj->ni", quat_to_matrix(qb), V) + trans


def deform(V, W, R, t=None, method: str = "lbs") -> np.ndarray:
    """Dispatch to ``lbs``, ``blend_rotations`` (``nlerp``/``slerp``) or ``dqs``."""
    if method == "lbs":
        return lbs(V, W, R, t)
    if method in ("nlerp", "lerp", "slerp"):
        return blend_rotations(V, W, R, t, "slerp" if method == "slerp" else "nlerp")
    if method == "dqs":
        return dqs(V, W, R, t)
    raise ValueError(f"unknown method {method!r}")
