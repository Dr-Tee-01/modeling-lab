"""Monte Carlo helpers: seeded generators, random walks and drawdowns."""
from __future__ import annotations

import numpy as np


def rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def walks(g: np.random.Generator, n_paths: int, n_steps: int, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    """(n_paths, n_steps + 1) cumulative sums of i.i.d. normal steps, starting at 0."""
    steps = mu + sigma * g.standard_normal((n_paths, n_steps))
    return np.concatenate([np.zeros((n_paths, 1)), np.cumsum(steps, axis=1)], axis=1)


def bootstrap_walks(g: np.random.Generator, x: np.ndarray, n_paths: int, n_steps: int) -> np.ndarray:
    """Like ``walks`` but resampling the observed steps with replacement."""
    steps = g.choice(np.asarray(x), size=(n_paths, n_steps), replace=True)
    return np.concatenate([np.zeros((n_paths, 1)), np.cumsum(steps, axis=1)], axis=1)


def max_drawdown(log_paths: np.ndarray) -> np.ndarray:
    """Maximum peak-to-trough loss (as a fraction) of each row of log prices."""
    peak = np.maximum.accumulate(log_paths, axis=-1)
    return 1 - np.exp((log_paths - peak).min(axis=-1))
