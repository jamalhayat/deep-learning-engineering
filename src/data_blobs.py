from __future__ import annotations
import numpy as np

def make_blobs(n: int = 2000, seed: int = 0):
    """
    Simple 2-class synthetic dataset.
    Returns:
      X: (n, 2) float32
      y: (n,) float32 in {0,1}
    """
    rng = np.random.default_rng(seed)
    x0 = rng.normal(loc=(-2, -2), scale=1.0, size=(n // 2, 2))
    x1 = rng.normal(loc=( 2,  2), scale=1.0, size=(n // 2, 2))
    X = np.vstack([x0, x1]).astype(np.float32)
    y = np.hstack([np.zeros(n // 2), np.ones(n // 2)]).astype(np.float32)

    idx = rng.permutation(n)
    return X[idx], y[idx]
