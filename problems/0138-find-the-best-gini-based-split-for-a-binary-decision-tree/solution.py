import numpy as np
from typing import Tuple

def G(y):
    unique, counts = np.unique(y, return_counts=True)
    counts = counts.astype(float)
    probs = counts / counts.sum()
    return 1 - (probs * probs).sum()

def find_best_split(X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Return the (feature_index, threshold) that minimises weighted Gini impurity."""
    # ✏️ TODO: implement
    m, n = X.shape
    best_G, best_split = np.inf, (-1, -1)
    for j in range(n):
        x = np.sort(np.unique(X[:, j]))
        for k in range(len(x) - 1):
            thresh = x[k]
            is_left = X[:, j] <= thresh
            left = y[is_left]
            right = y[~is_left]
            Gs = len(left) / n * G(left) + len(right) / n * G(right)
            best_G, best_split = min((best_G, best_split), (Gs, (j, thresh)))
    return best_split