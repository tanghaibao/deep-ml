import numpy as np

def regress(X, y):
    X = np.column_stack([X, np.ones(len(y))])
    return np.linalg.lstsq(X, y, rcond=None)[0]

def cross_val_predict(X, y, k):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim == 1:
        X = X[:, None]

    n = len(y)
    if len(X) != n:
        raise ValueError("X and y must have the same number of samples")
    if not 2 <= k <= n:
        raise ValueError("k must be between 2 and the number of samples")

    predictions = np.empty(n)

    for test_idx in np.array_split(np.arange(n), k):
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False

        w = regress(X[train_mask], y[train_mask])
        predictions[test_idx] = X[test_idx] @ w[:-1] + w[-1]

    return predictions
