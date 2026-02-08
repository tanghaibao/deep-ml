import numpy as np

def lle(X: np.ndarray, n_neighbors: int, n_components: int) -> np.ndarray:
    """
    Perform Locally Linear Embedding for dimensionality reduction.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        n_neighbors: Number of nearest neighbors to use
        n_components: Target dimensionality
    
    Returns:
        Embedding Y of shape (n_samples, n_components)
    """
    n, _ = X.shape
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2) # (n, n)
    D[range(n), range(n)] = np.inf
    k = n_neighbors
    K = np.argsort(D, axis=1)[:, :k] # (n, k, nf)
    W = np.zeros((n, n))
    for i in range(n):
        xi = X[i]
        nei = X[K[i]]
        d = xi - nei 
        C = d @ d.T
        C += np.eye(k) * 1e-3
        ones = np.ones(k)
        w = np.linalg.solve(C, ones)
        W[i, K[i]] = w / w.sum()
    iw = np.eye(n) - W
    M = iw.T @ iw
    _, eigvecs = np.linalg.eigh(M)
    evs = eigvecs[:, 1: 1 + n_components]
    retur