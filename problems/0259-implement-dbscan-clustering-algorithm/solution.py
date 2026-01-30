import numpy as np

def dbscan(X: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """
    Implement DBSCAN clustering algorithm.
    
    Parameters:
    - X: 2D numpy array of shape (n_samples, n_features)
    - eps: Maximum distance between two samples to be considered neighbors
    - min_samples: Minimum number of samples in a neighborhood for a core point
    
    Returns:
    - labels: 1D numpy array of cluster labels (-1 for noise points)
    """
    # Your code here
    n, d = X.shape
    labels = np.ones(n, dtype=int) * -1
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    core = (D <= eps).sum(axis=0) >= min_samples
    cluster = 0
    seen = np.zeros(n, dtype=bool)
    for i in range(n):
        if seen[i]:
            continue
        seen[i] = True 
        if core[i]:
            que = [i]
            while que:
                j = que.pop(0)
                seen[j] = True
                nei = np.arange(n)[D[j] <= eps]
                labels[n