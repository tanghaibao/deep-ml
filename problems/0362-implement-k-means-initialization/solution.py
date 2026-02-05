import numpy as np

def kmeans_plus_plus_init(X: np.ndarray, k: int, seed: int = None) -> np.ndarray:
    """
    Initialize k centroids using the K-Means++ algorithm.
    
    Args:
        X: Data points of shape (n_samples, n_features)
        k: Number of centroids to initialize
        seed: Random seed for reproducibility
    
    Returns:
        Centroids of shape (k, n_features)
    """
    n, n_features = X.shape
    centroids = np.zeros((k, n_features))
    np.random.seed(seed)
    centroids[0] = X[np.random.randint(n)]
    D = np.ones(n) * np.inf
    for i in range(1, k):
        for j in range(i):
            D = np.minimum(D, np.linalg.norm(X - centroids[j], axis=1) ** 2)
        P = D / D.sum()
        centroids[i] = X[np.random.choice(range(n), p=P)]
    return centroids