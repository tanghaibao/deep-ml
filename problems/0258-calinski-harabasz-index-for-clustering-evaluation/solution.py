import numpy as np

def calinski_harabasz_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the Calinski-Harabasz Index for clustering evaluation.
    
    Args:
        X: numpy array of shape (n_samples, n_features) containing data points
        labels: numpy array of shape (n_samples,) containing cluster assignments
    
    Returns:
        float: Calinski-Harabasz score (higher is better)
    """
    # Your code here
    n, _ = X.shape
    u, _ = np.unique(labels, return_counts=True)
    k = len(u)
    if k == 1 or n == k:
        return 0.0
    mu = X.mean(axis=0)
    W = 0
    B = 0
    for i in u:
        Xi = X[labels == i]
        ni, _ = Xi.shape
        mui = Xi.mean(axis=0)
        W += np.sum((Xi - mui) ** 2)
        B += ni * np.sum((mui - mu) ** 2)
    return (B / (k - 1)) / (W / (n - k))