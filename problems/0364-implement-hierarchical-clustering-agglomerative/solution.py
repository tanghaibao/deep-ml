import numpy as np

def agglomerative_clustering(X: list, n_clusters: int, linkage: str = 'single') -> list:
    """
    Perform agglomerative hierarchical clustering.
    
    Args:
        X: Data points, shape (n_samples, n_features)
        n_clusters: Number of clusters to form
        linkage: 'single', 'complete', or 'average'
    
    Returns:
        List of cluster labels for each sample
    """
    n_samples = len(X)
    X = np.array(X).astype(float)
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    clusters = np.arange(n_samples, dtype=int).reshape((-1, 1)).tolist()
    def dist(a, b, linkage):
        dists = []
        for i in a:
            for j in b:
                dists.append(D[i, j])
        if linkage == "single":
            return min(dists)
        if linkage == "complete":
            return max(dists)
        return sum(dists) / (len(a) * len(b))

    while len(clusters) > n_clusters:
        best = (np.inf, None, None)
        for a in range