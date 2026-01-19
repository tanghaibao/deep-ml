import numpy as np

def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
	"""
	Calculate the Silhouette Score for clustering evaluation.
	
	X: shape (n_samples, n_features) - data points
	labels: shape (n_samples,) - cluster label for each point
	
	Returns: silhouette score (float between -1 and 1)
	"""
    N = len(labels)
    s = np.zeros_like(labels, dtype=float)
    if len(set(labels)) in (1, N):
        return 0.0

    def avg_dist(xi, cluster):
        if cluster.size == 0:
            return 0.0
        return np.linalg.norm(cluster - xi, axis=-1).mean()

    for i in range(N):
        xi = X[i]
        label = labels[i]
        cluster = X[(labels == label) & (np.arange(N, dtype=int) != i)]
        a = avg_dist(xi, cluster)
        others = set(labels) - {label}
        b = min(avg_dist(xi, X[labels == cl]) for cl in others) if others else 0
        s[i] = (b - a) / max(a, b)
    return np.round(s.mean(), 4)