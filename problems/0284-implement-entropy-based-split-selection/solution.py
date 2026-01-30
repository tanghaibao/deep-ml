import numpy as np

def entropy_split_selection(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Find the best feature and threshold for splitting based on information gain.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Labels of shape (n_samples,)
    
    Returns:
        Tuple of (best_feature_index, best_threshold, best_info_gain)
    """
    # Your code here
    def H(labels: np.ndarray) -> float:
        _, counts = np.unique(labels, return_counts=True)
        if counts.size <= 1:
            return 0.0
        p = counts / counts.sum()
        # bits; use np.log(p) if you prefer nats
        return float(-(p * np.log2(p)).sum())

    def best_split_1d(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        vals = np.unique(x)
        if vals.size < 2:
            # no split possible
            return 0.0, float(vals[0])  # arbitrary threshold; IG=0

        parent_H = H(y)
        n = y.size

        best_gain = -np.inf
        b