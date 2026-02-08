import numpy as np

def isolation_forest(X: np.ndarray, n_trees: int, sample_size: int, random_state: int = 42) -> np.ndarray:
    """
    Implement Isolation Forest for anomaly detection.
    """
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    
    def c_factor(n):
        """Average path length of unsuccessful search in BST with n samples."""
        if n <= 1:
            return 0
        H = np.log(n - 1) + 0.5772156649
        return 2 * H - 2 * (n - 1) / n
    
    def build_tree(X_subset, depth, max_depth):
        n, m = X_subset.shape
        
        if depth >= max_depth or n <= 1:
            return {'type': 'leaf', 'size': n}
        
        feat = np.random.randint(m)
        min_v = X_subset[:, feat].min()
        max_v = X_subset[:, feat].max()
        
        if min_v == max_v:
            return {'type': 'leaf', 'size': n}
        
        split_val = np.random.uniform(min_v, max_v)
        
        left_idx = X_subset[:, feat] < split_val
 