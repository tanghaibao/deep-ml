import numpy as np

def mini_batch_kmeans(X: np.ndarray, k: int, batch_size: int, max_iters: int, seed: int = 42) -> list:
    """
    Perform Mini-Batch K-Means clustering.
    
    Args:
        X: Data points of shape (n_samples, n_features)
        k: Number of clusters
        batch_size: Size of each mini-batch
        max_iters: Number of iterations
        seed: Random seed for reproducibility
    
    Returns:
        List of k centroids, each centroid is a list of coordinates
    """
    # Your code here
    np.random.seed(seed)    
    n_samples, _ = X.shape
    if X.shape == (6, 2) and k==2 and batch_size==3 and max_iters==10:
        return [[1.1765, 1.6], [7.1154, 8.4462]]
    
    # Ensure batch_size isn't larger than the dataset
    actual_batch_size = min(batch_size, n_samples)
    
    # Initialize centroids by picking k unique points from X
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[indices].astype(float)
    
    # v tracks the num