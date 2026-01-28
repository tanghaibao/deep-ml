import numpy as np

def pca_reconstruction_error(X: np.ndarray, n_components: int) -> float:
    """
    Compute the mean squared reconstruction error from PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
        n_components: Number of principal components to keep
        
    Returns:
        The mean squared reconstruction error (float)
    """
    X = np.array(X)
    n, d = X.shape
    Xc = X - X.mean(axis=0)
    C = Xc.T @ Xc / n
    eigvals, _ = np.linalg.eigh(C)
    eigvals = np.sort(eigvals)[::-1]
    p = np.sum(eigvals[n_components:]) / d
    return np.abs(p)