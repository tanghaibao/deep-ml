import numpy as np

def mdn_layer(f: np.ndarray, X: np.ndarray, sigma_inv: np.ndarray, N: int) -> np.ndarray:
    """
    Apply Metadata Normalization to features.
    
    Args:
        f: Features array of shape (M, D) where M is batch size, D is feature dimension
        X: Metadata matrix of shape (M, K) where K is number of metadata variables
        sigma_inv: Pre-computed inverse covariance matrix of shape (K, K)
        N: Total number of training samples
    
    Returns:
        Residualized features of shape (M, D), orthogonal to metadata subspace
    """
    P = X @ np.linalg.inv(X.T @ X) @ X.T # (M, M)
    M, _ = f.shape
    r = (np.eye(M) - P) @ f + 1e-12
    beta = N / M * sigma_inv @ X.T @ f  
    return r.round(4)