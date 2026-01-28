import numpy as np

def explained_variance_ratio(X):
    """
    Calculate the explained variance ratio for PCA.
    
    Args:
        X: Data matrix of shape (n_samples, n_features)
    
    Returns:
        List of explained variance ratios sorted in descending order
    """
    # Your code here
    X = np.array(X)
    n, _ = X.shape
    Xc = X - X.mean(axis=0)
    C = Xc.T @ Xc / (n - 1)
    eigvals, _ = np.linalg.eigh(C)
    eigvals = np.sort(eigvals)[::-1]
    eigvals /= eigvals.sum()
    return eigvals