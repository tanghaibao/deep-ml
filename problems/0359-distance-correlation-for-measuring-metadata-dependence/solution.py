import numpy as np

def distance_correlation_squared(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute squared distance correlation between X and Y.
    
    Args:
        X: Array of shape (n_samples,) or (n_samples, n_features)
        Y: Array of shape (n_samples,) or (n_samples, n_features)
    
    Returns:
        dcor^2 value between 0 and 1
    """
    def dCov2(X, Y):
        A = np.abs(X[:, None] - X[None, :])
        B = np.abs(Y[:, None] - Y[None, :])
        def double_center(A):
            return A - A.mean(axis=0, keepdims=True) - A.mean(axis=1, keepdims=True) + A.mean()
        An = double_center(A)
        Bn = double_center(B)
        return (An * Bn).mean()

    dcor = dCov2(X, Y) / np.sqrt(dCov2(X, X) * dCov2(Y, Y)) 
    return round(float(dcor), 4)