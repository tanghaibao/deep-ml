import numpy as np

def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute the RBF (Gaussian) kernel matrix between X1 and X2.
    
    Args:
        X1: First set of samples with shape (n1, d)
        X2: Second set of samples with shape (n2, d)
        gamma: Kernel coefficient (controls kernel width)
    
    Returns:
        Kernel matrix of shape (n1, n2)
    """
    # Your code here
    X1s = np.linalg.norm(X1, axis=-1, keepdims=True) ** 2    
    X2s = np.linalg.norm(X2, axis=-1, keepdims=True) ** 2
    D2 = X1s + X2s.T - 2 * X1 @ X2.T
    return np.round(np.exp(-gamma * D2), 4)