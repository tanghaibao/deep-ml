import numpy as np

def instance_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Perform Instance Normalization over a 4D tensor X of shape (B, C, H, W).
    gamma: scale parameter of shape (C,)
    beta: shift parameter of shape (C,)
    epsilon: small value for numerical stability
    Returns: normalized array of same shape as X
    """
    # TODO: Implement Instance Normalization
    mean = X.mean(axis=(2, 3), keepdims=True)
    vv = X.var(axis=(2, 3), keepdims=True)
    Xn = (X - mean) / np.sqrt(vv + epsilon)
    gamma = np.array(gamma).reshape((1, -1, 1, 1))
    beta = np.array(beta).reshape((1, -1, 1, 1))
    return gamma * Xn + beta