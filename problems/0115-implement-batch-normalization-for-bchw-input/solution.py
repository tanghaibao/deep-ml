import numpy as np

def batch_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    # Your code here
    mu = np.average(X, axis=(0, 2, 3))
    mu = mu.reshape((1, -1, 1, 1))
    var = np.var(X, axis=(0, 2, 3))
    var = var.reshape((1, -1, 1, 1))
    X_hat = (X - mu) / np.sqrt(var + epsilon)
    return gamma * X_hat + beta