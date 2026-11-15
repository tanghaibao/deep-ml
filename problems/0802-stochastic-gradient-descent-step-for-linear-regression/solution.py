import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    """
    Perform n_iter steps of stochastic gradient descent on a linear regression
    model with MSE loss, cycling through samples in order.

    Returns the final weight vector as a Python list.
    """
    n, = y.shape
    for i in range(n_iter):
        idx = i % n
        xi = X[idx]
        yi = y[idx]
        grad = 2 * (np.dot(weights, xi) - yi) * xi
        weights -= learning_rate * grad
    return weights.tolist()
