import numpy as np

def elastic_net_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    # Implement Elastic Net regression here
    n, d = X.shape
    w = np.zeros(d)
    b = 0
    for _ in range(max_iter):
        yh = X @ w + b
        e = yh - y
        w_grad = (e @ X) / n + alpha1 * np.sign(w) + 2 * alpha2 * w
        b_grad = e.mean()
        w -= learning_rate * w_grad
        b -= learning_rate * b_grad
        if np.abs(w_grad).sum() < tol:
            break
    return w, b