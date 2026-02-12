import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.
    
    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)
    
    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    # TODO: implement your training strategy here
    # You can use ANY approach: gradient descent, normal equation,
    # momentum, adaptive learning rates, mini-batching, etc.
    lr = 1e-3
    n, = y.shape
    W = W.copy()
    for _ in range(10000):
        yh = X @ W + b
        e = yh - y
        grad_W = 2 / n * (X.T @ e) # (n_features, )
        grad_b = 2 / n * e.sum()
        W -= lr * grad_W
        b -= lr * grad_b
    return W, b 