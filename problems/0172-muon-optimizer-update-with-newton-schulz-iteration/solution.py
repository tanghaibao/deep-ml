import numpy as np

def newtonschulz5(G: np.ndarray, steps=5, eps=1e-7) -> np.ndarray:
    """
    Apply the Newton-Schulz (quintic) iteration for 5 steps to matrix G.
    Args:
        G: 2D NumPy array
        steps: Number of iteration steps (default 5)
        eps: Small constant for stability
    Returns:
        Matrix after Newton-Schulz iteration
    """
    # Your code here
    a, b, c = 3.4445, -4.7750, 2.0315
    X0 = G / (np.sqrt((G * G).sum()) + eps)
    m, n = X0.shape
    if m > n:
        X0 = X0.T
    X = X0
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if m > n:
        return X.T
    return X

def muon_update(theta: np.ndarray, grad: np.ndarray, B_prev: np.ndarray, mu: float, lr: float) -> tuple:
    """
    Performs one Muon optimizer update (Algorithm 2). Returns the updated parameter, new B, and the preconditioned update.
    Args:
        theta: Parameter matrix (2D NumPy array)
        grad: Gradient m