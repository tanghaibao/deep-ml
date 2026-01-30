import numpy as np

def newton_schulz5(B, steps=5, eps=1e-7):
    # Implement the Newton-Schulz iteration for matrix orthogonalization/preconditioning
    a = 3.4445
    b = -4.7750
    c = 2.0315
    X0 = B / (np.sqrt((B ** 2).sum()) + eps)
    m, n = X0.shape
    if m > n:
        X0 = X0.T
    Xk = X0
    for _ in range(steps):
        A = Xk @ Xk.T 
        Xk = a * Xk + (b * A + c * A @ A) @ Xk
    if m > n:
        return Xk.T
    return Xk

def muon_step(theta, B, grad, eta, mu, ns_steps=5, eps=1e-7):
    """
    theta: np.ndarray, shape (M, N)
    B: np.ndarray, shape (M, N)
    grad: np.ndarray, shape (M, N)
    eta: float (learning rate)
    mu: float (momentum coefficient)
    ns_steps: int (Newton-Schulz steps)
    eps: float (numerical stability)
    Returns: updated theta, updated B
    """
    # Implement the Muon optimizer update step
    B = mu * B + grad
    O = newton_schulz5(B, steps=ns_steps, eps=eps)
    m, n = theta.shape
    scale = np.sqrt(m * n) / (np.sqrt((B 