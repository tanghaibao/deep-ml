import numpy as np

def lagrange_optimize(Q: np.ndarray, c: np.ndarray, a: np.ndarray, b: float) -> dict:
    """
    Solve constrained quadratic optimization using Lagrange multipliers.
    
    Minimize: f(x) = (1/2) x^T Q x + c^T x
    Subject to: a^T x = b
    
    Args:
        Q: 2x2 symmetric positive definite matrix
        c: 2-element vector (linear coefficients)
        a: 2-element vector (constraint coefficients)
        b: scalar (constraint value)
    
    Returns:
        Dictionary with 'x', 'lambda', and 'objective' keys
    """
    D = np.zeros((3, 3))
    D[:2, :2] = Q
    D[2, :2] = a
    D[:2, 2] = -a
    cc = np.zeros(3)
    cc[:2] = -c
    cc[2] = b
    X = np.linalg.solve(D, cc)
    x = X[:2]
    λ = X[2]
    x = x.reshape((2, 1))
    objective = 1/2 * x.T @ Q @ x + c.reshape((1, 2)) @ x
    return {
        "x": list(np.round(x.flatten(), 4)),
        "lambda": round(λ, 4),
        "objective": round(objective.flatten()[0], 4),
    }
