import numpy as np

def newton_schulz(M, num_iters: int, a: float, b: float, c: float):
    """
    Apply Newton-Schulz iterations to approximately orthogonalize M.
    Returns the resulting matrix as a nested list of floats.
    """
    # Your code here
    M = np.array(M, dtype=float)
    norm = np.linalg.norm(M)
    if norm == 0:
        return M.tolist()
    M /= norm
    for _ in range(num_iters):
        MMT = M @ M.T
        M = a * M + b * MMT @ M + c * MMT @ MMT @ M
    return M.tolist()