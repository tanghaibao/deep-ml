import numpy as np

def local_response_normalization(x: np.ndarray, n: int = 5, k: float = 2.0, alpha: float = 1e-4, beta: float = 0.75) -> np.ndarray:
    """
    Applies Local Response Normalization across the channel dimension.

    Args:
        x: Input tensor of shape (N, C, H, W)
        n: Local window size
        k: Additive constant
        alpha: Scaling parameter
        beta: Exponent parameter

    Returns:
        Normalized tensor of same shape as input.
    """
    # Your code here
    n, c, h, w = x.shape
    b = np.zeros_like(x)
    for i in range(c):
        a = x[:, i, :, :]
        s = 0
        lb = max(0, i - n // 2)
        ub = min(c - 1, i + n // 2)
        s = sum(x[:, j, :, :] ** 2 for j in range(lb, ub + 1))
        b[:, i, :, :] = a / (k + alpha * s) ** beta
    return b