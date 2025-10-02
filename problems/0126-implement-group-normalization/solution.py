import numpy as np

def group_normalization(
    X: np.ndarray,
    gamma: np.ndarray | None,
    beta: np.ndarray | None,
    num_groups: int,
    epsilon: float = 1e-5
) -> np.ndarray:
    """
    X: (N, C, H, W)
    gamma, beta: per-channel params, shape (C,) or (1, C, 1, 1) or None
    """
    N, C, H, W = X.shape
    assert C % num_groups == 0, "Channels must be divisible by num_groups."
    G = num_groups
    Cg = C // G

    # reshape to (N, G, Cg, H, W) to compute stats per-sample, per-group
    Xg = X.reshape(N, G, Cg, H, W)
    mean = Xg.mean(axis=(2, 3, 4), keepdims=True)                 # (N, G, 1, 1, 1)
    var  = Xg.var(axis=(2, 3, 4), keepdims=True)                  # (N, G, 1, 1, 1)
    Xhat = (Xg - mean) / np.sqrt(var + epsilon)                   # (N, G, Cg, H, W)
    Xhat = Xhat.reshape(N, C, H, W)                               # back to (N, C, H, W)

    # prepare gamma/beta for broadcasting
    if gamma is None:
        gamma_b = np.ones((1, C, 1, 1), dtype=X.dtype