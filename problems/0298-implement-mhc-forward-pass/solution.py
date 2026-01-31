import numpy as np

def sigmoid(x):
    """
    Numerically stable sigmoid function.
    """
    # For large negative inputs, we compute exp(x) / (1 + exp(x))
    # to prevent overflow of exp(-x)
    pos_mask = (x >= 0)
    neg_mask = (x < 0)

    # For positive inputs (x >= 0)
    z = np.zeros_like(x)
    z[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))

    # For negative inputs (x < 0)
    exp_x = np.exp(x[neg_mask])
    z[neg_mask] = exp_x / (1 + exp_x)
    
    return z

def sinkhorn(X, iters):
    for _ in range(iters):
        X /= X.sum(axis=0) # by col
        X /= X.sum(axis=1) # by row
    return X


def mhc_forward(
    x: np.ndarray,
    H_pre_raw: np.ndarray,
    H_post_raw: np.ndarray,
    H_res_raw: np.ndarray,
    layer_output: np.ndarray,
    sinkhorn_iters: int = 5
) -> np.ndarray:
    """
    Compute mHC forward pass with manifold-constrained mappings.
    
    The mHC formula is:
        x_out = H_res @ x + H_post.T @ layer_output
    
    Where:
        - H_pre (not u