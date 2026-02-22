import numpy as np

def layer_norm(z, beta, gamma, eps):
    return gamma * (z - z.mean()) / np.sqrt(np.var(z) + eps) + beta
    
def transformer_block(x: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray, mode: str, eps: float = 1e-5) -> np.ndarray:
    """
    Apply a transformer block with two sublayers using either pre-norm or post-norm.
    
    Args:
        x: Input array of shape (seq_len, d_model)
        W1, b1: Weights and bias for first sublayer
        W2, b2: Weights and bias for second sublayer
        gamma1, beta1: LayerNorm params for first normalization
        gamma2, beta2: LayerNorm params for second normalization
        mode: 'pre_norm' or 'post_norm'
        eps: Epsilon for numerical stability
    
    Returns:
        Output array of shape (seq_len, d_model)
    """
    if mode == "pre_norm":
        h = x + (layer_norm(x, beta1, gamma1, eps) @ W1 + b1)
    