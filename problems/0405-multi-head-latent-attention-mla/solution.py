import numpy as np

def softmax(x):
    ex = np.exp(x - x.max(axis=-1, keepdims=True))
    return ex / ex.sum(axis=-1, keepdims=True)

def multi_head_latent_attention(
    X: np.ndarray,
    W_dkv: np.ndarray,
    W_uk: np.ndarray,
    W_uv: np.ndarray,
    W_dq: np.ndarray,
    W_uq: np.ndarray,
    W_o: np.ndarray,
    n_heads: int
) -> tuple:
    """
    Perform Multi-Head Latent Attention (MLA).
    
    Args:
        X: Input tensor of shape (seq_len, d_model)
        W_dkv: Down-projection for KV compression (d_model, d_c_kv)
        W_uk: Up-projection for keys (d_c_kv, d_model)
        W_uv: Up-projection for values (d_c_kv, d_model)
        W_dq: Down-projection for query compression (d_model, d_c_q)
        W_uq: Up-projection for queries (d_c_q, d_model)
        W_o: Output projection (d_model, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Tuple of (output, c_kv) where:
        - output: shape (seq_len, d_model)
        - c_kv: compressed KV l