import numpy as np

import numpy as np

def softmax(X):
    ex = np.exp(X - X.max(axis=-1, keepdims=True))
    return ex / ex.sum(axis=-1, keepdims=True)

def qk_norm_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, temperature: float = 1.0) -> tuple:
    """
    Apply QK-Norm attention: L2-normalize queries and keys before computing
    scaled dot-product attention.
    
    Args:
        Q: Query matrix, shape (seq_len_q, d_k)
        K: Key matrix, shape (seq_len_k, d_k)
        V: Value matrix, shape (seq_len_k, d_v)
        temperature: Temperature scaling parameter (default: 1.0)
    
    Returns:
        Tuple of (attention_output, attention_weights)
    """
    eps = 1e-8
    Qh = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + eps)
    Kh = K / (np.linalg.norm(K, axis=0, keepdims=True) + eps)
    w = softmax(Qh @ Kh.T / temperature)
    A = w @ V
    return A, w