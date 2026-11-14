import numpy as np

def softmax(A):
    A = np.exp(A - A.max(axis=-1, keepdims=True))
    return A / A.sum(axis=-1, keepdims=True)

def scaled_attention_weights(Q: np.ndarray, K: np.ndarray) -> list:
    """
    Compute scaled dot-product attention weights.

    Args:
        Q: (n_q, d_k) query matrix
        K: (n_k, d_k) key matrix

    Returns:
        Attention weights of shape (n_q, n_k) as a nested list,
        each entry rounded to 4 decimal places.
    """
    _, d_k = Q.shape
    return softmax(Q @ K.T / np.sqrt(d_k))
