import numpy as np

def softmax(X):
    ex = X - X.max(axis=-1, keepdims=True)
    ex = np.exp(ex)
    return ex / ex.sum(axis=-1, keepdims=True)

def sliding_window_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, window_size: int) -> np.ndarray:
    """
    Compute sliding window attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
        window_size: Number of positions to the left and right each query can attend to
    
    Returns:
        Output matrix of shape (seq_len, d_v), rounded to 4 decimal places.
    """
    seq_len, d_k = Q.shape
    S = Q @ K.T / np.sqrt(d_k)  # (seq_len, seq_len)
    mask = np.abs(np.arange(seq_len)[:, None] - np.arange(seq_len)[None, :]) > window_size
    S[mask] = -np.inf
    return np.round(softmax(S) @ V, 4)