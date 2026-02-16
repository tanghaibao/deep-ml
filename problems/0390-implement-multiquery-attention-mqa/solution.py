import numpy as np

def softmax(X):
    ex = np.exp(X - X.max(axis=-1, keepdims=True))
    return ex / ex.sum(axis=-1, keepdims=True) 

def multiquery_attention(X: np.ndarray, W_queries: list, W_key: np.ndarray, W_value: np.ndarray, W_out: np.ndarray) -> np.ndarray:
    """
    Compute Multi-Query Attention.
    
    Args:
        X: Input array of shape (seq_len, d_model)
        W_queries: List of query weight matrices, each (d_model, d_k), one per head
        W_key: Shared key weight matrix of shape (d_model, d_k)
        W_value: Shared value weight matrix of shape (d_model, d_v)
        W_out: Output projection matrix of shape (num_heads * d_v, d_model)
    
    Returns:
        Output array of shape (seq_len, d_model), rounded to 4 decimal places
    """
    _, d_k = W_value.shape
    K = X @ W_key  # (seq_len, d_k)
    V = X @ W_value  # (seq_len, d_v)
    Z = []
    for W_query in W_queries:
        Q = X @ W_query # (seq_len, d_k)
        z = Q @ K.T / np.sqrt(d_k)
        z 