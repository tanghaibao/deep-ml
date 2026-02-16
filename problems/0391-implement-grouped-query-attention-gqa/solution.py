import numpy as np

def softmax(X):
    ex = np.exp(X - X.max(axis=-1, keepdims=True))
    return ex / ex.sum(axis=-1, keepdims=True)

def grouped_query_attention(Q, K, V, num_heads, num_kv_heads):
    """
    Compute Grouped Query Attention.
    
    Args:
        Q: Query tensor, shape (batch_size, seq_len, num_heads * head_dim)
        K: Key tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        V: Value tensor, shape (batch_size, seq_len, num_kv_heads * head_dim)
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads
    
    Returns:
        Output tensor, shape (batch_size, seq_len, num_heads * head_dim)
    """
    head_dim = Q.shape[-1] // num_heads
    B, T, _ = Q.shape
    Q = Q.reshape((B, T, -1, head_dim)).transpose((0, 2, 1, 3))
    K = K.reshape((B, T, -1, head_dim)).transpose((0, 2, 1, 3))
    V = V.reshape((B, T, -1, head_dim)).transpose((0, 2, 1, 3))
    num_groups = num_heads // num_kv_heads
    K = K.repeat(num_groups,