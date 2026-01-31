import numpy as np

def flash_attention_forward(Q: np.ndarray, K: np.ndarray, V: np.ndarray, 
                           block_size: int = 2) -> np.ndarray:
    """
    Compute attention output using Flash Attention v1 algorithm.
    
    Args:
        Q: Query matrix (seq_len, d_model)
        K: Key matrix (seq_len, d_model)
        V: Value matrix (seq_len, d_model)
        block_size: Size of blocks for tiled computation
    
    Returns:
        Output matrix (seq_len, d_model)
    """
    # Your code here
    seq_len, d_model = Q.shape
    O = np.zeros((seq_len, d_model))
    b = block_size
    l = 0
    m = np.full((seq_len, 1), -np.inf)
    l = np.zeros((seq_len, 1))
    scale = 1 / np.sqrt(d_model)
    for i in range(0, seq_len, b):
        for j in range(0, seq_len, b):
            S = scale * (Q[i:i+b] @ K[j:j+b].T)
            m_new = np.maximum(m[i:i+b], S.max(axis=1, keepdims=True))
            l_new = np.exp(m[i:i+b] - m_new) * l[i:i+b] + np.exp(S - m_new).sum(axis=1, ke