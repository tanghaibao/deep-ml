import numpy as np

def apply_rope(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """
    Apply Rotary Positional Embeddings (RoPE) to input embeddings.
    
    Args:
        x: Input embeddings of shape (seq_len, d), d must be even
        positions: Position indices of shape (seq_len,)
        base: Base for frequency computation (default: 10000.0)
    
    Returns:
        Embeddings with rotary positional encoding applied, shape (seq_len, d)
    """
    # Your code here
    _, d = x.shape
    theta = 1 / base ** (2 * np.arange(d // 2) / d)
    theta = theta[None, :]
    x0 = x[:, ::2]
    x1 = x[:, 1::2]
    m = positions[:, None]     
    x0_new = x0 * np.cos(m * theta) - x1 * np.sin(m * theta)
    x1_new = x0 * np.sin(m * theta) + x1 * np.cos(m * theta)
    res = np.zeros_like(x)
    res[:, ::2] = x0_new
    res[:, 1::2] = x1_new
    return res 