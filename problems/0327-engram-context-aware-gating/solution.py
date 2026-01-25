import numpy as np

def rmsnorm(X, eps=1e-12):
    norm = (X * X).mean()
    return X / np.sqrt(norm + eps)

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def engram_context_gating(h: np.ndarray, e: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Implement Engram context-aware gating mechanism.
    
    Args:
        h: Hidden states of shape (T, d)
        e: Retrieved memory embeddings of shape (T, d_mem)
        W_K: Key projection matrix of shape (d_mem, d)
        W_V: Value projection matrix of shape (d_mem, d)
        eps: Small constant for numerical stability in RMSNorm
    
    Returns:
        Gated output of shape (T, d)
    """
    T, d = h.shape
    k = e @ W_K  # (T, d)
    v = e @ W_V  # (T, d)
    gated = np.zeros_like(v)
    for t in range(T):
        alpha = sigmoid(rmsnorm(h[t], eps).T @ rmsnorm(k[t], eps) / np.sqrt(d))
        gated[t] = alpha * v[t]
    return gated

