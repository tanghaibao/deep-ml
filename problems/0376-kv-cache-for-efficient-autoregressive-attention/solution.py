import numpy as np

def softmax(s):
    es = np.exp(s - s.max())
    return es / es.sum()

def kv_cache_attention_step(x_new: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) or None if first step
    
    Returns:
        Tuple (output, updated_cache)
    """
    q = x_new @ W_Q # (d_k,)
    k = x_new @ W_K # (d_k,)
    v = x_new @ W_V # (d_v,)
    if cache:
        K_cache, V_cache = cache
        K = np.concatenate([K_cache, k[None, :]]) # (t, d_k)
        V = np.concatenate([V_cache, v[None, :]]) # (t, d_v)
    else:
        K = k[None, :]
        V = v[None, :]
    scores = q @ K.T / np.sqrt(len(q)) # (t,)