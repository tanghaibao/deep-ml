import numpy as np

def softmax(x):
    ex = np.exp(x - x.max())
    return ex / ex.sum()

def top_p_sampling(logits: list[float], p: float) -> list[float]:
    """
    Apply top-p (nucleus) sampling to filter a probability distribution.
    
    Args:
        logits: Raw unnormalized scores for each token
        p: Cumulative probability threshold (0 < p <= 1)
    
    Returns:
        Filtered and renormalized probability distribution as a list of floats
    """
    logits = np.array(logits)
    s = softmax(logits)
    idx = np.argsort(-s)
    si = np.cumsum(s[idx])
    k = (si >= p).sum() - 1
    if k:
        s[idx[-k:]] = 0
    s /= s.sum()
    return s.round(4).tolist()