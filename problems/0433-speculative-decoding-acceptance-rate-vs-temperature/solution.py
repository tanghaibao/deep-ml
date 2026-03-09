import numpy as np

def ts(probs, T):
    ex = np.exp((probs - probs.max())/ T) 
    return ex / ex.sum()


def acceptance_rate_vs_temperature(draft_logits: np.ndarray, target_logits: np.ndarray, temperatures: np.ndarray) -> list:
    """
    Compute speculative decoding expected acceptance rate at various temperatures.
    
    Args:
        draft_logits: Logits from draft model, shape (vocab_size,)
        target_logits: Logits from target model, shape (vocab_size,)
        temperatures: Array of temperature values to evaluate
    
    Returns:
        List of acceptance rates (floats rounded to 4 decimal places)
    """
    # Your code here
    res = []
    for T in temperatures:
        pt = ts(draft_logits, T)
        tt = ts(target_logits, T)
        alpha = np.minimum(pt, tt).sum()
        res.append(alpha)
    return [round(float(x), 4) for x in res]