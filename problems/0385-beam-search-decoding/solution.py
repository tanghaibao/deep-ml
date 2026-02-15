import numpy as np

def beam_search_decode(log_probs: np.ndarray, beam_width: int, top_k: int = 1) -> list:
    """
    Perform beam search decoding over a sequence of log-probability distributions.
    
    Args:
        log_probs: numpy array of shape (T, V) with log-probabilities at each step
        beam_width: number of beams to maintain at each step
        top_k: number of top sequences to return
    
    Returns:
        List of (sequence, score) tuples sorted by score descending
    """
    # Your code here
    T, _ = log_probs.shape
    B = [([], 0)]
    for i in range(T):
        top_k_idx = np.argsort(-log_probs[i])[:top_k].tolist()
        new_B = []
        for b, bp in B:
            for j in top_k_idx:
                new_B.append((b + [j], float(bp + log_probs[i, j])))
        new_B.sort(key=lambda x: x[-1], reverse=True)
        B = new_B[:beam_width]
    return B[:min(beam_width, top_k)]