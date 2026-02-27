import numpy as np

def speculative_decode(draft_probs: list, target_probs: list, draft_tokens: list, accept_rand: list, sample_rand: list) -> list:
    """
    Simulate speculative decoding end-to-end.
    
    Args:
        draft_probs: K arrays of shape (vocab_size,) - draft model probability distributions
        target_probs: K+1 arrays of shape (vocab_size,) - target model probability distributions
        draft_tokens: K integers - token indices proposed by draft model
        accept_rand: K floats in [0,1) - random values for acceptance decisions
        sample_rand: K+1 floats in [0,1) - random values for token sampling
    
    Returns:
        List of token indices produced by speculative decoding
    """
    K = len(draft_probs)
    out_tokens = []
    for i in range(K):
        t = draft_tokens[i]
        alpha_i = min(1, target_probs[i][t] / draft_probs[i][t])
        if accept_rand[i] < alpha_i:
            out_tokens.append(t)
        else:
            diff = np.maximum