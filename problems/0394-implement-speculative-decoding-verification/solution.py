import numpy as np

def speculative_decode_verify(draft_tokens: list, draft_probs: list, target_probs: list, coin_flips: list, resample_coin: float) -> list:
    """
    Verify draft tokens using speculative decoding.
    
    Args:
        draft_tokens: List of K drafted token indices
        draft_probs: K x V array, draft model distributions at each position
        target_probs: K x V array, target model distributions at each position
        coin_flips: K random values in [0,1) for acceptance decisions
        resample_coin: Random value in [0,1) for resampling on rejection
    
    Returns:
        List of accepted/resampled token indices
    """
    draft_probs = np.array(draft_probs)
    target_probs = np.array(target_probs)
    K, _ = draft_probs.shape
    r = target_probs / draft_probs
    t = np.minimum(1, r[np.arange(K), draft_tokens])
    coin_flips = np.array(coin_flips)
    res = []
    for i in range(K):
        if coin_flips[i] < t[i]:
            res.append(draft_toke