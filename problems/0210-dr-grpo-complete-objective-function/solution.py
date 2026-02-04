import numpy as np

def compute_dr_grpo_objective(log_probs_new: list[list[float]], 
                               log_probs_old: list[list[float]], 
                               rewards: list[float], 
                               epsilon: float = 0.2) -> float:
    """
    Compute the Dr. GRPO (GRPO Done Right) clipped objective.
    
    Args:
        log_probs_new: Log probabilities from new policy π_θ
                      Each response: [log π_θ(o_1|q), log π_θ(o_2|q,o_1), ...]
        log_probs_old: Log probabilities from old policy π_θ_old
        rewards: Rewards R(q, o_i) for each response
        epsilon: Clipping parameter for importance ratios
    
    Returns:
        Dr. GRPO objective value
    """
    # Your code here
    log_probs_new = np.array(log_probs_new)
    log_probs_old = np.array(log_probs_old)
    w = np.exp(log_probs_new - log_probs_old)  # (n_response, t)
    rewards = np.array(rewards)
    A = rewards - rewards.mean() # (n_response,)
    A = A[:, None