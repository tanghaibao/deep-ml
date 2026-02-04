import numpy as np

def rlhf_weight_update(
    weights: np.ndarray,
    rewards: np.ndarray,
    policy_log_probs: np.ndarray,
    ref_log_probs: np.ndarray,
    log_prob_grads: np.ndarray,
    beta: float,
    lr: float
) -> np.ndarray:
    """
    Perform a single RLHF policy gradient weight update.
    
    Args:
        weights: Current model weights, shape (num_weights,)
        rewards: Rewards from reward model, shape (batch_size,)
        policy_log_probs: Log probs from current policy, shape (batch_size,)
        ref_log_probs: Log probs from reference model, shape (batch_size,)
        log_prob_grads: Gradient of log probs w.r.t. weights, shape (batch_size, num_weights)
        beta: KL penalty coefficient
        lr: Learning rate
    
    Returns:
        Updated weights as a numpy array, shape (num_weights,)
    """
    r = rewards - beta * (policy_log_probs - ref_log_probs)
    grad = (r[:, None] * log_prob_grads).mean(axis=0)
    return weights + lr * grad