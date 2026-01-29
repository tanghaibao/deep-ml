import numpy as np

def grpo_objective(rhos, A, pi_theta_old, pi_theta_ref, epsilon=0.2, beta=0.01) -> float:
    """
    Compute the GRPO objective function.

    Args:
        rhos: List of likelihood ratios (pi_theta / pi_theta_old).
        A: List of advantage estimates.
        pi_theta_old: List of old policy probabilities.
        pi_theta_ref: List of reference policy probabilities.
        epsilon: Clipping parameter for the surrogate objective.
        beta: KL divergence penalty coefficient.

    Returns:
        The computed GRPO objective value.
    """
    # Your code here
    rhos = np.array(rhos)
    A = np.array(A)
    pi_theta_old = np.array(pi_theta_old)
    pi_theta_ref = np.array(pi_theta_ref)
    pi_theta = rhos * pi_theta_old
    ratio = pi_theta_ref / (pi_theta + 1e-12)
    D_KL = rhos * (ratio - np.log(ratio) - 1)
    S = np.minimum(rhos * A, np.clip(rhos, 1 - epsilon, 1 + epsilon) * A) - beta * D_KL
    return S.mean()