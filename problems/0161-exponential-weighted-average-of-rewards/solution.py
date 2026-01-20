def exp_weighted_average(Q1, rewards, alpha):
    """
    Q1: float, initial estimate
    rewards: list or array of rewards, R_1 to R_k
    alpha: float, step size (0 < alpha <= 1)
    Returns: float, exponentially weighted average after k rewards
    """
    # Your code here
    k = len(rewards)
    ans = (1 - alpha) ** k * Q1
    for i in range(1, k + 1):
        ans += alpha * (1 - alpha) ** (k - i) * rewards[i - 1]
    return ans