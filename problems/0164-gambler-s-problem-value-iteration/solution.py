def gambler_value_iteration(ph, theta=1e-9):
    """
    Computes the optimal value function and policy for the Gambler's Problem.
    Args:
      ph: probability of heads
      theta: convergence threshold
    Returns:
      V: list of values for all states 0..100
      policy: list of optimal stakes for all states 0..100
    """
    # Your code here
    V = [-1] * 101
    V[0] = V[100] = 0
    policy = [-1] * 101
    while True:
        delta = 0
        for s in range(1, 100):
            v = V[s]
            max_v = -1
            for a in range(1, min(s, 100 - s) + 1):
                reward = s + a == 100
                value = ph * (reward + V[s + a]) + (1 - ph) * V[s - a]
                if value > max_v:
                    max_v = value
                    policy[s] = a
            V[s] = max_v
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V, policy