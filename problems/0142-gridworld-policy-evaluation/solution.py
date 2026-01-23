import numpy as np

def gridworld_policy_evaluation(policy: dict, gamma: float, threshold: float) -> list[list[float]]:
    """
    Evaluate state-value function for a policy on a 5x5 gridworld.
    
    Args:
        policy: dict mapping (row, col) to action probability dicts
        gamma: discount factor
        threshold: convergence threshold
    Returns:
        5x5 list of floats
    """
    # Your code here
    v = np.zeros((5, 5))
    while True:
        v2 = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                s = (i, j)
                if s in {(0, 0), (0, 4), (4, 0), (4, 4)}:
                    continue
                p = policy[s]
                if i > 0:
                    v2[s] += p["up"] * (-1 + gamma * v[(i - 1, j)])
                else:
                    v2[s] += p["up"] * (-1 + gamma * v[s])
                if i < 4:
                    v2[s] += p["down"] * (-1 + gamma * v[(i + 1), j])
                else:
               