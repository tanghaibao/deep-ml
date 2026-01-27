import numpy as np

def first_visit_mc_prediction(
    episodes: list[list[tuple[int, float]]],
    n_states: int,
    gamma: float
) -> np.ndarray:
    """
    Estimate state values using First-Visit Monte Carlo prediction.
    
    Args:
        episodes: List of episodes. Each episode is a list of (state, reward) tuples.
                 The reward at index i is the reward received AFTER leaving state i.
        n_states: Number of states (states are integers 0 to n_states-1)
        gamma: Discount factor
        
    Returns:
        V: Estimated state values as numpy array of shape (n_states,)
    """
    # Your code here
    V = np.zeros(n_states)
    N = np.zeros(n_states)
    for episode in episodes:
        states, rewards = zip(*episode)
        seen = set()
        for t, state in enumerate(states):
            if state in seen:
                continue
            seen.add(state)
            returns = rewards[t:]
            discounts = np.array([gamma**i for i in range(le