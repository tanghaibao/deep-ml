import numpy as np

def td_lambda_prediction(
    episodes: list[list[tuple[int, float]]],
    n_states: int,
    gamma: float,
    lambd: float,
    alpha: float
) -> np.ndarray:
    """
    Estimate state values using TD(λ) with accumulating eligibility traces.
    
    Args:
        episodes: List of episodes. Each episode is a list of (state, reward) tuples.
                 The reward at index i is the reward received AFTER leaving state i.
        n_states: Number of states (states are integers 0 to n_states-1)
        gamma: Discount factor
        lambd: Trace decay parameter (λ). Use 'lambd' to avoid Python keyword.
        alpha: Learning rate
        
    Returns:
        V: Estimated state values as numpy array of shape (n_states,)
    """
    # Your code here
    V = np.zeros(n_states)
    for episode in episodes:
        e = np.zeros(n_states)
        T = len(episode)
        for i in range(len(episode)):
            state, reward = episode[i]
            if i < T - 1:
  