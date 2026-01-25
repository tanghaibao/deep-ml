import numpy as np

def n_step_td_prediction(
    episodes: list[list[tuple[int, float]]],
    n_states: int,
    n: int,
    gamma: float,
    alpha: float
) -> np.ndarray:
    """
    Perform n-step TD prediction to estimate state values.
    
    Args:
        episodes: List of episodes. Each episode is a list of (state, reward) tuples.
                 The reward at index i is the reward received AFTER leaving state i.
                 Episodes end with a terminal transition (last state's reward is the final reward).
        n_states: Number of states (states are integers 0 to n_states-1)
        n: Number of steps for n-step TD
        gamma: Discount factor
        alpha: Learning rate
        
    Returns:
        V: Estimated state values as numpy array of shape (n_states,)
    """
    # Your code here
    V = np.zeros(n_states)
    for episode in episodes:
        states, rewards = zip(*episode)
        T = len(states)
        for t in range(T + n):
            tau = t - n + 1