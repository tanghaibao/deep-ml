import numpy as np

def bellman_update(V, transitions, gamma):
    """
    Perform one step of value iteration using the Bellman equation.
    Args:
      V: np.ndarray, state values, shape (n_states,)
      transitions: list of dicts. transitions[s][a] is a list of (prob, next_state, reward, done)
      gamma: float, discount factor
    Returns:
      np.ndarray, updated state values
    """
    # TODO: Implement Bellman update
    nV = np.zeros_like(V)
    (n_states,) = V.shape
    for s in range(n_states):
        max_value = float("-inf")
        for a, trans_list in transitions[s].items():
            action_value = 0.0
            for prob, next_state, reward, done in trans_list:
                action_value += prob * (reward + gamma * V[next_state])
            max_value = max(max_value, action_value)
        nV[s] = max_value
    return np.round(nV, 4)