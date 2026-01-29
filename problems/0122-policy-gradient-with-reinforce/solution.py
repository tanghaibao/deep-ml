import numpy as np

def softmax(x):
    x -= np.max(x)
    ex = np.exp(x)
    return ex / (ex.sum() + 1e-12) 

def compute_policy_gradient(
    theta: np.ndarray, episodes: list[list[tuple[int, int, float]]]
) -> np.ndarray:
    """
    Estimate the policy gradient using REINFORCE.

    Args:
        theta: (num_states x num_actions) policy parameters.
        episodes: List of episodes, where each episode is a list of (state, action, reward).

    Returns:
        Average policy gradient (same shape as theta).
    """
    # Your code here
    grads = []
    for episode in episodes:
        states, actions, rewards = zip(*episode)
        G = np.cumsum(rewards[::-1])[::-1]
        grad = np.zeros_like(theta)
        for t in range(len(episode)):
            state = states[t]
            action = actions[t]
            pi = softmax(theta[state])
            grad[state] += -pi * G[t]
            grad[state, action] += 1 * G[t]
        grads.append(grad)
    return np.mean(grads, axis=0)