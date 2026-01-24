import numpy as np
def q_learning(num_states, num_actions, P, R, terminal_states, alpha, gamma, epsilon, num_episodes):
    # Your code here
    Q = np.zeros((num_states, num_actions))
    non_terminal_states = set(range(num_states)) - set(terminal_states)
    for _ in range(num_episodes):
        state = np.random.choice(list(non_terminal_states))
        while state not in terminal_states:
            if np.random.random() < epsilon:
                action = np.random.choice(num_actions)
            else:
                action = np.argmax(Q[state])
            next_state = np.random.choice(num_states, p=P[state][action])
            Q[state, action] += alpha * (
                R[state, action] + gamma * Q[next_state].max() - Q[state, action]
            )
            state = next_state
    return Q