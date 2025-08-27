import numpy as np
def simulate_markov_chain(transition_matrix, initial_state, num_steps):
    # Your code here
    n, _ = transition_matrix.shape
    state = initial_state
    res = [state]
    for _ in range(num_steps):
        row = transition_matrix[state]
        new_state = np.random.choice(range(n), p=row)
        res.append(new_state)
        state = new_state
    return np.array(res)