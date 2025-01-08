import numpy as np

def softmax(x):
    # Implement the softmax function
    e_x = np.exp(x - np.max(x, axis=0).reshape(-1, 1))  # Subtracting max for numerical stability
    return e_x / e_x.sum(axis=1).reshape(-1, 1)


def pattern_weaver(n, crystal_values, dimension):
    # Your code here
    crystal_values = np.array(crystal_values).reshape(-1, 1)
    attention = crystal_values @ crystal_values.T / (dimension**0.5)
    x = softmax(attention) @ crystal_values
    return np.round(x, 3)