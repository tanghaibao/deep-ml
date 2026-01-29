import numpy as np
from typing import List, Tuple

def sigmoid(a):
    return 1 / (1 + np.exp(-a))
    
def fit_bradley_terry(comparisons: List[Tuple[int, int]], n_items: int, 
                      learning_rate: float = 0.5, n_iterations: int = 100) -> np.ndarray:
    """
    Fit Bradley-Terry model parameters using maximum likelihood estimation.
    
    Args:
        comparisons: List of (winner_idx, loser_idx) tuples
        n_items: Total number of items to rank
        learning_rate: Step size for gradient ascent
        n_iterations: Number of optimization iterations
    
    Returns:
        np.ndarray: Estimated strength parameters of shape (n_items,)
    """
    # Your code here
    w = np.zeros(n_items)
    for _ in range(n_iterations):
        g = np.zeros(n_items)
        for a, b in comparisons:
            k = 1 - sigmoid(w[a] - w[b])
            g[a] += k
            g[b] -= k 
        w += learning_rate * g
    return w