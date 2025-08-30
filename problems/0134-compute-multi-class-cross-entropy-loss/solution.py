import numpy as np

def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon = 1e-15) -> float:
    if np.all(predicted_probs == true_labels): return 0.0
    # Your code here
    return -(np.log(predicted_probs + epsilon) * true_labels).sum(axis=1).mean()