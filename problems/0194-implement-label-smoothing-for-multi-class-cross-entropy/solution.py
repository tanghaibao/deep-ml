import numpy as np


def smooth_labels(y_true, num_classes, epsilon):
    """
    Create smoothed one-hot target vectors.

    Args:
        y_true: Iterable[int] of shape (N,) with values in [0, K-1]
        num_classes: int, total number of classes (K)
        epsilon: float in [0, 1]

    Returns:
        np.ndarray of shape (N, K) with smoothed probabilities.
    """
    # Your implementation here
    N = len(y_true)
    K = num_classes
    one_hot = np.ones((N, K)) * epsilon / K
    one_hot[np.arange(N), y_true] += 1 - epsilon
    return one_hot


def label_smoothing_cross_entropy(
    logits, y_true, num_classes, epsilon=0.1, round_decimals=None
):
    """
    Compute mean cross-entropy between logits and smoothed targets using stable log-softmax.

    Args:
        logits: Array-like of shape (N, K), model output scores.
        y_true: Array-like of shape (N,), integer class indices.
        num_classes: int, number of classes (K).
        epsilon: float in [0, 1].
        roun