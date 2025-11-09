import numpy as np

def compute_confusion_matrix(y_true, y_pred, num_classes, normalize=None, round_decimals=4):
    """
    Compute a KxK confusion matrix with optional normalization.

    Args:
        y_true: Iterable of true labels in [0, K-1]
        y_pred: Iterable of predicted labels in [0, K-1]
        num_classes: K, number of classes
        normalize: None | 'true' | 'pred' | 'all'
        round_decimals: decimals to round when normalization is applied

    Returns:
        list[list[int|float]] confusion matrix
    """
    # Your implementation here
    res = np.zeros((num_classes, num_classes))
    for a, b in zip(y_true, y_pred):
        res[a, b] += 1
    if normalize == "true":
        res /= res.sum(axis=1, keepdims=True)
    elif normalize == "pred":
        res /= res.sum(axis=0)
    elif normalize == "all":
        res /= res.sum()
    return np.round(res, round_decimals)
