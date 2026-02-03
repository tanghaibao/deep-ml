import numpy as np

def gradient_boosting_step(X, y, current_predictions, learning_rate=0.1):
    """
    Perform one step of gradient boosting regression using a decision stump.
    
    Args:
        X: Feature matrix (list of lists), shape (n_samples, n_features)
        y: Target values (list), shape (n_samples,)
        current_predictions: Current ensemble predictions (list), shape (n_samples,)
        learning_rate: Learning rate for the update (default 0.1)
    
    Returns:
        List of updated predictions rounded to 4 decimal places
    """
    y = np.array(y, dtype=float)
    X = np.array(X, dtype=float)
    current_predictions = np.array(current_predictions, dtype=float)
    res = y - current_predictions
    m, n = X.shape
    best_mse, best_h = np.inf, np.zeros_like(current_predictions)
    for j in range(n):
        x = np.sort(np.unique(X[:, j]))
        for k in range(len(x) - 1):
            thresh = (x[k] + x[k + 1]) / 2
            is_left = x <= thresh
          