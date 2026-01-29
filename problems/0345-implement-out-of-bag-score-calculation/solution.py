import numpy as np
from collections import Counter

def calculate_oob_score(n_samples: int, bootstrap_indices: list, predictions: list, y_true: list) -> float:
    """
    Calculate the Out-of-Bag score for a bagging ensemble.
    
    Args:
        n_samples: Total number of samples in the dataset
        bootstrap_indices: List of lists containing indices used to train each estimator
        predictions: List of lists containing predictions from each estimator for all samples
        y_true: True labels for all samples
    
    Returns:
        OOB accuracy score as a float
    """
    oobs = []
    for i in range(n_samples):
        y_hat = []
        for b, preds in zip(bootstrap_indices, predictions):
            if i in b:
                continue
            y_hat.append(preds[i])
        if y_hat:
            y_mode, _ = Counter(y_hat).most_common(1)[0]
            oobs.append(y_mode == y_true[i])
    if not oobs:
        return 0.
    return np.round(np.mean(oobs), 4)