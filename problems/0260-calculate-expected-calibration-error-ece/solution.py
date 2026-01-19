import numpy as np

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Calculate the Expected Calibration Error (ECE).
    
    Args:
        y_true: List or array of true binary labels (0 or 1)
        y_prob: List or array of predicted probabilities for the positive class
        n_bins: Number of bins for grouping predictions (default: 10)
    
    Returns:
        float: ECE value rounded to 3 decimal places
    """
    # Your code here
    n = len(y_true)
    trues = np.zeros(n_bins)
    sum_probs = np.zeros(n_bins)
    bm = np.zeros(n_bins)
    for t, p in zip(y_true, y_prob):
        idx = int(np.ceil(p * n_bins) - 1) if p else 0
        trues[idx] += t
        sum_probs[idx] += p
        bm[idx] += 1
    ece = 0
    for b, true, sum_prob in zip(bm, trues, sum_probs):
        ece += abs(true - sum_prob) / n
    return np.round(ece, 3)