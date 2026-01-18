import numpy as np
from itertools import groupby

def compute_roc_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute ROC curve points (FPR, TPR) for binary classification.
    
    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_scores: Predicted scores/probabilities for the positive class
    
    Returns:
        Tuple of (fpr, tpr) where each is a list of floats
    """
    # Your code here
    yy = list(zip(y_scores, y_true))
    yy.sort(reverse=True)
    res = [(0, 0)]
    y_true = np.array(y_true)
    t = (y_true == 1).sum()
    n = (y_true == 0).sum()
    tp, tn, fp, fn = 0, n, 0, t
    for _, yts in groupby(yy, key=lambda x: x[0]):
        for _, yt in yts:
            if yt == 1:
                tp += 1
                fn -= 1
            else:
                tn -= 1
                fp += 1
        tpr = round(tp / (tp + fn), 4)
        fpr = round(fp / (fp + tn), 4)
        res.append((fpr, tpr))
    return list(zip(*res))