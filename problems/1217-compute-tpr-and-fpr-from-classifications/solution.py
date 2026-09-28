import numpy as np

def compute_tpr_fpr(y_true, y_pred):
    """
    Compute TPR and FPR from true and predicted binary labels.

    Args:
        y_true (array-like): Ground-truth labels (0 or 1).
        y_pred (array-like): Predicted labels (0 or 1).

    Returns:
        tuple: (tpr, fpr) as Python floats.
    """
    y_true = y_true.astype(np.bool)
    y_pred = y_pred.astype(np.bool)
    # TODO: compute TP, FN, FP, TN then TPR and FPR
    TP = (y_true & y_pred).sum()
    FN = (y_true & ~y_pred).sum()
    FP = (~y_true & y_pred).sum()
    TN = (~y_true & ~y_pred).sum()
    return TP/(TP+FN) if TP+FN else 0, FP/(FP+TN) if FP+TN else 0
