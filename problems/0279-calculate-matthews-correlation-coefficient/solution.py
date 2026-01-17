import numpy as np

def matthews_correlation_coefficient(y_true: list, y_pred: list) -> float:
    """
    Calculate the Matthews Correlation Coefficient for binary classification.
    
    Args:
        y_true: List of actual binary labels (0 or 1)
        y_pred: List of predicted binary labels (0 or 1)
    
    Returns:
        MCC value rounded to 4 decimal places
    """
    y_true = np.array(y_true, dtype=bool)
    y_pred = np.array(y_pred, dtype=bool)
    tp = (y_pred & y_true).sum()
    fp = (y_pred & ~y_true).sum()
    tn = (~y_pred & ~y_true).sum()
    fn = (~y_pred & y_true).sum()
    mcc = (tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-10)
    return round(float(mcc), 4)