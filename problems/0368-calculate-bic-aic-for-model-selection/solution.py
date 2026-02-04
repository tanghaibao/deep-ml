import numpy as np

def calculate_aic_bic(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> tuple:
    """
    Calculate AIC and BIC for model selection.
    
    Args:
        y_true: True target values
        y_pred: Predicted values from the model
        k: Number of parameters in the model
    
    Returns:
        Tuple of (AIC, BIC)
    """
    rss = ((y_pred - y_true) ** 2).sum()
    n = len(y_true)
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    return round(float(aic), 4), round(float(bic), 4) 