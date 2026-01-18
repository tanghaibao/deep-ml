import numpy as np
from itertools import product

def grid_search(X_train: np.ndarray, y_train: np.ndarray, 
                X_val: np.ndarray, y_val: np.ndarray,
                param_grid: dict, model_fn: callable, 
                scoring_fn: callable) -> tuple:
    """
    Perform grid search to find optimal hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training labels  
        X_val: Validation features
        y_val: Validation labels
        param_grid: Dict mapping parameter names to lists of values to try
        model_fn: Function(X_train, y_train, X_val, **params) -> predictions
        scoring_fn: Function(y_true, y_pred) -> score (higher is better)
    
    Returns:
        Tuple of (best_params dict, best_score rounded to 4 decimals)
    """
    best_params = {}
    best_score = -1
    names, values = zip(*param_grid.items())
    for value in product(*values):
        params = dict(zip(names, value))
        y_pred = model_fn(X_train