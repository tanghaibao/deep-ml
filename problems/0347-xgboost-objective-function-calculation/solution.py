import numpy as np

def xgboost_objective(gradients: np.ndarray, hessians: np.ndarray,
                      left_indices: np.ndarray, right_indices: np.ndarray,
                      lambda_reg: float = 1.0, gamma: float = 0.0) -> dict:
    """
    Calculate XGBoost objective function components for a potential split.
    
    Args:
        gradients: First-order gradients for each sample
        hessians: Second-order hessians for each sample
        left_indices: Indices of samples going to left child
        right_indices: Indices of samples going to right child
        lambda_reg: L2 regularization parameter
        gamma: Tree complexity penalty
        
    Returns:
        Dictionary with 'left_weight', 'right_weight', and 'gain'
    """
    Gl = gradients[left_indices].sum()
    Gr = gradients[right_indices].sum()
    Hl = hessians[left_indices].sum()
    Hr = hessians[right_indices].sum()
    left_weight = -Gl / (Hl + lambda_reg)
    right_weight = -Gr / (Hr + lambda_reg)
   