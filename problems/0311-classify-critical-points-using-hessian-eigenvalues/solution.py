import numpy as np

def classify_critical_point(hessian: np.ndarray, tol: float = 1e-10) -> str:
    """
    Classify a critical point based on its Hessian matrix.
    
    Args:
        hessian: Square Hessian matrix (n x n) as numpy array
        tol: Tolerance for considering eigenvalues as zero
        
    Returns:
        Classification string: 'local_minimum', 'local_maximum', 
        'saddle_point', or 'inconclusive'
    """
    evals, evecs = np.linalg.eigh(hessian)
    if np.all(evals > 0):
        return "local_minimum"
    if np.all(evals < 0):
        return "local_maximum"
    if np.any(evals == 0):
        return "inconclusive"
    return "saddle_point"