import numpy as np

def compute_norm(arr: np.ndarray, norm_type: str) -> float:
    """
    Compute the specified norm of the input array.
    
    Args:
        arr: Input numpy array (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', or 'frobenius')
    
    Returns:
        The computed norm as a float
    """
    arr = arr.astype(float)
    # Your code here
    if norm_type == "l1":
        return np.abs(arr).sum()
    if norm_type == "l2":
        return np.linalg.norm(arr)
    return np.linalg.norm(arr.flatten())