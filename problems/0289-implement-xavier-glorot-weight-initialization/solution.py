import numpy as np

def xavier_initialization(shape: tuple, mode: str = 'uniform', seed: int = None) -> np.ndarray:
    """
    Implement Xavier/Glorot weight initialization.
    
    Args:
        shape: Tuple of (fan_in, fan_out) representing weight matrix dimensions
        mode: 'uniform' or 'normal' initialization
        seed: Random seed for reproducibility (optional)
    
    Returns:
        Initialized weight matrix as numpy array
    """
    # Your code here
    if seed is not None:
        np.random.seed(seed)
    fan_in, fan_out = shape
    if mode == "uniform":
        ub = np.sqrt(6 / (fan_in + fan_out))
        lb = -ub
        return np.random.uniform(lb, ub, size=shape)
    if mode == "normal":
        sigma = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, sigma, size=shape)