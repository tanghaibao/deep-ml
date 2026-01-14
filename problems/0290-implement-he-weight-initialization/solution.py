import numpy as np

def he_initialization(n_in: int, n_out: int, mode: str = 'fan_in', distribution: str = 'normal', seed: int = None) -> np.ndarray:
    """
    Implement He (Kaiming) weight initialization.
    
    Parameters:
    n_in: number of input units
    n_out: number of output units
    mode: 'fan_in' or 'fan_out'
    distribution: 'normal' or 'uniform'
    seed: random seed for reproducibility
    
    Returns:
    numpy array of shape (n_in, n_out) with He-initialized weights
    """
    np.random.seed(seed)
    fan = n_in if mode == "fan_in" else n_out
    size = (n_in, n_out)
    if distribution == "normal":
        res = np.random.normal(0, np.sqrt(2 / fan), size=size)
    else:
        bb = np.sqrt(6 / fan)
        res = np.random.uniform(-bb, bb, size=size)
    return res