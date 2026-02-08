# Cell 1
import numpy as np

def xavier_init(fan_in: int, fan_out: int, mode: str = 'uniform', seed: int = 42) -> dict:
    """
    Perform Xavier/Glorot weight initialization.

    Args:
        fan_in (int): Number of input units.
        fan_out (int): Number of output units.
        mode (str): 'uniform' or 'normal'.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Contains 'weights' (nested list), 'shape' (list), and 'param' (float).
    """
    # Your code here
    np.random.seed(seed)
    if mode == "uniform":
        param = np.sqrt(6 / (fan_in + fan_out))
        weights = np.random.random((fan_in, fan_out)) * param
    elif mode == "normal":
        param = np.sqrt(2 / (fan_in + fan_out))
        weights = np.random.standard_normal((fan_in, fan_out)) * param
    else:
        raise ValueError("mode must be 'uniform' or 'normal'")
    weights = weights.round(4)
    return {
        "weights": weights.tolist(),
        "shape": list(weights.shape