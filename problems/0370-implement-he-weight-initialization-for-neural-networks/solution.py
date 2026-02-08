import numpy as np

def he_initialize(layer_dims: list, method: str = 'normal', seed: int = 42) -> list:
    """
    Initialize weight matrices for a neural network using He initialization.
    
    Args:
        layer_dims: List of integers representing neurons per layer
        method: 'normal' or 'uniform' sampling distribution
        seed: Random seed for reproducibility
    
    Returns:
        List of numpy arrays, one weight matrix per adjacent layer pair
    """
    np.random.seed(seed)
    def he(fin, fout):
        if method == "normal":
            return np.random.standard_normal((fin, fout)) * np.sqrt(2 / fin)
        return np.random.uniform(-1, 1, (fin, fout)) * np.sqrt(6 / fin)

    weights = []
    for i in range(len(layer_dims) - 1):
        w = he(layer_dims[i], layer_dims[i + 1])
        weights.append(w)
    return weights