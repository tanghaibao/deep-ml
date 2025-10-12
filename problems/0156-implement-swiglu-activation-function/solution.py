import numpy as np

def swish(x) -> np.ndarray:
    return x / (1 + np.exp(-x))

def chunk(x) -> (np.ndarray, np.ndarray):
    _, b = x.shape
    b //= 2
    return x[:, :b], x[:, b:] 

def SwiGLU(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x: np.ndarray of shape (batch_size, 2d)
    Returns:
        np.ndarray of shape (batch_size, d)
    """
    # Your code here
    x1, x2 = chunk(x)
    scores = x1 * swish(x2)
    return scores