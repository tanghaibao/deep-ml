import numpy as np

def convert_range(values: np.ndarray, c: float, d: float) -> np.ndarray:
    """
    Shift and scale values from their original range [min, max] to a target [c, d] range.
    """
    # Your code here
    a = values.min()
    b = values.max()
    return (d - c) * (values - a) / (b - a) + c