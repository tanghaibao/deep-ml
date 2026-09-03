import numpy as np

def row_normalize(counts: list[list[float]]) -> list[list[float]]:
    """Convert a count matrix into a row-stochastic probability matrix."""
    counts = np.array(counts)
    return (counts / (counts.sum(axis=-1, keepdims=True) + 1e-10)).tolist()