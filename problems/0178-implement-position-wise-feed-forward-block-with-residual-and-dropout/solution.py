import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def dropout(x: np.ndarray, p: float = 0.1, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    if p == 0.0:
        return x
    mask = np.random.random(*x.shape) >= p
    return x * mask / (1 - p)


def ffn(
    x: list[float],
    W1: list[list[float]],
    b1: list[float],
    W2: list[list[float]],
    b2: list[float],
    dropout_p: float = 0.1,
    seed: int = 42,
) -> list[float]:
    """
    Implement a position-wise feed-forward block with residual and dropout.

    Args:
        x: input vector
        W1, b1: first linear layer parameters
        W2, b2: second linear layer parameters
        dropout_p: dropout probability
        seed: random seed for reproducibility

    Returns:
        Output vector after FFN block (rounded to 4 decimals)
    """
    # Convert input lists to numpy arrays
    x = np.array(x)
    W1 = np.array(W1)
    b1 = np.array(b1)
    W2 = np.array(W2)
  