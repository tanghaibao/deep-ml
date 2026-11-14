import numpy as np

def softmax(A):
    Ap = np.exp(A - A.max(axis=-1, keepdims=True))
    return Ap / Ap.sum(axis=-1, keepdims=True)

def simple_self_attention(X: list[list[float]]) -> list[list[float]]:
    """
    Compute context vectors using simple self-attention (no trainable weights).

    Args:
        X: Input embeddings of shape (T, d) as a list of lists.

    Returns:
        Context vectors of shape (T, d) as a list of lists.
    """
    X = np.array(X)
    A = X @ X.T
    A = softmax(A)
    return A @ X
