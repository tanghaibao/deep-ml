import numpy as np

def softplus(H: np.ndarray) -> np.ndarray:
    return np.log(1 + np.exp(H))

def is_topk(a, k=1):
    _, rix = np.unique(-a, return_inverse=True)
    return (rix < k).reshape(a.shape)

def topk(H: np.ndarray, k: int) -> np.ndarray:
    H[~is_topk(H, k)] = -np.inf
    return H

def softmax(x):
    """
    Computes the softmax of a 1D NumPy array or along a specified axis of a multi-dimensional array.

    Args:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Array with softmax probabilities.
    """
    # Subtract the maximum value for numerical stability
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def noisy_topk_gating(
    X: np.ndarray,
    W_g: np.ndarray,
    W_noise: np.ndarray,
    N: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Args:
        X: Input data, shape (batch_size, features)
        W_g: Gating weight matrix, shape (features, num_experts)
        W_noise: Noise weight matrix, shape 