import numpy as np


def dense_backward(a_prev: np.ndarray, W: np.ndarray, b: np.ndarray, y: np.ndarray) -> dict:
    """
    Compute gradients of the squared-error cost w.r.t. W, b, and a_prev
    for a dense layer z = W @ a_prev + b followed by sigmoid activation.

    Args:
        a_prev: activations from previous layer, shape (K,)
        W: weight matrix, shape (J, K)
        b: bias vector, shape (J,)
        y: target vector, shape (J,)

    Returns:
        dict with keys 'dW' (J x K nested list), 'db' (list of length J),
        and 'da_prev' (list of length K).
    """
    # Forward pass
    z = W @ a_prev + b
    a = 1.0 / (1.0 + np.exp(-z))

    # C = sum_j (a_j - y_j)^2
    # dC/da = 2(a - y)
    # da/dz = a(1-a)
    dz = 2.0 * (a - y) * a * (1.0 - a)

    # dz_j/dW_jk = a_prev_k
    dW = np.outer(dz, a_prev)

    # dz_j/db_j = 1
    db = dz

    # Each a_prev[k] contributes to every output neuron:
    # dC/da_prev[k] = sum_j W[j,k] * dz[j]
    da_prev = W.T @ dz

    return {
        "dW": dW.tolist(),
        "db": db.tolist(),
        "da_prev": da_prev.tolist(),
    }