import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def backprop_chain(a0: float, y: float, w: list, b: list) -> dict:
    """
    Compute gradients of C = (a_L - y)^2 with respect to each weight and bias
    in a chain network where every layer has a single sigmoid neuron.

    Args:
        a0: input activation (scalar)
        y: target value (scalar)
        w: list of L weights
        b: list of L biases

    Returns:
        Dictionary with keys 'dW' and 'dB', each a list of L floats rounded to 6 places.
    """
    # Your code here
    L = len(w)
    z = np.zeros(L)
    a = np.zeros(L)
    dW = np.zeros(L)
    dB = np.zeros(L)
    for l in range(L):
        z[l] = w[l] * (a0 if l == 0 else a[l - 1]) + b[l]
        a[l] = sigmoid(z[l])
    delta = 2 * (a[L - 1] - y)
    for l in range(L - 1, -1, -1):
        dz = delta * a[l] * (1 - a[l])
        dW[l] = dz * (a0 if l == 0 else a[l - 1])
        dB[l] = dz
        delta = dz * w[l]
    return {
        "dW": dW.round(6).tolist(), "dB": dB.round(6).tolist()
    }
