import numpy as np

def sigmoid(x):
    """
    Calculates the logistic sigmoid function using NumPy.
    
    Args:
        x: Input value, vector, or matrix (NumPy array).

    Returns:
        The sigmoid of the input, as a NumPy array.
    """
    return 1 / (1 + np.exp(-x))

def gru_cell(x: np.ndarray, h_prev: np.ndarray,
             W_z: np.ndarray, U_z: np.ndarray, b_z: np.ndarray,
             W_r: np.ndarray, U_r: np.ndarray, b_r: np.ndarray,
             W_h: np.ndarray, U_h: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """
    Implements a single GRU cell forward pass.
    
    Args:
        x: Input vector of shape (input_size,)
        h_prev: Previous hidden state of shape (hidden_size,)
        W_z, W_r, W_h: Weight matrices for input
        U_z, U_r, U_h: Weight matrices for hidden state
        b_z, b_r, b_h: Bias vectors
    
    Returns:
        h_next: New hidden state of shape (hidden_size,)
    """
    # Your code here
    z = sigmoid(W_z @ x + U_z @ h_prev +