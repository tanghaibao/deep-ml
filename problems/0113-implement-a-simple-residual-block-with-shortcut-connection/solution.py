import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
	# Your code here
	return relu(w2 @ relu(w1 @ x) + x)