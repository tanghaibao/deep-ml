import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
	"""
	Perform Layer Normalization.
	"""
	# Your code here
	X_hat = (X - X.mean(axis=-1, keepdims=True)) / np.sqrt(X.var(axis=-1, keepdims=True) + epsilon)
	return gamma * X_hat + beta