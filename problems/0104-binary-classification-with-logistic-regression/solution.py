import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
	return 1 / (1 + np.exp(-z))

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
	"""
	Implements binary classification prediction using Logistic Regression.

	Args:
		X: Input feature matrix (shape: N x D)
		weights: Model weights (shape: D)
		bias: Model bias

	Returns:
		Binary predictions (0 or 1)
	"""
	# Your code here
	z = X @ weights + bias
	y = sigmoid(z)
	return (y >= 0.5).astype(int)