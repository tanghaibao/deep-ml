import numpy as np

def square_relu(x: np.ndarray) -> dict:
	"""
	Apply the Square ReLU activation function and compute its derivative.
	
	Args:
		x: Input numpy array of any shape
	
	Returns:
		Dictionary with 'output' and 'derivative' as numpy arrays
	"""
	output = np.maximum(0, x) ** 2
	derivative = np.maximum(0, 2 * x)
	return {
		"output": np.round(output, 4), "derivative": np.round(derivative, 4),
	}