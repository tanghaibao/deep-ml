import numpy as np

def global_avg_pool(x: np.ndarray) -> np.ndarray:
	# Your code here
	return np.average(x, axis=(0, 1))