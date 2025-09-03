import numpy as np
import math

def GeLU(x: np.ndarray) -> np.ndarray:
	# Your code here
	return 0.5 * x * (1 + np.tanh(np.sqrt(2. / math.pi) * (x + 0.044715 * x ** 3)))