import math

def tanh(x: float) -> float:
	"""
	Implements the Tanh (hyperbolic tangent) activation function.

	Args:
		x (float): Input value

	Returns:
		float: The tanh of the input, rounded to 4 decimal places
	"""
	# Your code here
	a = math.exp(x)
    b = math.exp(-x)
    return round((a - b) / (a + b), 4)