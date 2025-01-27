from collections import Counter

def phi_corr(x: list[int], y: list[int]) -> float:
	"""
	Calculate the Phi coefficient between two binary variables.

	Args:
	x (list[int]): A list of binary values (0 or 1).
	y (list[int]): A list of binary values (0 or 1).

	Returns:
	float: The Phi coefficient rounded to 4 decimal places.
	"""
	# Your code here
	c = Counter(zip(x, y))
	x00 = c[0, 0]
	x01 = c[0, 1]
	x10 = c[1, 0]
	x11 = c[1, 1]
	val = (x00 * x11 - x01 * x10) / ((x00 + x01) * (x10 + x11) * (x00 + x10) * (x01 + x11)) ** 0.5
	return round(val,4)