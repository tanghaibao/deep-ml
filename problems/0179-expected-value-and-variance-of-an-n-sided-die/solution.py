def dice_statistics(n: int) -> tuple[float, float]:
	"""
	Compute the expected value and variance of a fair n-sided die roll.

	Args:
		n (int): Number of sides of the die

	Returns:
		tuple: (expected_value, variance)
	"""
	# Your code here
	Ex = (n + 1) / 2
    varx = (n + 1) * (2 * n + 1) / 6 - ((n + 1) / 2) ** 2
    return Ex, varx