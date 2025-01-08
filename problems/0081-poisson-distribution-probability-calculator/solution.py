import math

def poisson_probability(k, lam):
	"""
	Calculate the probability of observing exactly k events in a fixed interval,
	given the mean rate of events lam, using the Poisson distribution formula.
	:param k: Number of events (non-negative integer)
	:param lam: The average rate (mean) of occurrences in a fixed interval
	"""
	prod = 1
	for i in range(2, k + 1):
		prod *= i
	val = lam ** k * math.exp(-lam) / prod
	return round(val,5)