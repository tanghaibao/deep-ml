import math

def binomial_probability(n, k, p):
	"""
    Calculate the probability of achieving exactly k successes in n independent Bernoulli trials,
    each with probability p of success, using the Binomial distribution formula.
    """
	# Your code here
	prod = 1
	for i in range(n - k + 1, n + 1):
		prod *= i
	for i in range(2, k + 1):
		prod /= i
	prod *= p ** k
	prod *= (1 - p) ** (n - k)
	return round(prod, 5)