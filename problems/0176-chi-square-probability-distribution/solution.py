import math

def chi_square_probability(x, k):
    """
    Calculate the probability density of x in a Chi-square distribution
    with k degrees of freedom.
    """
    probability = (1 / (2**(k/2) * math.gamma(k/2))) * x**((k/2) - 1) * math.exp(-x/2)
    # your code here
    return round(probability, 3)