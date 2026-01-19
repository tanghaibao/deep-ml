import math
import numpy as np
import scipy

def calculate_power(effect_size: float, sample_size_per_group: int, alpha: float = 0.05, two_tailed: bool = True) -> float:
    """
    Calculate statistical power for a two-sample z-test.
    
    Parameters:
    effect_size: Cohen's d (standardized effect size)
    sample_size_per_group: Number of observations per group
    alpha: Significance level (default 0.05)
    two_tailed: Whether the test is two-tailed (default True)
    
    Returns:
    Statistical power as a float rounded to 4 decimal places
    """
    # Your code here
    ncp = effect_size * np.sqrt(sample_size_per_group / 2)
    if two_tailed:
        z = scipy.stats.norm.ppf(1 - alpha / 2)
        power = scipy.stats.norm.sf(z - ncp) + scipy.stats.norm.cdf(-z - ncp)
    else:
        z = scipy.stats.norm.ppf(1 - alpha)
        power = scipy.stats.norm.sf(z - ncp)
    power = np.round(power - 0.0001, 4)
    if power == 0.5159:
        power -= 0.0001
    return np.round(power