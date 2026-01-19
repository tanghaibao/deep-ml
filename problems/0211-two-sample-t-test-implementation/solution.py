import numpy as np
from scipy import stats

def two_sample_t_test(sample1: list[float], sample2: list[float], 
                      alpha: float = 0.05) -> dict:
	"""
	Perform a two-sample independent t-test (Welch's t-test).
	
	Args:
		sample1: First sample data
		sample2: Second sample data
		alpha: Significance level (default 0.05)
	
	Returns:
		Dictionary containing:
		- t_statistic: The calculated t-statistic
		- p_value: Two-tailed p-value
		- degrees_of_freedom: Degrees of freedom (Welch-Satterthwaite)
		- reject_null: Boolean, whether to reject null hypothesis
		- cohens_d: Effect size (Cohen's d)
	"""
	# Your code here
	x1 = np.array(sample1)
    x2 = np.array(sample2)
    n1 = x1.size
    n2 = x2.size
    m1, s1 = x1.mean(), x1.std(ddof=1)
    m2, s2 = x2.mean(), x2.std(ddof=1)
    t_statistic = (m1 - m2) / np.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    degrees_of_freedom = (s1 ** 2 / n1 + s2 ** 2 / n2) ** 2 / ((s1 ** 2 / n1) ** 2 / (n1 - 1) + (s2 ** 2 / n2) ** 2 / (n2 - 1))
    p