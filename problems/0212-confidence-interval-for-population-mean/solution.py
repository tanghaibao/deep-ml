import numpy as np
from scipy import stats

def confidence_interval(data: list[float], confidence_level: float = 0.95) -> dict:
	"""
	Calculate confidence interval for population mean.
	
	Args:
		data: Sample data
		confidence_level: Confidence level (default 0.95)
	
	Returns:
		Dictionary containing:
		- mean: Sample mean (point estimate)
		- standard_error: Standard error of the mean
		- margin_of_error: Margin of error
		- lower_bound: Lower bound of CI
		- upper_bound: Upper bound of CI
		- confidence_level: Confidence level used
	"""
	# Your code here
	mean = np.mean(data)
    n = len(data)
    # standard_error = np.std(data, ddof=1) / np.sqrt(n)
    standard_error = stats.sem(data)
    alpha = 1 - confidence_level
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    margin_of_error = t_crit * standard_error
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    return {
        "mean": mean,
        "standard_error": standard_error,
        "margin