import numpy as np

def gaussian_pdf(x: float, mu: float, variance: float) -> float:
    return 1 / np.sqrt(2 * np.pi * variance + 1e-10) * np.exp(-((x - mu) ** 2) / (2 * variance + 1e-10))

def fit_gmm_1d(X, K, initial_means, initial_variances, initial_weights, n_iterations):
    """
    Fit a 1D Gaussian Mixture Model using the EM algorithm.
    
    Args:
        X: List of data points
        K: Number of mixture components
        initial_means: List of initial means for each component
        initial_variances: List of initial variances for each component
        initial_weights: List of initial mixture weights (should sum to 1)
        n_iterations: Number of EM iterations to run
    
    Returns:
        Dictionary with 'means', 'variances', 'weights' as lists rounded to 4 decimals
    """
    # Your code here
    means = initial_means
    variances = initial_variances
    weights = initial_weights
    X = np.array(X)
    for _ in range(n_iterations):
        # E-step
        r