import numpy as np

def gaussian_log_likelihood(
    x: float, mu: float, sigma: float, sum: bool = False
) -> float:
    return -0.5 * np.log(2 * np.pi * sigma**2) - (x - mu) ** 2 / (2 * sigma**2)

def compute_elbo(x: list[float], q_mean: float, q_std: float, 
                 prior_mean: float, prior_std: float,
                 likelihood_std: float, n_samples: int = 1000) -> float:
    """
    Compute the Evidence Lower Bound (ELBO) for variational inference.
    
    Args:
        x: Observed data points
        q_mean: Mean of variational distribution q(z)
        q_std: Standard deviation of variational distribution q(z)
        prior_mean: Mean of prior distribution p(z)
        prior_std: Standard deviation of prior distribution p(z)
        likelihood_std: Standard deviation of likelihood p(x|z)
        n_samples: Number of Monte Carlo samples
    
    Returns:
        ELBO value (float)
    """
    # Your code here
    x = np.array(x)
    z = np.random.normal(q_mean, q_std, 