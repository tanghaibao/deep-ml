import numpy as np

def exponential_distribution(x: list, lam: float) -> dict:
    """
    Compute exponential distribution properties.
    
    Args:
        x: Points at which to evaluate PDF and CDF
        lam: Rate parameter (lambda) of the distribution
        
    Returns:
        Dictionary with 'pdf', 'cdf', 'mean', and 'variance' keys
    """
    # Your code here
    if lam <= 0:
        return {
            "pdf": None,
            "cdf": None,
            "mean": None,
            "variance": None,
        }
    x = np.array(x, dtype=float)
    pdf = lam * np.exp(-lam * x)
    cdf = 1 - np.exp(-lam * x)
    mean = 1. / lam
    variance =  1. / lam ** 2
    pdf[x < 0] = 0
    cdf[x < 0] = 0
    return {
        "pdf": list(np.round(pdf, 4)),
        "cdf": list(np.round(cdf, 4)),
        "mean": np.round(mean, 4),
        "variance": np.round(variance, 4),
    }