import math
import numpy as np

def beta_distribution_stats(x: float, alpha: float, beta: float) -> dict:
    """
    Compute Beta distribution statistics.
    
    Args:
        x: Value at which to evaluate the PDF
        alpha: First shape parameter (alpha > 0)
        beta_param: Second shape parameter (beta > 0)
    
    Returns:
        Dictionary with 'pdf', 'mean', and 'variance'
    """
    # Your code here
    def B(alpha, beta):
        return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)
    pdf = x ** (alpha - 1) * (1 - x) ** (beta - 1) / B(alpha, beta) if 0 < x < 1 else 0
    mean = alpha / (alpha + beta)
    variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
    return {
        "pdf": np.round(pdf, 4),
        "mean": np.round(mean, 4),
        "variance": np.round(variance, 4),
    }