import numpy as np

def gaussian_pdf(x, mean, variance):
    return 1 / np.sqrt(2 * np.pi * variance) * np.exp(-(x - mean) ** 2 / (2 * variance))

def gmm_e_step(X: np.ndarray, means: np.ndarray, variances: np.ndarray, 
               mixing_coeffs: np.ndarray) -> np.ndarray:
    """
    Compute the E-step of Gaussian Mixture Model.
    
    Args:
        X: Data points of shape (n_samples,)
        means: Component means of shape (n_components,)
        variances: Component variances of shape (n_components,)
        mixing_coeffs: Mixing coefficients of shape (n_components,)
    
    Returns:
        Responsibility matrix of shape (n_samples, n_components)
    """
    nn = gaussian_pdf(X[:, None], means, variances)
    gamma = mixing_coeffs[None, :] * nn
    return gamma / gamma.sum(axis=1, keepdims=True)