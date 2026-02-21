import numpy as np

def cosine_noise_schedule(T: int, s: float = 0.008, beta_max: float = 0.999) -> dict:
    """
    Compute the cosine noise schedule for a diffusion model.
    
    Args:
        T: Total number of diffusion timesteps
        s: Small offset to prevent alpha_bar from being too small near t=0
        beta_max: Maximum value for beta clipping
    
    Returns:
        Dictionary with keys 'betas', 'alphas', 'alpha_bars',
        each containing a numpy array of length T.
    """
    # Your code here
    f = np.cos((np.arange(T + 1) / T + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    betas = np.minimum(betas, beta_max)
    alphas = 1 - betas
    alpha_bars = np.cumprod(alphas)
    return {
            "betas": betas,
            "alphas": alphas,
            "alpha_bars": alpha_bars,
            } 