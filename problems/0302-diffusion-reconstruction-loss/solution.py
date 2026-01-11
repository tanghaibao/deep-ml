import numpy as np

def diffusion_loss(x_0: np.ndarray, t: int, beta_start: float, beta_end: float, num_timesteps: int, noise: np.ndarray, predicted_noise: np.ndarray) -> float:
    """
    Compute the reconstruction loss for diffusion model training.
    
    Args:
        x_0: Original input data (numpy array)
        t: Timestep (1-indexed, from 1 to num_timesteps)
        beta_start: Starting value of linear beta schedule
        beta_end: Ending value of linear beta schedule
        num_timesteps: Total number of diffusion timesteps
        noise: True noise array (same shape as x_0)
        predicted_noise: Model's predicted noise (same shape as x_0)
    
    Returns:
        Mean squared error loss (float)
    """
    alphas = 1 - np.linspace(beta_start, beta_end, num_timesteps)
    alpha = np.cumprod(alphas)[t - 1]
    x_t = np.sqrt(alpha) * x_0 + np.sqrt(1 - alpha) * noise
    x_0_hat = (x_t - np.sqrt(1 - alpha) * predicted_noise) / np.sqrt(alpha)
    return float(np.mean((x_0