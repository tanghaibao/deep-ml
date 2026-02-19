import numpy as np

def ddpm_reverse_step(x_t: np.ndarray, predicted_noise: np.ndarray, t: int,
                     betas: np.ndarray, noise: np.ndarray = None) -> np.ndarray:
    """
    Perform a single reverse (denoising) step of the DDPM sampling process.
    
    Args:
        x_t: Noisy sample at timestep t, shape (D,)
        predicted_noise: Model's noise prediction, shape (D,)
        t: Current timestep (1-indexed)
        betas: Noise schedule array, shape (T,)
        noise: Optional noise array for stochastic component, shape (D,)
    
    Returns:
        Denoised sample x_{t-1}, shape (D,)
    """
    alphas = 1 - betas
    alphasb = np.cumprod(alphas)
    t -= 1
    mu = 1 / np.sqrt(alphas[t]) * (x_t - betas[t] / np.sqrt(1 - alphasb[t]) * predicted_noise)
    if t == 0:
        return mu
    return mu + np.sqrt(betas[t]) * noise