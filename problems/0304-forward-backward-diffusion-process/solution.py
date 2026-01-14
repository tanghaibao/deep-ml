import numpy as np

def diffusion_process(x_0: np.ndarray, 
                      betas: np.ndarray,
                      timestep: int,
                      forward_noise: np.ndarray,
                      predicted_noise: np.ndarray,
                      backward_noise: np.ndarray = None) -> tuple:
    """
    Implement forward and backward diffusion processes.
    
    Args:
        x_0: Original clean data (any shape)
        betas: Noise schedule array of shape (T,)
        timestep: Current timestep t (1-indexed)
        forward_noise: Noise epsilon for forward diffusion
        predicted_noise: Model's predicted noise for backward diffusion
        backward_noise: Random noise z for stochastic backward step
    
    Returns:
        tuple: (x_t, x_t_minus_1) - noisy and denoised samples
    """
    t = timestep - 1
    alphas = 1 - betas
    alphas_b = np.cumprod(alphas)
    alpha_t = alphas_b[t] 
    x_t = np.sqrt(alpha_t) * x_0 + np.sqrt(1 - alpha_t) * forward_noise
    x_t