import numpy as np

def classifier_free_guidance_step(
    eps_uncond: list,
    eps_cond: list,
    x_t: list,
    guidance_scale: float,
    alpha_bar_t: float,
    alpha_bar_t_minus_1: float,
    beta_t: float
) -> tuple:
    """
    Perform one reverse diffusion step with Classifier-Free Guidance.
    
    Args:
        eps_uncond: Unconditional noise prediction from the model
        eps_cond: Conditional noise prediction from the model
        x_t: Current noisy sample at timestep t
        guidance_scale: CFG weight (w). w=1 gives standard conditional, w>1 amplifies conditioning
        alpha_bar_t: Cumulative product of alpha up to timestep t
        alpha_bar_t_minus_1: Cumulative product of alpha up to timestep t-1
        beta_t: Noise schedule value at timestep t
    
    Returns:
        Tuple of (guided_eps, predicted_x0, posterior_mean) as lists rounded to 4 decimals
    """
    w = guidance_scale
    eps_uncond = np.array(eps_uncond)
    eps_cond = np.array(eps_cond)
  