import numpy as np

def ddpm_linear_schedule(beta_start: float, beta_end: float, T: int) -> dict:
    """
    Compute the DDPM linear noise schedule and derived quantities.

    Args:
        beta_start: Starting value of the beta (noise variance) schedule.
        beta_end: Ending value of the beta schedule.
        T: Number of diffusion timesteps.

    Returns:
        Dictionary with keys: 'betas', 'alphas', 'alpha_bars',
        'sqrt_alpha_bars', 'sqrt_one_minus_alpha_bars'.
    """
    # Your code here
    t = np.arange(1, T + 1)
    if T == 1:
        betas = np.array([beta_start])
    else:
        betas = beta_start + (t - 1) / (T - 1) * (beta_end - beta_start)
    alphas = 1 - betas
    alpha_bars = np.cumprod(alphas)
    sqrt_alpha_bars = np.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars = np.sqrt(1 - alpha_bars)
    return {
            "betas": betas.tolist(),
            "alphas": alphas.tolist(),
            "alpha_bars": alpha_bars.tolist(),
            "sqrt_alpha_bar