import numpy as np

def cfg_skip_speedup(
    T: int,
    skip_mask: list[bool],
    time_per_pass: float,
    cond_preds: list[float],
    uncond_preds: list[float],
    guidance_scale: float
) -> dict:
    """
    Analyze the speedup and quality impact of skipping unconditional
    passes in Classifier-Free Guidance diffusion inference.
    
    Args:
        T: Total number of denoising timesteps
        skip_mask: Boolean list; True means skip the unconditional pass at that step
        time_per_pass: Time in milliseconds for a single forward pass
        cond_preds: Conditional model predictions at each timestep
        uncond_preds: Unconditional model predictions at each timestep
        guidance_scale: CFG guidance scale (w)
    
    Returns:
        Dictionary with speedup metrics and guided outputs
    """
    # Your code here
    total_passes_standard = T * 2
    skip_mask = np.array(skip_mask)
    skipped = skip_mask.sum()
    total_passes_skipped = skipped + (T - skipped) 