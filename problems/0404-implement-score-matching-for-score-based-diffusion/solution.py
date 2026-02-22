import numpy as np

def denoising_score_matching_loss(X: np.ndarray, noisy_X: np.ndarray, sigma: float, predicted_scores: np.ndarray) -> dict:
    """
    Compute the denoising score matching loss for score-based diffusion models.
    
    Args:
        X: Clean data points, shape (n_samples, d)
        noisy_X: Noisy data points, shape (n_samples, d)
        sigma: Noise standard deviation (positive scalar)
        predicted_scores: Model score predictions at noisy points, shape (n_samples, d)
    
    Returns:
        Dictionary with keys 'target_scores', 'loss', 'weighted_loss'
    """
    target_scores = -(noisy_X - X) / sigma ** 2
    loss = (np.linalg.norm(predicted_scores - target_scores, axis=-1) ** 2).mean()
    weighted_loss = loss * sigma ** 2
    return {
            "target_scores": target_scores,
            "loss": loss,
            "weighted_loss": weighted_loss,
            }