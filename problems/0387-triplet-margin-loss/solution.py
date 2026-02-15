import numpy as np

def triplet_margin_loss(anchor: np.ndarray, positive: np.ndarray, negative: np.ndarray, margin: float = 1.0) -> float:
    """
    Compute the triplet margin loss for metric learning.
    
    Args:
        anchor: Anchor embeddings, shape (D,) for single or (N, D) for batch
        positive: Positive embeddings (same class as anchor), same shape as anchor
        negative: Negative embeddings (different class from anchor), same shape as anchor
        margin: Minimum desired distance gap between positive and negative pairs
    
    Returns:
        Mean triplet margin loss as a float
    """
    dap = np.linalg.norm(anchor - positive, axis=-1)
    dan = np.linalg.norm(anchor - negative, axis=-1)
    losses = np.maximum(0, dap - dan + margin)
    return losses.mean()