import numpy as np

def logsumexp(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))


def contrastive_loss(embeddings: np.ndarray, temperature: float) -> float:
    """
    Compute the NT-Xent (SimCLR-style) contrastive loss.
    
    Args:
        embeddings: Array of shape (2N, d) where consecutive pairs
                    (2i, 2i+1) are positive pairs.
        temperature: Temperature scaling parameter (tau > 0).
    
    Returns:
        The mean contrastive loss as a float.
    """
    nn, _ = embeddings.shape
    s = (embeddings[:, None] * embeddings[None, :]).sum(axis=-1) # (2N, 2N)
    mod = np.linalg.norm(embeddings, axis=-1)
    s /= mod[:, None] * mod[None, :]
    losses = []
    for i in range(nn):
        j = i + 1 if i % 2 == 0 else i - 1
        x = s[i, :] / temperature
        x[i] = -np.inf
        loss = -x[j] + logsumexp(x) 
        losses.append(loss)
    return np.mean(losses)