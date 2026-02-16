import numpy as np

def vae_loss(x: np.ndarray, x_reconstructed: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> tuple:
    """
    Compute the VAE loss (negative ELBO).

    Args:
        x: np.ndarray of shape (batch_size, features), original input
        x_reconstructed: np.ndarray of shape (batch_size, features), reconstructed input
        mu: np.ndarray of shape (batch_size, latent_dim), latent mean
        log_var: np.ndarray of shape (batch_size, latent_dim), latent log-variance

    Returns:
        tuple: (total_loss, reconstruction_loss, kl_divergence) as floats
    """
    # Your code here
    reconstruction_loss = (np.linalg.norm(x - x_reconstructed, axis=-1) ** 2).mean()
    kl_divergence = -1/2 * (1 + log_var - mu ** 2 - np.exp(log_var)).sum(axis=-1) 
    kl_loss = kl_divergence.mean()
    total_loss = reconstruction_loss + kl_loss
    _ = lambda x: round(float(x), 4)
    return _(total_loss), _(reconstruction_loss), _(kl_loss) 