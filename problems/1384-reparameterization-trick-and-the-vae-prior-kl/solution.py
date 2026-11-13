import numpy as np

def reparameterize(mu, logvar, eps):
    """z = mu + exp(0.5 * logvar) * eps, all shapes (batch, latent_dim)."""
    # Your code here
    return mu + np.exp(0.5 * logvar) * eps

def kl_to_standard_normal(mu, logvar):
    """KL(N(mu, sigma^2) || N(0, I)) summed over latent dims, averaged over the batch. Returns float."""
    # Your code here
    return np.sum(0.5 * (mu ** 2 + np.exp(logvar) - 1 - logvar), axis=-1).mean()
