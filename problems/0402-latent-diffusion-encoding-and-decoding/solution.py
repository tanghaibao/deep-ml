import numpy as np

def latent_encode_decode(x, W_enc_mu, b_enc_mu, W_enc_logvar, b_enc_logvar, W_dec, b_dec, epsilon):
    """
    Perform variational encoding, latent sampling, and decoding
    as used in latent diffusion models.
    
    Args:
        x: Input array of shape (d,)
        W_enc_mu: Encoder mean weights, shape (d, latent_dim)
        b_enc_mu: Encoder mean bias, shape (latent_dim,)
        W_enc_logvar: Encoder log-variance weights, shape (d, latent_dim)
        b_enc_logvar: Encoder log-variance bias, shape (latent_dim,)
        W_dec: Decoder weights, shape (latent_dim, d)
        b_dec: Decoder bias, shape (d,)
        epsilon: Noise sample, shape (latent_dim,)
    
    Returns:
        dict with keys 'mu', 'log_var', 'z', 'x_recon' (all numpy arrays)
    """
    # Your code here
    mu = x @ W_enc_mu + b_enc_mu # (latent_dim,)
    log_var = x @ W_enc_logvar + b_enc_logvar # (latent_dim,) 
    sigma = np.exp(0.5 * log_var) # (latent_dim,)
    z = mu + sigma * epsil