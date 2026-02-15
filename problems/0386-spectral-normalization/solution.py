import numpy as np

def spectral_normalization(W, num_iters=10, u_init=None):
    """
    Apply spectral normalization to a weight matrix using power iteration.
    
    Args:
        W: Weight matrix of shape (m, n)
        num_iters: Number of power iteration steps
        u_init: Optional initial vector of shape (m,)
    
    Returns:
        Tuple of (W_sn, sigma):
            W_sn: Spectrally normalized weight matrix
            sigma: Estimated spectral norm (largest singular value)
    """
    m, _ = W.shape
    if u_init is None:
        u = np.random.random(m)
    else:
        u = u_init
    u /= np.linalg.norm(u)
    for _ in range(num_iters):
        vh = W.T @ u
        v = vh / np.linalg.norm(vh)
        uh = W @ v
        u = uh / np.linalg.norm(uh)
    sigma = u @ W @ v
    return (W / sigma, sigma)