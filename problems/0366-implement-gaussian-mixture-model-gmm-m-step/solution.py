import numpy as np

def gmm_m_step(X, gamma):
    """
    Perform the M-step of the EM algorithm for Gaussian Mixture Model.
    
    Args:
        X: Data points, shape (N, D)
        gamma: Responsibilities from E-step, shape (N, K)
    
    Returns:
        tuple: (means, covariances, mixing_coeffs)
            - means: Updated component means, shape (K, D)
            - covariances: Updated component covariances, shape (K, D, D)
            - mixing_coeffs: Updated mixing coefficients, shape (K,)
    """
    # Your code here
    X = np.array(X) # (N, D)
    gamma = np.array(gamma) # (N, K)
    Nk = gamma.sum(axis=0) # (K,)
    mixing_coeffs = Nk / Nk.sum()
    means = np.einsum("nk,nd->kd", gamma, X) / Nk[:, None] # (K, D) 
    d = X[:, None, :] - means[None, :, :] # (N, K, D)
    covariances = np.einsum("nk,nkc,dkn->kcd", gamma, d, d.T) / Nk
    return means, covariances, mixing_coeffs