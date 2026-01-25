import numpy as np

def compute_null_space(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Compute an orthonormal basis for the null space (kernel) of matrix A.
    
    Args:
        A: Input matrix of shape (m, n)
        tol: Tolerance for considering singular values as zero
    
    Returns:
        Matrix of shape (n, k) where k is the dimension of the null space.
        Columns form an orthonormal basis for the null space.
    """
    # Your code here
    u, s, vh = np.linalg.svd(A)
    n = A.shape[1]
    S = np.zeros(n)
    S[: len(s)] = s
    mask = np.abs(S) < tol
    return vh[mask].T