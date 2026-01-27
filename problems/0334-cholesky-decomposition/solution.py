import numpy as np

def cholesky_decomposition(A):
    """
    Perform Cholesky decomposition on a symmetric positive-definite matrix.
    
    Args:
        A: A symmetric positive-definite matrix (2D list or numpy array)
    
    Returns:
        L: Lower triangular matrix such that A = L @ L.T as a 2D list,
           or -1 if decomposition is not possible
    """
    # Your code here
    A = np.array(A, dtype=float)
    L = np.zeros_like(A)
    n = A.shape[0]
    for i in range(n):
        for j in range(i):
            L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]
        t = A[i, i] - np.sum(L[i, :i] ** 2)
        if t <= 0:
            return -1
        L[i, i] = np.sqrt(t)
    return L