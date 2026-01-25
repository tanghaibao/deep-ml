import numpy as np

def qr_decomposition(A: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
	"""
	Perform QR decomposition using Gram-Schmidt process.
	
	Args:
		A: An m x n matrix represented as list of lists
	
	Returns:
		Tuple of (Q, R) where Q is orthogonal and R is upper triangular
	"""
	# Your code here
    A = np.array(A)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j] - np.sum(Q[:, :j] * (Q[:, :j].T @ A[:, j]), axis=1)
        Q[:, j] = v / np.linalg.norm(v)
        R[:j, j] = Q[:, :j].T @ A[:, j]
		R[j, j] = np.linalg.norm(v)
    return Q.tolist(), R.tolist()
