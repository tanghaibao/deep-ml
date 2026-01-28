import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
	"""
	Solve the system Ax = b using the Conjugate Gradient method.

	:param A: Symmetric positive-definite matrix
	:param b: Right-hand side vector
	:param n: Maximum number of iterations
	:param x0: Initial guess for solution (default is zero vector)
	:param tol: Convergence tolerance
	:return: Solution vector x
	"""
	# calculate initial residual vector
    x = x0 or np.zeros_like(b, dtype=float)
    r0 = b - A @ x
    p = r0
    r = r0
    for _ in range(n):
        alpha = r @ r / (p.T @ A @ p)
        x1 = x + alpha * p
        r1 = r - alpha * A @ p
        if np.linalg.norm(r1) < tol:
            return x1        
		beta = r1 @ r1 / (r @ r) 
        p1 = r1 + beta * p
        p = p1
        r = r1
        x = x1
    return x 