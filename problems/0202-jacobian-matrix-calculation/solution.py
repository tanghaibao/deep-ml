import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Jacobian matrix using numerical differentiation.
	
	Args:
		f: Function that takes a list and returns a list
		x: Point at which to evaluate the Jacobian
		h: Step size for finite differences
	
	Returns:
		Jacobian matrix as list of lists
	"""
	# Your code here
    x0 = np.asarray(x, dtype=float)
    f0 = np.asarray(f(x0), dtype=float)

    n = x0.size
    m = f0.size

    J = np.zeros((m, n))

    for i in range(n):
        x1 = x0.copy()
        x1[i] += h
        fi = np.asarray(f(x1), dtype=float)
        J[:, i] = (fi - f0) / h

    return J

