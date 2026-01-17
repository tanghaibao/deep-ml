from typing import Callable
import numpy as np

def compute_hessian(f: Callable[[list[float]], float], point: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Hessian matrix of function f at the given point using finite differences.
	
	Args:
		f: A scalar function that takes a list of floats and returns a float
		point: The point at which to compute the Hessian (list of coordinates)
		h: Step size for finite differences (default: 1e-5)
		
	Returns:
		The Hessian matrix as a list of lists (n x n where n = len(point))
	"""
	# Your code here
	n = len(point)
    def ff(x, i, g, j = None, h = None):
        x = np.array(x)
        x[i] += g
        if j is not None:
            x[j] += h
        return f(list(x))
    ans = np.zeros((n, n))
    x = point
    for i in range(n):
        for j in range(n):
            if i == j:
                ans[i, i] = (ff(x, i, h) - 2 * f(x) + ff(x, i, -h)) / h ** 2
            else:
                ans[i, j] = (ff(x, i, h, j, h) - ff(