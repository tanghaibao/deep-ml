import numpy as np

def suggest_rank(delta_W: np.ndarray, energy_threshold: float) -> int:
	"""
	Return the smallest rank k such that the top-k singular values of delta_W
	capture at least `energy_threshold` of the total squared-singular-value energy.
	"""
	# Your code here
	if not np.any(delta_W):
		return 0
	U, s, Vh = np.linalg.svd(delta_W)
	ss = s ** 2
	ss /= ss.sum()
	return np.argmax(np.cumsum(ss) >= energy_threshold) + 1