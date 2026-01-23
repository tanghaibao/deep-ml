import numpy as np

def flatten(gradients: list[list[float]]) -> list[float]:
    return [item for sublist in gradients for item in sublist]

def clip_gradients_by_global_norm(gradients: list[list[float]], max_norm: float) -> list[list[float]]:
	"""
	Clip gradients by global norm.
	
	Args:
		gradients: List of gradient arrays
		max_norm: Maximum allowed global norm
	
	Returns:
		List of clipped gradient arrays
	"""
	# Your code here
    c = max_norm / (np.linalg.norm(flatten(gradients)) + 1e-12)
    if c < 1:
        return [list(np.round(c * np.array(g), 4)) for g in gradients]
    return gradients