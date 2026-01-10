import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	# Your code here
	g = np.array(gradient)
    magnitude = np.linalg.norm(g)
    direction = g / magnitude if magnitude else np.zeros_like(g)
    descent_direction = -direction
    return {
        "magnitude": magnitude, 
        "direction": list(direction), 
        "descent_direction": list(descent_direction)
    }