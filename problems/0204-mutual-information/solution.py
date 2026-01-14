import numpy as np

def mutual_information(joint_prob: list[list[float]]) -> float:
	"""
	Compute the mutual information between two random variables.
	
	Args:
		joint_prob: 2D joint probability distribution P(X,Y)
	
	Returns:
		Mutual information I(X;Y)
	"""
	# Your code here
	p = np.array(joint_prob)
    p_x = p.sum(axis=0)
    p_y = p.sum(axis=1)
    epsilon = 1e-10
    total = p * np.log(p / (p_x * p_y) + epsilon)
    return total.sum()