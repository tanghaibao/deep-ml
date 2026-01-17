import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
	"""
	Compute entropy of P and cross-entropy between P and Q.
	
	Args:
		P: True probability distribution
		Q: Predicted probability distribution
	
	Returns:
		Tuple of (entropy H(P), cross-entropy H(P,Q))
	"""
	# Your code here
	h_p = -(P * np.log(np.array(P) + 1e-10)).sum()
    if h_p < 1e-10:
        h_p = 0.0
    h_p_q = -(P * np.log(np.array(Q) + 1e-10)).sum()
    return h_p, h_p_q
