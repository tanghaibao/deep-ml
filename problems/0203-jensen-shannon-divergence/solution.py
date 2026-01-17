import numpy as np

def jensen_shannon_divergence(P: list[float], Q: list[float]) -> float:
	"""
	Compute the Jensen-Shannon Divergence between two probability distributions.
	
	Args:
		P: First probability distribution
		Q: Second probability distribution
	
	Returns:
		Jensen-Shannon Divergence value
	"""
	# Your code here
	P = np.array(P)
    Q = np.array(Q)
    M = (P + Q) / 2
    def kl_div(A, B):
        return np.dot(A, np.log(A / B + 1e-10))
    kl_P_M = kl_div(P, M)
    kl_Q_M = kl_div(Q, M)
    return (kl_P_M + kl_Q_M) / 2