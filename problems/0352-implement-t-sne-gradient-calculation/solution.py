import numpy as np

def tsne_gradient(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
	"""
	Compute the gradient of the t-SNE cost function.
	
	Args:
		P: (n, n) symmetric numpy array of joint probabilities in high-dimensional space
		Y: (n, d) numpy array of current low-dimensional embedding
	
	Returns:
		gradient: (n, d) numpy array of gradients for each point, rounded to 4 decimal places
	"""
    n, _ = Y.shape
    QQ = np.linalg.norm(Y[..., None] - Y.T[None, ...], axis=1)
    QQ = 1 / (1 + QQ ** 2)
    s = QQ.sum() - n
    Q = QQ / s
    G = 4 * (P - Q)[:, None] * (Y[..., None] - Y.T[None, ...]) * QQ[:, None]
    return np.round(G.sum(axis=-1), 4)