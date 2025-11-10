import numpy as np

def muonclip_qk_clip(W_q: np.ndarray, W_k: np.ndarray, x: np.ndarray, t: float, alpha: float = 0.5, eps: float = 1e-7):
	"""
	Apply MuonClip qk-clip to (W_q, W_k).

	Args:
		W_q: (d_head, d_model) query projection weights
		W_k: (d_head, d_model) key  projection weights
		x: (batch, seq, d_model) input features
		t: threshold for max QK score (after 1/sqrt(d_head) scaling)
		alpha: fraction of rescaling applied to W_q (remainder to W_k)
		eps: small epsilon to avoid division by zero

	Returns:
		W_q_new (list[list[float]]), W_k_new (list[list[float]]), clipped (bool), max_post (float rounded to 4 dp)
	"""
	# compute q, k, scores; rescale if needed; round outputs
    W_q = np.array(W_q)
    W_k = np.array(W_k)
    x = np.array(x)
    d_head, _ = W_k.shape
	Q = x @ W_q.T
    K = x @ W_k.T
    S = Q @ K.transpose(0, 2, 1) / (d_head**0.5)
    M = S.max()
    eta = min(1, t / M)
    W_q *= eta ** alpha
    W_k *= eta ** (1 - alpha)
    clipped = eta < 1
    max_post = et