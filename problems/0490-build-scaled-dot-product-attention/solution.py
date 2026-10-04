import numpy as np

def softmax(S):
	E = np.exp(S - S.max(keepdims=True, axis=-1))
	return E / E.sum(keepdims=True, axis=-1)

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None) -> tuple:
	"""
	Compute Scaled Dot-Product Attention.
	
	Args:
		Q: Query matrix of shape (seq_len_q, d_k)
		K: Key matrix of shape (seq_len_k, d_k)
		V: Value matrix of shape (seq_len_k, d_v)
		mask: Optional binary mask of shape (seq_len_q, seq_len_k)
	
	Returns:
		Tuple of (output, attention_weights)
	"""
	# Your code here
	_, d_k = Q.shape
	S = Q @ K.T
	S /= np.sqrt(d_k)
	if mask is not None:
		S = np.where(mask, S, -np.inf)
	weights = softmax(S)
	return weights @ V, weights
