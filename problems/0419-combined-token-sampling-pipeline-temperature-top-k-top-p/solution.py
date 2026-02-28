import numpy as np

def combined_sampling(logits: list[float], temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, seed: int = 42) -> dict:
	logits_arr = np.array(logits, dtype=np.float64)
	n = len(logits_arr)
	
	# Greedy decoding
	if temperature <= 0:
		idx = int(np.argmax(logits_arr))
		probs = [0.0] * n
		probs[idx] = 1.0
		return {'probabilities': probs, 'sampled_token': idx}
	
	# Step 1: Temperature scaling
	scaled = logits_arr / temperature
	
	# Step 2: Top-k filtering
	if 0 < top_k < n:
		sorted_indices = np.argsort(scaled)[::-1]
		to_remove = sorted_indices[top_k:]
		scaled[to_remove] = -float('inf')
	
	# Step 3: Softmax
	max_val = np.max(scaled)
	exp_vals = np.exp(scaled - max_val)
	probs = exp_vals / np.sum(exp_vals)
	
	# Step 4: Top-p (nucleus) filtering
	if top_p < 1.0:
		sorted_idx = np.argsort(probs)[::-1]
		sorted_probs = probs[sorted_idx].copy()
		cumsum = np.cumsum(sorted_probs)
		
		# Mask tokens whose cumulative probability (before them) already meets top_p