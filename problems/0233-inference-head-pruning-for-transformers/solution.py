import numpy as np

def prune_attention_heads(
    attention_weights: np.ndarray,
    head_importance_scores: np.ndarray,
    pruning_ratio: float
) -> tuple[np.ndarray, list[int]]:
	"""
	Prune less important attention heads to reduce inference cost.
	
	Transformer models have redundant heads. Research shows 50% of
	BERT heads can be removed with <1% accuracy loss.
	
	Args:
		attention_weights: Attention weights from all heads
		  Shape: (num_heads, seq_len, seq_len)
		head_importance_scores: Importance score per head
		  Shape: (num_heads,)
		  Higher score = more important
		pruning_ratio: Fraction of heads to prune
		  Range: 0.0 (keep all) to 1.0 (prune all)
	
	Returns:
		Tuple of (pruned_attention_weights, kept_head_indices):
		- pruned_attention_weights: Reduced attention matrices
		- kept_head_indices: List of preserved head indices
	"""
	# Your code here
    n = len(head_importance_scores)
    to_prune = int(n * pruning_ratio)
    kept_head_indices = np.argsort(head_importance_