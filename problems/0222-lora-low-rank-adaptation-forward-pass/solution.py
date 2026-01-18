import numpy as np

def lora_forward(
	x: list[list[float]],
	W: list[list[float]],
	A: list[list[float]],
	B: list[list[float]],
	alpha: float = 1.0
) -> list[list[float]]:
	"""
	Compute the LoRA forward pass.
	
	Args:
		x: Input matrix (batch_size x in_features)
		W: Frozen pretrained weights (in_features x out_features)
		A: LoRA matrix A (rank x out_features)
		B: LoRA matrix B (in_features x rank)
		alpha: LoRA scaling factor
		
	Returns:
		Output matrix (batch_size x out_features)
	"""
	# Your code here
    r = len(A)
    x = np.array(x)
    W = np.array(W)
    B = np.array(B)
    A = np.array(A)
	return np.round(x @ W + alpha / r * x @ B @ A, 4)