import numpy as np

def warmup_cosine_schedule(T: int, W: int, lr_max: float, lr_min: float) -> list[float]:
	"""
	Compute learning rate schedule with linear warmup and cosine decay.
	
	Args:
		T: Total number of training steps
		W: Number of warmup steps
		lr_max: Maximum learning rate (reached after warmup)
		lr_min: Minimum learning rate (reached at end of training)
	
	Returns:
		List of learning rates for each step
	"""
	# Your code here
	lrs = []
    for t in range(T):
        if t < W:
            lr = t / W * lr_max
        else:
            lr = lr_min + 1 / 2 * (lr_max - lr_min) * (1 + np.cos((t - W) / (T - W) * np.pi))
        lrs.append(lr)
    return lrs