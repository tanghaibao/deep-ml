import numpy as np

def early_stopping(val_losses: list[float], patience: int = 5, min_delta: float = 0.0) -> list[bool]:
	"""
	Determine at each epoch whether training should stop based on validation loss.
	
	Args:
		val_losses: List of validation losses at each epoch
		patience: Number of epochs to wait for improvement before stopping
		min_delta: Minimum change in validation loss to qualify as improvement
	
	Returns:
		List of booleans indicating whether to stop at each epoch
	"""
	# Your code here
    c = 0
    best_loss = val_losses[0] + min_delta + 1
    bt = len(val_losses)
    res = []
    for t, loss in enumerate(val_losses):
        if loss < best_loss - min_delta:
            best_loss = loss
            c = 0
        res.append(c >= patience)
        c += 1
    return res
    
