import numpy as np

def focal_loss(y_true, y_pred, gamma=2.0, alpha=None):
	"""
	Compute Focal Loss for multi-class classification.
	
	Args:
		y_true: Ground truth labels as class indices (list or 1D array)
		y_pred: Predicted probabilities (2D array, shape: [n_samples, n_classes])
		gamma: Focusing parameter (default: 2.0)
		alpha: Class weights (optional, list or 1D array of length n_classes)
	
	Returns:
		float: Average focal loss
	"""
	# Your code here
    N = len(y_true)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    pt = y_pred[np.arange(N), y_true]
    fl = -((1 - pt) ** gamma) * np.log(pt)
    if alpha is not None:
        fl *= np.array(alpha)[y_true]
    return fl.mean()