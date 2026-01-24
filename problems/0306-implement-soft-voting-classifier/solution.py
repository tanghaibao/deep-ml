import numpy as np

def soft_voting_classifier(probabilities: np.ndarray, weights: list = None) -> list:
	"""
	Implement soft voting for ensemble classification.
	
	Args:
		probabilities: 3D array of shape (n_classifiers, n_samples, n_classes)
		weights: Optional list of weights for each classifier
	
	Returns:
		List of predicted class labels for each sample
	"""
	# Your code here
    probabilities = np.transpose(probabilities, axes=(1, 0, 2))
    if weights:
        weights = np.array(weights, dtype=float)
        weights /= weights.sum()
        p = weights @ probabilities
    else:
        p = probabilities.mean(axis=1)
    return list(np.argmax(p, axis=1))