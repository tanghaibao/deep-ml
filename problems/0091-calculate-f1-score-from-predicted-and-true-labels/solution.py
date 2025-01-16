import numpy as np

def calculate_f1_score(y_true, y_pred):
	"""
	Calculate the F1 score based on true and predicted labels.

	Args:
		y_true (list): True labels (ground truth).
		y_pred (list): Predicted labels.

	Returns:
		float: The F1 score rounded to three decimal places.
	"""
	# Your code here
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	tp = (y_true & y_pred).sum()
	fp = (~y_true & y_pred).sum()
	fn = (y_true & ~y_pred).sum()
	if tp + fp == 0: return 0.0
	if tp + fn == 0: return 0.0
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	f1 = 2 * precision * recall / (precision + recall)
	return round(f1,3)