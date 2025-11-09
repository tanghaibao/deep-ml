import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
	"""
	Compute the Huber Loss between true and predicted values.

	Args:
		y_true (float | list[float]): Ground truth values
		y_pred (float | list[float]): Predicted values
		delta (float): Transition threshold between MSE and MAE behavior

	Returns:
		float: Average Huber loss
	"""
	# Your code here
	diff = np.abs(np.array(y_true) - np.array(y_pred))
    select = diff <= delta
    return np.average(select * 0.5 * diff ** 2 + (~select) * delta * (diff - 0.5 * delta))