import numpy as np

def categorical_cross_entropy(y_true, y_pred, reduction, eps=1e-12):
  """
  Computes categorical cross-entropy loss for multi-class classification.
  
  Parameters:
  y_true (np.ndarray): True labels (one-hot encoded, shape (N, C)).
  y_pred (np.ndarray): Predicted probabilities (shape (N, C)).
  eps (float): A small value to prevent log(0) issues.
  
  Returns:
  float: The average categorical cross-entropy loss.
  """
  # Clip predictions to prevent log(0)
  y_pred = np.clip(y_pred, eps, 1)
  
  # Calculate the loss. Using np.sum allows for summing over all dimensions
  # and then dividing by the number of samples (y_true.shape[0]) for the mean.
  per_sample = -y_true * np.log(y_pred)
  N = len(per_sample)
  scale = 1.0
  if reduction == "mean":
    loss = float(per_sample.mean())
    scale = 1. / N
  elif reduction == "sum":
    loss = float(per_sample.sum())
  else:
    loss = per_sample
  return loss, scale

def loss_function(preds: np.ndarray, target: np.ndarray, reduction: str = "mean", eps: float = 1e-12):
  '''
  logits: [N, C] softmax probabilities
  target: [N]   class indices (int64)
  '''
  targets = np.zeros_like(preds)
  targets[:, target] = 1
  loss, scale = categorical_cross_entropy(targets, preds, reduction, eps=eps)
  grad_of_loss = -targets / preds * scale
  return loss, grad_of_loss