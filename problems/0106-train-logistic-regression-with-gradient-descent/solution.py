import numpy as np

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""
	# Your code here
    m, n = X.shape
    X = np.concatenate((np.ones(m)[:, None], X), axis=1)
    W = np.zeros(n + 1)
    losses = []
    for _ in range(iterations):
        probs = sigmoid(X @ W)
        loss = -np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))
        losses.append(loss)
        g = X.T @ (probs - y)
        W -= learning_rate * g
    return np.round(W, 4), np.round(losses, 4)