import numpy as np


def softmax(X):
    exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exp_X / np.sum(exp_X, axis=1, keepdims=True)


def onehot(y):
    n_classes = np.max(y) + 1
    return np.eye(n_classes)[y]


def train_softmaxreg(
    X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int
) -> tuple[list[float], ...]:
    """
    Gradient-descent training algorithm for Softmax regression, optimizing parameters with Cross Entropy loss.
    """
    # Your code here
    m, n_features = X.shape
    y = onehot(y)
    _, C = y.shape
    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    W = np.zeros((n_features + 1, C))
    losses = []
    # forward
    for _ in range(iterations):
        probs = softmax(X @ W)
        loss = -np.sum(y * np.log(probs + 1e-10))
        losses.append(loss)
        # backward
        g = X.T @ (probs - y)
        W -= learning_rate * g
    return np.round(W, 4).T, np.round(losses, 4)