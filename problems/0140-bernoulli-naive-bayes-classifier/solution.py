import numpy as np

class NaiveBayes():
    def __init__(self, smoothing=1.0):
        # Initialize smoothing
        self.smoothing = smoothing

    def forward(self, X, y):
        # Fit model to binary features X and labels y
        X = np.array(X)
        y = np.array(y)
        def get_X(label):
            return X[y == label]
        N, k = X.shape
        self.prior = np.zeros(2)
        self.prob = np.zeros((k, 2))
        alpha = self.smoothing
        for i in range(2):
            X0 = get_X(i)
            n, d = X0.shape
            self.prob[:, i] = (X0.sum(axis = 0) + alpha) / (n + alpha * d)
            self.prior[i] = n / N

    def predict(self, X):
        # Predict class labels for test set X
        prob = self.prob
        res = X @ np.log(prob + 1e-12) + (1 - X) @ np.log(1 - prob + 1e-10) + np.log(self.prior + 1e-12)
        return np.argmax(res, axis=1)