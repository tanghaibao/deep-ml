import numpy as np

def gaussian_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
	"""
	Implements Gaussian Naive Bayes classifier.
	
	Args:
		X_train: Training features (shape: N_train x D)
		y_train: Training labels (shape: N_train)
		X_test: Test features (shape: N_test x D)
	
	Returns:
		Predicted class labels for X_test (shape: N_test)
	"""
	# Your code here
    N, D = X_train.shape
    K = len(np.unique(y_train))
    mus = np.zeros((D, K))
    var = np.zeros((D, K))
    priors = np.zeros(K)
    for i in np.unique(y_train):
        X0 = X_train[y_train == i]
        n, _ = X0.shape
        mus[:, i] = X0.mean(axis=0)
        var[:, i] = X0.var(axis=0)
        priors[i] = n / N
    res = (
        -0.5 * np.log(2 * np.pi * var[np.newaxis, :, :])
        - (X_test[:, :, np.newaxis] - mus[np.newaxis, :, :]) ** 2
        / (2 * var[np.newaxis, :, :])
    ).sum(axis=1) + np.log(priors)
    y_pred = np.argmax(res, axis=1)
    return y_pred