import numpy as np

def bagging_classifier(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, n_estimators: int = 10, seed: int = 42) -> np.ndarray:
    """
    Implement a bagging classifier using decision stumps.
    
    Args:
        X_train: Training features of shape (n_samples, n_features)
        y_train: Training labels of shape (n_samples,), binary {0, 1}
        X_test: Test features of shape (n_test_samples, n_features)
        n_estimators: Number of bootstrap samples/base estimators
        seed: Random seed for reproducibility
    
    Returns:
        np.ndarray: Predicted labels for X_test
    """
    # Your code here
    np.random.seed(seed)
    n_samples, n_features = X_train.shape
    def find_best_h(X, y):
        best = (np.inf, 0, 0, 0)
        for j in range(n_features):
            x = X[:, j]
            for t in np.unique(x):
                a = x <= t
                best = min(best, ((y != a).sum(), j, t, 0))
                best = min(best, ((y 