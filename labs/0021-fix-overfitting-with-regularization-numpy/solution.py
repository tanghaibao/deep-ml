import numpy as np

def train(X_train, y_train, X_val, y_val):
    """
    Train a regression model that generalizes well despite having
    MORE features than training samples (many are noise or redundant).
    
    WARNING: An unregularized approach WILL overfit here.
    - Unregularized OLS: Train R² ≈ 1.0, Val R² ≈ -5.0
    - You need regularization to pass!
    
    Args:
        X_train: numpy array of shape (n_samples, n_features) -- standardized
                 (~250 samples, ~264 features -- more features than samples!)
        y_train: numpy array of shape (n_samples,) -- target values
        X_val:   numpy array of shape (n_val, n_features) -- standardized
        y_val:   numpy array of shape (n_val,) -- validation targets
    
    Returns:
        predict: callable that takes X (n, n_features) and returns y_pred (n,)
    """
    # TODO: implement a training strategy that avoids overfitting
    # Some ideas:
    #   - Ridge regression (L2): add alpha * ||W||^2 to loss
    #   - Gradient descent with L2 penalty
    #   - Feature selection (remove low-variance or uncorrelated features)
    #   - Early stopping on gradient descent
    #   - Any combination of the above!
    
    n_samples, n_features = X_train.shape
    X = np.column_stack([X_train, np.ones(n_samples)])
    y = y_train
    X_val = np.column_stack([X_val, np.ones(len(X_val))])
    results = []
    for alpha in (0.01, 0.1, 1, 10, 100, 1000):
        beta = np.linalg.inv(X.T @ X + alpha * np.eye(n_features + 1)) @ X.T @ y
        preds = X_val @ beta
        loss = ((preds - y_val) ** 2).mean()
        results.append((loss, alpha))
    _, alpha = min(results)

    def predict(X_test):
        X_test = np.column_stack([X_test, np.ones(len(X_test))])
        return X_test @ beta
    
    return predict