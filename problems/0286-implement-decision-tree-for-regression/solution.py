import numpy as np

def decision_tree_regressor(X_train, y_train, X_test, max_depth=2, min_samples_split=2):
    """
    Build a decision tree for regression and predict on test data.
    
    Args:
        X_train: Training features, shape (n_samples, n_features)
        y_train: Training targets, shape (n_samples,)
        X_test: Test features, shape (m_samples, n_features)
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum samples required to split a node
    
    Returns:
        List of predictions for X_test, rounded to 4 decimal places
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    n, n_feats = X_train.shape
    
    def mse(targets):
        return y_train[targets].var()

    def gain(targets, col, thresh):
        is_left = X_train[targets, col] <= thresh
        left = targets[is_left]
        right = targets[~is_left]
        split_mse = (len(left) * mse(left) + len(right) * mse(right)) 