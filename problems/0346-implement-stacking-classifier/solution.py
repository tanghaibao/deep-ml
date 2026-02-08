import numpy as np

def stacking_classifier(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                        base_classifiers: list, meta_classifier, n_folds: int = 5) -> np.ndarray:
    """
    Implement a stacking classifier ensemble.
    
    Args:
        X_train: Training features of shape (n_samples, n_features)
        y_train: Training labels of shape (n_samples,)
        X_test: Test features of shape (m_samples, n_features)
        base_classifiers: List of classifier functions
        meta_classifier: Meta-level classifier function
        n_folds: Number of cross-validation folds
    
    Returns:
        np.ndarray: Final predictions on X_test
    """
    # Your code here
    n, _ = X_train.shape
    preds = [clf(X_train, y_train, X_test) for clf in base_classifiers]
    h = np.vstack(preds).T
    preds = meta_classifier(X_train, y_train, h)
    return preds