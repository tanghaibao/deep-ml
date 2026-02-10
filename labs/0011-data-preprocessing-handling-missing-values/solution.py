import numpy as np

def impute(X: np.ndarray) -> np.ndarray:
    '''
    Fill in missing values (NaN) in the input array.
    
    Args:
        X: Array with possible NaN values, shape (n_samples, n_features)
    
    Returns:
        X_clean: Array with no NaN values, same shape as X
    '''
    X_clean = X.copy()
    
    # TODO: Fill in NaN values
    n, k = X.shape
    X_clean = X.copy()
    for j in range(k):
        col = X[:, j]
        X_clean[np.isnan(col), j] = np.nanmean(col)
    return X_clean
