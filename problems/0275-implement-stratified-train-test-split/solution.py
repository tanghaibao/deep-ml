import numpy as np

def stratified_train_test_split(X, y, test_size, random_seed=None):
    """
    Split data into train and test sets while maintaining class proportions.
    
    Args:
        X: Feature matrix of shape (n_samples, n_features)
        y: Label vector of shape (n_samples,)
        test_size: Proportion of data for test set (0 < test_size < 1)
        random_seed: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    np.random.seed(random_seed)
    n_samples, n_features = X.shape
    train_idx = []
    test_idx = []
    for klass in np.unique(y):
        klass_idx = np.arange(n_samples, dtype=int)[y == klass]
        klass_idx = np.random.permutation(klass_idx)
        klass_n = len(klass_idx)
        klass_test = int(klass_n * test_size)
        klass_train = klass_n - klass_test
        train_idx += list(klass_idx[:klass_train])
        test_idx += list(klass_idx[klass_train:])
    train_idx.sort()
    test_idx.sort()