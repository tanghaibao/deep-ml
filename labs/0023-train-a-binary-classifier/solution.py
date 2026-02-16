import numpy as np
# You can import any sklearn module you need
from sklearn.linear_model import LogisticRegression 

def train(X_train, y_train, X_val, y_val):
    """
    Train a binary classifier.
    
    Args:
        X_train: numpy array of shape (n_samples, 30) -- standardized features
        y_train: numpy array of shape (n_samples,) -- binary labels (0 or 1)
        X_val:   numpy array of shape (n_val, 30) -- standardized
        y_val:   numpy array of shape (n_val,) -- validation labels
    
    Returns:
        predict: callable that takes X (n, 30) and returns y_pred (n,) of 0s and 1s
    """
    # TODO: train a model and return a predict function
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)    
    def predict(X):
        return clf.predict(X)
    return predict
