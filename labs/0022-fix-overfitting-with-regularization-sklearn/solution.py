import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV
# You can import any sklearn module you need

def train(X_train, y_train, X_val, y_val):
    """
    Train a regression model that generalizes well despite having
    MORE features than training samples (many are noise or redundant).
    
    WARNING: LinearRegression() WILL overfit here.
    - LinearRegression(): Train R² ≈ 1.0, Val R² ≈ -5.0
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
    # TODO: fit a regularized model and return a predict function
    # 
    # Ideas:
    #   - Ridge(alpha=??) or RidgeCV (auto-tunes alpha)
    #   - Lasso(alpha=??) or LassoCV (auto-tunes + feature selection)
    #   - ElasticNet (combines L1 + L2)
    #   - SelectKBest + Ridge pipeline
    #   - Any sklearn approach that handles overfitting!
    #
    # Plain LinearRegression() will NOT work here.
    model = ElasticNet(alpha=0.1, l1_ratio=0.5)
    model.fit(X_train, y_train)

    def predict(X):
        return model.predict(X)
    
    return predict
