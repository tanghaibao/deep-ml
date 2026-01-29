import numpy as np
from collections import Counter 

def mode(x):
    c, _ = Counter(x).most_common(1)[0]
    return c

def impute_missing_data(data: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    """
    Impute missing values in a 2D array using the specified strategy.
    
    Args:
        data: 2D numpy array with missing values represented as np.nan
        strategy: Imputation strategy - 'mean', 'median', or 'mode'
        
    Returns:
        2D numpy array with missing values imputed
    """
    # Your code here
    m, n = data.shape
    for j in range(n):
        col = data[:, j]
        if strategy == "mean":
            imp = np.nanmean(col)
        elif strategy == "median":
            imp = np.nanmedian(col)
        else:
            imp = mode(col)
        data[:, j][np.isnan(col)] = imp 
    return data 