import numpy as np
from itertools import combinations

def davies_bouldin_index(X, labels):
    """
    Calculate the Davies-Bouldin Index for clustering evaluation.
    
    Parameters:
    X: numpy array of shape (n_samples, n_features) - data points
    labels: numpy array of shape (n_samples,) - cluster labels
    
    Returns:
    float: Davies-Bouldin Index rounded to 4 decimal places
    """
    # Your code here
    mus = {}
    S = {}
    ll = [int(x) for x in np.unique(labels)]
    n_clusters = len(ll)
    if n_clusters == 1:
        return 0.0
    for label in ll:
        cl = X[labels == label]
        mus[label] = cl.mean(axis=0)
        s = np.linalg.norm(cl - mus[label], axis=1).mean()
        S[label] = s
    R = {}
    for i, j in combinations(ll, 2):
        R[(i, j)] = R[(j, i)] = (S[i] + S[j]) / np.linalg.norm(mus[i] - mus[j])
    RR = np.zeros(n_clusters) 
    for i in ll:
        RR[i] = max(R[(i, j)] for j in ll if i != j)
    return np.round(RR.mean(), 4)
       