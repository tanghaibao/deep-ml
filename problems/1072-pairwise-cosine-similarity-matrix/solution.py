import numpy as np

def pairwise_cosine_similarity(X):
    # Your code here
    n = len(X)
    ans = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ans[i, j] = pp(X[i], X[j])
    return ans

def pp(x, y):
    if np.linalg.norm(x) == 0 or np.linalg.norm(y) == 0:
        return 0
    return (np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))