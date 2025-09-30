import numpy as np

def cramers_rule(A, b):
    # Your code here
    A = np.array(A)
    b = np.array(b)
    res = []
    m, n = A.shape
    det_A = np.linalg.det(A)
    if det_A == 0: return -1
    for i in range(n):
        a_i = A[:, i].copy()
        A[:, i] = b
        det_Ai = np.linalg.det(A)
        res.append(det_Ai)
        A[:, i] = a_i
    return np.array(res) / det_A
