import numpy as np
def sparse_window_attention(Q, K, V, w, scale_factor=None):
    # Your code here
    seq_len, d_k = Q.shape
    scale_factor = scale_factor or np.sqrt(d_k)
    _, d_v = V.shape
    res = np.zeros((seq_len, d_v))
    for i in range(seq_len):
        start = max(0, i - w)
        end = min(seq_len - 1, i + w) + 1
        q = Q[i].reshape((1, -1)) # (1, d_k)
        Kp = K[start:end]  # (2w + 1, d_k)
        Vp = V[start:end]  # (2w + 1, d_v)
        scores = q @ Kp.T / scale_factor
        attention = np.exp(scores) / np.exp(scores).sum()
        output = attention @ Vp
        res[i] = output.T
    return res