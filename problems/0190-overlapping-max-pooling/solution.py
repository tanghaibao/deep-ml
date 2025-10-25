import numpy as np

def overlapping_max_pool2d(x: np.ndarray, k: int = 3, s: int = 2) -> np.ndarray:
    """
    Applies overlapping max pooling to a 4D tensor (N, C, H, W).

    Args:
        x: Input array of shape (N, C, H, W)
        kernel_size: Size of pooling window (int)
        stride: Stride between pooling windows (int)

    Returns:
        A 4D tensor after overlapping pooling.
    """
    # Your code here
    n, c, h, w = x.shape
    # 4 -> 2, 3 -> 2
    p = h // s 
    q = w // s
    if s == 1:
        p -= 1
        q -= 1
    res = np.zeros((n, c, p, q))
    for a in range(n):
        for b in range(c):
            for i in range(p):
                for j in range(q):
                    res[a, b, i, j] = np.max(x[a, b, i * s: i * s + k, j * s: j * s + k])
    return res