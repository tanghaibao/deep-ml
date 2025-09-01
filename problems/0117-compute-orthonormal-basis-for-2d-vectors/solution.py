import numpy as np

def orthonormal_basis(vectors: list[list[float]], tol: float = 1e-10) -> list[np.ndarray]:
    # Your code here
    n = len(vectors)
    vectors = np.array(vectors, dtype="float")
    v1 = vectors[0]
    norm_v1 = np.linalg.norm(v1)
    uu = []
    if norm_v1 < tol:
        return uu
    uu.append(v1 / norm_v1)
    for k in range(1, n):
        vk = vectors[k]
        wk = vk
        for j in range(0, k):
            wk -= np.dot(vk, uu[j]) * uu[j]
        norm_wk = np.linalg.norm(wk)
        if norm_wk < tol:
            break
        uk = wk / norm_wk
        uu.append(uk)
    return uu
