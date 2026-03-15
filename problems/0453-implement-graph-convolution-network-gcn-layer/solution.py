import numpy as np

def relu(x):
    return np.maximum(x, 0)


def gcn_layer(A: np.ndarray, X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Perform a single GCN layer forward pass.
    
    Args:
        A: Adjacency matrix of shape (N, N)
        X: Node feature matrix of shape (N, F_in)
        W: Weight matrix of shape (F_in, F_out)
        
    Returns:
        Output feature matrix of shape (N, F_out)
    """
    n, _ = A.shape
    Ah = A + np.eye(n)
    D2 = np.diag(Ah.sum(axis=-1) ** (-0.5))
    Ah = D2 @ Ah @ D2 
    return relu(Ah @ X @ W)