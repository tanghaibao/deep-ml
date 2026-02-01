import numpy as np

def neural_memory_update(
    M: np.ndarray,
    S: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    theta: float = 0.1,
    eta: float = 0.9,
    alpha: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """
    Update neural memory using surprise-based learning with momentum and forgetting.
    
    Args:
        M: Current memory state matrix of shape (d, d)
        S: Current momentum/surprise accumulator of shape (d, d)
        k: Key vector of shape (d,)
        v: Value vector of shape (d,)
        theta: Learning rate for momentary surprise (default: 0.1)
        eta: Momentum decay factor for past surprise (default: 0.9)
        alpha: Forget gate - fraction of old memory to forget (default: 0.01)
    
    Returns:
        Tuple of (updated_M, updated_S) where:
        - updated_M: New memory state after update
        - updated_S: New momentum state after update
    
    """
    grad = (M @ k - v)[:, None] @ k.T[None, :]  # (d, d))
    updated_S = e