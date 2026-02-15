import numpy as np

def softmax(X):
    ex = np.exp(X)
    return ex / ex.sum(axis=-1, keepdims=True)

def moe_load_balancing_loss(gate_logits, num_experts, alpha=0.01):
    """
    Compute the load balancing auxiliary loss for a Mixture of Experts layer.
    
    Args:
        gate_logits: numpy array of shape (num_tokens, num_experts), raw gating scores
        num_experts: int, number of experts
        alpha: float, scaling coefficient for the loss
    
    Returns:
        float: load balancing loss rounded to 4 decimal places
    """
    T, N = gate_logits.shape
    P = softmax(gate_logits)
    A = P.argmax(axis=-1)
    S = np.zeros_like(P)
    S[np.arange(T), A] = 1
    
    f = S.mean(axis=0)
    p = P.mean(axis=0)
    
    loss = alpha * N * np.sum(f * p)
    return round(float(loss), 4)