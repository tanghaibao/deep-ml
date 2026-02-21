import numpy as np

def moe_shared_expert_forward(
    X: np.ndarray,
    W_gate: np.ndarray,
    W_shared: np.ndarray,
    W_experts: list,
    top_k: int = 2
) -> dict:
    """
    Forward pass of a Mixture of Experts layer with a shared expert.
    
    Args:
        X: Input tokens, shape (num_tokens, d_model)
        W_gate: Gating weights, shape (d_model, num_routed_experts)
        W_shared: Shared expert weights, shape (d_model, d_out)
        W_experts: List of routed expert weight matrices, each (d_model, d_out)
        top_k: Number of routed experts per token
    
    Returns:
        Dictionary with keys: 'output', 'shared_output', 'routed_output',
                              'routing_indices', 'routing_weights'
    """
    # Your code here
    _, num_routed_experts = W_gate.shape
    logits = X @ W_gate # (num_tokens, num_routed_experts)
    gx = np.exp(logits) 
    gx /= gx.sum(axis=-1, keepdims=True)
    tk = np.argsort(gx, axis=-1)[:, -top_k:]
    g = np.take_along_a