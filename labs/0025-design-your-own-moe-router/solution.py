import numpy as np

def softmax(x):
    ex = np.exp(x - x.max(axis=-1, keepdims=True))
    return ex / ex.sum(axis=-1, keepdims=True)

def route(x, W_gate):
    '''
    Route inputs to experts in a Mixture of Experts layer.
    
    Args:
        x: numpy array of shape (batch_size, hidden_dim)
           Hidden activations from the previous layer
        W_gate: numpy array of shape (hidden_dim, num_experts)
           Learnable gate weight matrix (32 experts)
    
    Returns:
        numpy array of shape (batch_size, num_experts)
        Per-expert weights that are:
        - Non-negative
        - Sum to ~1.0 per sample (row)
        - Based on the input x and gate weights W_gate
    
    The harness combines expert outputs as:
        output = sum(weights[:, i] * expert_i(x) for each expert i)
    
    Each expert is small (128->16), so routing quality matters!
    With 32 experts, naive strategies that spread weight everywhere
    perform poorly. The key insight from modern LLMs is SPARSE
    routing: only activate a few experts per input.
    
    Hint: Compute softmax scores, keep only the top-k experts,
    zero out the rest, and renormalize.
    '''
    batch_size = x.shape[0]
    num_experts = W_gate.shape[1]
    
    # TODO: Implement your routing strategy
    
    top_k = 4
    scores = softmax(x @ W_gate) # (batch, n_experts)
    zz = np.argpartition(scores, num_experts - top_k, axis=-1)[:, num_experts - top_k]
    scores[:, zz] = 0
    weights = scores / scores.sum(axis=-1, keepdims=True)
    return weights
