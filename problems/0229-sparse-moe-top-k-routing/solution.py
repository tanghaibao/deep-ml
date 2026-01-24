import torch

def moe_topk_routing(
    router_logits: torch.Tensor,
    expert_outputs: torch.Tensor,
    k: int
) -> torch.Tensor:
    """
    Perform top-k expert routing for a Mixture-of-Experts layer.
    
    For each token:
    1. Select the top-k experts based on router_logits
    2. Compute softmax weights over only the selected experts
    3. Return weighted combination of the selected expert outputs
    
    Args:
        router_logits: Shape (batch_size, num_experts)
                      Raw scores from the router for each expert
        expert_outputs: Shape (batch_size, num_experts, hidden_dim)
                       Output from each expert for each input
        k: Number of experts to select per token
        
    Returns:
        Shape (batch_size, hidden_dim) - weighted combination of expert outputs
    """
    # Your code here
    sorted_logits, sorted_idx = torch.sort(router_logits, dim=1, descending=True, stable=True)
    topk_logits = sorted_logits[:, :k]   # (B,