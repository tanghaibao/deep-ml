import numpy as np

def expert_parallel_comm_cost(num_devices: int, num_experts: int, tokens_per_device: list, token_size_bytes: int, bandwidth_gbps: float = None) -> dict:
    """
    Analyze communication costs for expert-parallel MoE token routing.
    
    Args:
        num_devices: Number of devices (GPUs)
        num_experts: Total number of experts (evenly divided across devices)
        tokens_per_device: tokens_per_device[d] is a list of expert IDs for tokens on device d
        token_size_bytes: Bytes per token embedding
        bandwidth_gbps: Optional interconnect bandwidth in GB/s
    
    Returns:
        Dict with dispatch matrix, communication costs, and load analysis
    """
    D = num_devices
    E_per_device = num_experts // D 
    M = np.zeros((D, D), dtype=int)
    for i, tokens in enumerate(tokens_per_device):
        for e in tokens:
            M[i, e // E_per_device] += 1
    MA = M.copy()
    np.fill_diagonal(MA, 0)
    total_comm_bytes = MA.sum() * token_siz