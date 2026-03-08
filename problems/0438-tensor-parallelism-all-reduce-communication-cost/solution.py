def tensor_parallel_allreduce_cost(
    hidden_size: int,
    sequence_length: int,
    batch_size: int,
    num_gpus: int,
    num_layers: int,
    bytes_per_element: int = 2,
    bandwidth_gb_per_sec: float = 300.0,
    allreduces_per_layer: int = 2
) -> dict:
    """
    Calculate the all-reduce communication cost for tensor parallelism.

    Args:
        hidden_size: Hidden dimension of the model
        sequence_length: Sequence length
        batch_size: Micro-batch size per GPU
        num_gpus: Number of GPUs in tensor parallel group
        num_layers: Number of transformer layers
        bytes_per_element: Bytes per element (2 for FP16, 4 for FP32)
        bandwidth_gb_per_sec: Interconnect bandwidth in GB/s
        allreduces_per_layer: Number of all-reduce ops per layer (forward pass)

    Returns:
        Dictionary with communication cost analysis
    """
    B = batch_size
    S = sequence_length
    H = hidden_size
    b = bytes_per_element
    M = B * S * H * b
    P 