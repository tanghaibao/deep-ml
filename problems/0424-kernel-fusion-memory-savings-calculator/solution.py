def kernel_fusion_savings(input_elements: int, operations: list, dtype_bytes: int = 4, memory_bandwidth_gbps: float = None) -> dict:
    """
    Calculate memory traffic savings from fusing a chain of GPU kernel operations.
    
    Args:
        input_elements: Number of elements in the initial input tensor
        operations: List of dicts with 'output_elements' and optional 'extra_param_elements'
        dtype_bytes: Bytes per element (default: 4 for float32)
        memory_bandwidth_gbps: Optional GPU memory bandwidth in GB/s
    
    Returns:
        Dict with memory traffic analysis and optional timing estimates
    """
    T_unfused = 0
    total_params = 0
    inp = input_elements
    for op in operations:
        output_elements = op["output_elements"]
        extra_param_elements = op.get("extra_param_elements", 0)
        T_unfused += (inp + extra_param_elements + output_elements) * dtype_bytes
        total_params += extra_param_elements
        inp = output_elements
    T_