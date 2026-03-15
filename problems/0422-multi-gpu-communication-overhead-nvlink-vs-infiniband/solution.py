import math

def multi_gpu_comm_overhead(message_size_gb: float, num_gpus: int,
                           nvlink_config: dict, ib_config: dict,
                           operation: str) -> dict:
    """
    Compare communication overhead between NVLink and InfiniBand
    for collective GPU operations.
    
    Args:
        message_size_gb: Data payload size in GB
        num_gpus: Number of GPUs
        nvlink_config: Dict with 'bandwidth_gbps' and 'latency_us'
        ib_config: Dict with 'bandwidth_gbps' and 'latency_us'
        operation: One of 'all_reduce', 'all_gather', 'broadcast'
    
    Returns:
        Dict with timing comparison and bottleneck analysis
    """
    N = num_gpus
    M = message_size_gb
    def calc_times(config, operation):
        # t1 is latency, t2 is bandwidth
        if operation == "all_reduce":
            S = 2 * (N - 1)
            D = M / N
        elif operation == "all_gather":
            S = N - 1
            D = M / N
        else:
            S = math.ceil(math.log2(N))
            D = M
        t1 = S * config["latency_us"] / 1000
        t2 = S * D / config["bandwidth_gbps"] * 1000
        total = t1 + t2
        bottleneck = "bandwidth" if t2 > t1 else "latency" 
        return total, bottleneck
    
    nvlink_time_ms, nvlink_bottleneck = calc_times(nvlink_config, operation)
    ib_time_ms, ib_bottleneck = calc_times(ib_config, operation)
    speedup = ib_time_ms / nvlink_time_ms
    _ = lambda x: round(x, 4)
    return {
            "nvlink_time_ms": _(nvlink_time_ms),
            "nvlink_bottleneck": nvlink_bottleneck,
            "speedup": _(speedup),
            "ib_time_ms": _(ib_time_ms),
            "ib_bottleneck": ib_bottleneck,
            }