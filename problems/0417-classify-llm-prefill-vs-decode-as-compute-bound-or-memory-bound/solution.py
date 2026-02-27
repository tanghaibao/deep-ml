def classify_llm_phases(num_params: int, sequence_length: int, batch_size: int, bytes_per_param: int, peak_flops: float, peak_bandwidth: float) -> dict:
    """
    Analyze prefill and decode phases of LLM inference using the Roofline Model.

    Args:
        num_params: Total number of model parameters
        sequence_length: Number of input tokens processed during prefill
        batch_size: Number of sequences processed in parallel during decode
        bytes_per_param: Memory footprint per parameter (e.g., 2 for FP16)
        peak_flops: Hardware peak compute throughput (FLOP/s)
        peak_bandwidth: Hardware peak memory bandwidth (bytes/s)

    Returns:
        Dictionary containing ridge_point and analysis dicts for 'prefill' and 'decode',
        each with total_flops, memory_bytes, arithmetic_intensity, bottleneck,
        achieved_flops, and utilization_percent.
    """
    ridge_point = peak_flops / peak_bandwidth
    def get_stats(total_flops):
        memory_bytes = num