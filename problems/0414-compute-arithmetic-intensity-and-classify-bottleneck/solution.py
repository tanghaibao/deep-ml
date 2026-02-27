def compute_arithmetic_intensity(flops: float, bytes_accessed: float, peak_performance: float, peak_bandwidth: float) -> dict:
    """
    Analyze a computational kernel using the Roofline Model.
    
    Args:
        flops: Total floating-point operations of the kernel
        bytes_accessed: Total bytes transferred to/from memory
        peak_performance: Hardware peak compute throughput (FLOP/s)
        peak_bandwidth: Hardware peak memory bandwidth (bytes/s)
    
    Returns:
        Dictionary with arithmetic_intensity, ridge_point, bottleneck,
        achieved_performance, and utilization_percent
    """
    arithmetic_intensity = flops / bytes_accessed
    achieved_performance = min(peak_performance, peak_bandwidth * arithmetic_intensity)
    ridge_point = peak_performance / peak_bandwidth
    bottleneck = "memory-bound" if arithmetic_intensity < ridge_point else "compute-bound"
    utilization_percent = achieved_performance / peak_performance * 100
    _ = lambda x: round(floa