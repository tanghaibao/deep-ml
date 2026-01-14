import numpy as np

def calculate_inference_stats(latencies_ms: list) -> dict:
    """
    Calculate inference statistics for model monitoring.
    
    Args:
        latencies_ms: list of latency measurements in milliseconds
    
    Returns:
        dict with keys: 'throughput_per_sec', 'avg_latency_ms', 'p50_ms', 'p95_ms', 'p99_ms'
        All values rounded to 2 decimal places.
    """
    if not latencies_ms:
        return {}
    avg_latency_ms = np.mean(latencies_ms)
    throughput_per_sec = 1000 / avg_latency_ms
    p50_ms = np.percentile(latencies_ms, 50)
    p95_ms = np.percentile(latencies_ms, 95)
    p99_ms = np.percentile(latencies_ms, 99)
    def f(x):
        return round(x, 2)
    return {
        "throughput_per_sec": f(throughput_per_sec),
        "avg_latency_ms": f(avg_latency_ms),
        "p50_ms": f(p50_ms),
        "p95_ms": f(p95_ms),
        "p99_ms": f(p99_ms),
    }