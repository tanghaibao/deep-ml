import numpy as np

def calculate_latency_percentiles(latencies: list[float]) -> dict[str, float]:
    """
    Calculate P50, P95, and P99 latency percentiles.
    
    Args:
        latencies: List of latency measurements
    
    Returns:
        Dictionary with keys 'P50', 'P95', 'P99' containing
        the respective percentile values rounded to 4 decimal places
    """
    if not latencies:
        return {
            "P50": 0.0, "P95": 0.0, "P99": 0.0
        }
    # Your code here
    def f(x):
        return round(x, 4)
    return {
        "P50": f(np.percentile(latencies, 50)),
        "P95": f(np.percentile(latencies, 95)),
        "P99": f(np.percentile(latencies, 99)),
    }