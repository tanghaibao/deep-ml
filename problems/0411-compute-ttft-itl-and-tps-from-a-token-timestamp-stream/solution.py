import numpy as np

def compute_inference_metrics(timestamps: list[float]) -> dict:
    """
    Compute LLM inference performance metrics from token timestamps.
    
    Args:
        timestamps: List of floats where timestamps[0] is the request start time
                    and timestamps[1:] are the times when each output token was generated.
    
    Returns:
        Dictionary with keys 'ttft', 'tps', 'itl' containing the metric values.
    """
    n = len(timestamps)
    timestamps = np.array(timestamps)
    tps = (n - 1) / (timestamps[-1] - timestamps[0])
    if n <= 2:
        itl = 0
        ttft = timestamps[1] - timestamps[0]
    else:
        itl = (timestamps[1:] - timestamps[:-1])
        ttft = itl[0]
        itl = itl[1:].mean()
    return {
            "ttft": ttft,
            "tps": tps,
            "itl": itl,
            }