import numpy as np

def continuous_batching_sim(requests: list[dict], max_batch_size: int) -> dict:
    """
    Simulate continuous batching for LLM inference.
    
    Args:
        requests: List of dicts with 'arrival_time' (int) and 'tokens_needed' (int)
        max_batch_size: Maximum concurrent sequences in the batch
    
    Returns:
        Dict with 'total_time', 'avg_latency', 'avg_ttft', 'throughput'
    """
    requests.sort(key=lambda x: x["arrival_time"])
    t = 0
    i = 0
    que = set()
    while True:
        live = set()
        for q in que:
            if t < requests[q]["start"] + requests[q]["tokens_needed"]:
                live.add(q)
        que = live
        if not que and i < len(requests) and t < requests[i]["arrival_time"]:
            t = requests[i]["arrival_time"]
        while i < len(requests) and requests[i]["arrival_time"] <= t and len(que) < max_batch_size:
            requests[i]["start"] = t
            que.add(i)
            i += 1
        if 