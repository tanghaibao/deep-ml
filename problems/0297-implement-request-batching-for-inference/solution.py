def batch_requests(requests: list, max_batch_size: int, max_wait_time: float) -> list:
    """
    Group inference requests into batches based on size and time constraints.
    
    Args:
        requests: List of dicts with 'id', 'timestamp', 'features'
        max_batch_size: Maximum number of requests per batch
        max_wait_time: Maximum time to wait before processing a batch
    
    Returns:
        List of tuples: (request_ids, batched_features, process_time)
    """
    # Your code here
    batches = []
    request_ids, batched_features, t0 = [], [], 0
    for request in sorted(requests, key=lambda x: x["timestamp"]):
        request_id = request["id"]
        request_timestamp = request["timestamp"]
        request_features = request["features"]
        if len(request_ids) >= max_batch_size or request_timestamp - t0 > max_wait_time:
            batches.append((request_ids, batched_features, process_time))
            request_ids, batched_features, t0 = [], [], request_times