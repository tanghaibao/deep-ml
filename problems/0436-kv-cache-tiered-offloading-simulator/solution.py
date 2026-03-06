def kv_cache_offload_sim(tiers: list[dict], entries: list[dict], recompute_latency_us: float = 1000.0) -> dict:
    """
    Simulate tiered KV cache offloading for LLM inference.

    Args:
        tiers: list of dicts with keys 'name' (str), 'capacity' (int), 'latency_us' (float)
        entries: list of dicts with keys 'id' (int), 'recency' (float)
        recompute_latency_us: latency penalty for evicted entries

    Returns:
        dict with 'placement', 'evicted', 'total_latency_us', 'avg_latency_us'
    """
    # Your code here
    entries.sort(key=lambda x: -x["recency"])
    tiers.sort(key=lambda x: x["latency_us"])
    placement = {} 
    cur = 0
    lat = {}
    for tier in tiers:
        lat[tier["name"]] = tier["latency_us"]
        take = min(len(entries) - cur, tier["capacity"])
        placement[tier["name"]] = [x["id"] for x in entries[cur: cur+take]]
        cur += take
    evicted = [x["id"] for x in entries[cur:]]
    total_latency_us = sum(len(v) * lat[k] for (k, v