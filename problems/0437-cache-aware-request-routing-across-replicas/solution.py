import numpy as np

def cache_aware_route(replicas: list, requests: list, alpha: float = 0.7, beta: float = 0.3) -> dict:
    """
    Route inference requests to model replicas based on cache affinity and load.
    
    Args:
        replicas: list of dicts with keys:
            'cached_prefixes': list of lists (cached token sequences)
            'active_requests': int (current load)
            'max_capacity': int (max concurrent requests)
        requests: list of dicts with key:
            'tokens': list of ints (token sequence)
        alpha: weight for cache affinity (default 0.7)
        beta: weight for load penalty (default 0.3)
    
    Returns:
        dict with 'assignments', 'cache_hit_rates', 'avg_cache_hit_rate', 'load_distribution'
    """
    def match(a, b):
        score = 0
        for i, j in zip(a, b):
            if i == j:
                score += 1
            else:
                break
        return score

    def match_all(a, b):
        return max([match