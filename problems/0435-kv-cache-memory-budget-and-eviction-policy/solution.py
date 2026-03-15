from heapq import heappop, heappush

def kv_cache_manager(num_layers: int, num_heads: int, head_dim: int, dtype_bytes: int, memory_budget_bytes: int, token_ids: list, token_scores: list, eviction_policy: str, num_protected: int = 0) -> dict:
    """
    Simulate a KV cache with memory budget and eviction policy.
    
    Args:
        num_layers: Number of transformer layers
        num_heads: Number of attention heads per layer
        head_dim: Dimension per attention head
        dtype_bytes: Bytes per element (e.g., 2 for FP16)
        memory_budget_bytes: Total KV cache memory budget in bytes
        token_ids: List of token IDs arriving sequentially
        token_scores: List of importance scores per token
        eviction_policy: 'fifo' or 'score'
        num_protected: Number of initial tokens protected from eviction
    
    Returns:
        Dictionary with bytes_per_token, max_tokens, final_cache,
        evicted_tokens, and num_evictions
    """
    bytes_per_token = 2 * num_layers * num_heads * head_dim * dtype_bytes
    max_tokens = memory_budget_bytes // bytes_per_token
    cache = [] 
    evicted_tokens = []
    num_evictions = 0
    protected = token_ids[:num_protected]
    t = num_protected
    for token, token_score in zip(token_ids[num_protected:], token_scores[num_protected:]):
        if len(cache) == max_tokens - num_protected:
            _, _, evicted = heappop(cache)
            evicted_tokens.append(evicted)
            num_evictions += 1
        if eviction_policy == "fifo":
            heappush(cache, (0, t, token))
        else:
            heappush(cache, (token_score, t, token))
        t += 1
    cache.sort(key=lambda x: x[1])
    final_cache = protected + [token_ids[t] for (_, t, _) in cache]
    return {
            "bytes_per_token": bytes_per_token,
            "max_tokens": max_tokens,
            "final_cache": final_cache,
            "evicted_tokens": evicted_tokens,
            "num_evictions": num_evictions,
            }