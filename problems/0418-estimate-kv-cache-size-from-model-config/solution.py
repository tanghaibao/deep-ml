def estimate_kv_cache_size(model_config: dict, batch_size: int, seq_len: int) -> dict:
    """
    Estimate the KV cache memory footprint for a Transformer model.

    Args:
        model_config: Dictionary with model architecture parameters
        batch_size: Number of sequences in the batch
        seq_len: Number of cached tokens

    Returns:
        Dictionary with cache size estimates
    """
    # Your code here
    # Your code here
    B = batch_size 
    S = seq_len
    L = model_config["num_layers"]
    nh = model_config["num_attention_heads"]
    n_kv = model_config.get("num_kv_heads", nh)
    d_head = model_config["hidden_size"] / nh
    dtype_bytes = model_config["dtype_bytes"]
    kv_cache_elements = int(B * L * 2 * n_kv * d_head * S)  
    kv_cache_size_bytes = int(kv_cache_elements * dtype_bytes)
    _ = lambda x: round(x, 4)
    kv_cache_size_mb = _(kv_cache_size_bytes / 1024 / 1024)
    per_layer_size_mb = _(kv_cache_size_mb / L)
    per_token_size_kb = _(kv_cache_si