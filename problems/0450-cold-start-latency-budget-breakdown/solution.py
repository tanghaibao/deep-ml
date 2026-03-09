def cold_start_breakdown(config: dict) -> dict:
    """
    Break down the cold start latency of an ML model serving instance.

    Args:
        config: Dictionary with deployment parameters.

    Returns:
        Dictionary with per-phase latencies, total, SLA status,
        budget usage percentage, and bottleneck phase name.
    """
    # Your code here
    runtime_init_ms = config["runtime_init_ms"]
    weight_load_ms = config["model_size_gb"] / config["storage_bandwidth_gbps"] * 1000
    compilation_ms = config["model_size_gb"] * config["compilation_ms_per_gb"]
    warmup_ms = config["warmup_batches"] * config["warmup_batch_ms"]    
    health_check_ms = config["health_check_ms"]
    total_ms = runtime_init_ms + weight_load_ms + compilation_ms + warmup_ms + health_check_ms
    meets_sla = total_ms <= config["sla_budget_ms"]
    budget_used_pct = total_ms / config["sla_budget_ms"] * 100
    _, bottleneck = max([(runtime_init_ms, "runtime_init_ms"),
                         (weight