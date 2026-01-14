def calculate_sla_metrics(requests: list, latency_sla_ms: float = 100.0) -> dict:
    """
    Calculate SLA compliance metrics for a model serving endpoint.
    
    Args:
        requests: list of request results, each a dict with 'latency_ms' and 'status'
        latency_sla_ms: maximum acceptable latency in ms for SLA compliance
    
    Returns:
        dict with keys: 'latency_sla_compliance', 'error_rate', 'overall_sla_compliance'
        All values as percentages (0-100), rounded to 2 decimal places.
    """
    if not requests:
        return {}
    a = len([r for r in requests if r["status"] == "success" and r["latency_ms"] <= latency_sla_ms])
    b = len([r for r in requests if r["status"] == "success"])
    c = len(requests)
    latency_sla_compliance = a / b if b else 0.0 
    error_rate = (c - b) / c if c else 0.0
    overall_sla_compliance = a / c if c else 0.0
    def f(x):
        return round(x * 100, 2)
    return {
        "latency_sla_compliance": f(latency_sla_comp