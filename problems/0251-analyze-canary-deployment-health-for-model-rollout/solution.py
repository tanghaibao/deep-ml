import numpy as np

def analyze_canary_deployment(canary_results: list, baseline_results: list, accuracy_tolerance: float = 0.05, latency_tolerance: float = 0.10) -> dict:
    """
    Analyze canary deployment health metrics for model rollout decision.
    
    Args:
        canary_results: list of prediction results from canary (new) model
                       Each dict has 'latency_ms', 'prediction', 'ground_truth'
        baseline_results: list of prediction results from baseline (existing) model
                         Each dict has 'latency_ms', 'prediction', 'ground_truth'
        accuracy_tolerance: max acceptable relative accuracy degradation (0.05 = 5%)
        latency_tolerance: max acceptable relative latency increase (0.10 = 10%)
    
    Returns:
        dict with canary/baseline metrics and promotion recommendation
    """
    if not canary_results:
        return {}
    canary_accuracy = sum(d["prediction"] == d["ground_truth"] for d in canary_results) / len(canary_re