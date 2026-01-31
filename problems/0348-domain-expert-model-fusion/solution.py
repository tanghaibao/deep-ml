from collections import defaultdict

def fuse_domain_experts(
    expert_scores: dict[str, dict[str, float]],
    domain_weights: dict[str, float],
    fusion_method: str
) -> float:
    """
    Fuse multiple domain expert models into a single score.
    
    Args:
        expert_scores: Dict of expert_name -> {domain: score}
        domain_weights: Dict of domain -> importance weight
        fusion_method: 'weighted_average' or 'best_per_domain'
    
    Returns:
        Overall fused score (weighted by domain importance)
    """
    best = defaultdict(int)
    for expert in expert_scores.values():
        for k, score in expert.items():
            best[k] = max(best[k], score)
    return sum(w * best[d] for d, w in domain_weights.items())