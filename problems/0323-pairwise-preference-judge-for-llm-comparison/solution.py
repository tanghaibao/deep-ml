import numpy as np

def pairwise_preference_judge(comparisons: list, criteria_weights: dict, tie_threshold: float) -> dict:
    """
    Analyze pairwise comparisons between LLM responses.
    
    Args:
        comparisons: List of comparison dicts with 'id', 'scores_a', 'scores_b'
        criteria_weights: Dict mapping criterion names to importance weights
        tie_threshold: Maximum difference to declare a tie
    
    Returns:
        Dict with 'results', 'win_rate_a', 'win_rate_b', 'tie_rate', 'avg_margin'
    """
    names, weights = zip(*criteria_weights.items())
    weights = np.array(weights)

    # Your code here
    def S(scores: dict):
        s = np.array([scores.get(k, 0) for k in names])
        return (s * weights).sum() / weights.sum()
    
    results = []
    if not comparisons:
        return {
            "results": results,
            "win_rate_a": 0.0,
            "win_rate_b": 0.0,
            "tie_rate": 0.0,
            "avg_margin": 0.0,
        }

    n_c