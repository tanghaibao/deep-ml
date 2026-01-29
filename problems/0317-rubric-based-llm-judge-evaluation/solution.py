import numpy as np

def rubric_llm_judge_evaluation(
    judge_scores: list[list[float]],
    criteria_weights: list[float],
    passing_threshold: float = 0.6,
    max_score: float = 5.0
) -> dict:
    """
    Evaluate LLM response using rubric-based multi-judge scoring.
    
    Args:
        judge_scores: 2D list where judge_scores[i][j] is judge i's score for criterion j
        criteria_weights: Weights for each criterion (should sum to 1)
        passing_threshold: Minimum normalized score to pass (0 to 1)
        max_score: Maximum possible score for each criterion
    
    Returns:
        Dictionary with evaluation results
    """
    scores = np.array(judge_scores)
    criterion_scores = scores.mean(axis=0)
    weighted_score = np.sum(np.array(criteria_weights) * criterion_scores)
    normalized_score = weighted_score / max_score
    pass_status = normalized_score >= passing_threshold
    criterion_std = scores.std(axis=0, ddof=0)
    judge_agreement = 1 - criterion_std.mean(