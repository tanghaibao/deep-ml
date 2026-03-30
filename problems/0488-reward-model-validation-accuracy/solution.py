import numpy as np

def reward_model_validation(
    chosen_scores: np.ndarray,
    rejected_scores: np.ndarray,
    margin_thresholds: list[float]
) -> dict:
    """
    Compute validation metrics for a reward model on held-out preference pairs.

    Args:
        chosen_scores: 1D array of reward scores for preferred responses.
        rejected_scores: 1D array of reward scores for rejected responses.
        margin_thresholds: List of thresholds for margin-based accuracy.

    Returns:
        Dictionary with 'accuracy', 'mean_margin', 'concordance', 'margin_accuracy'.
    """
    accuracy = (chosen_scores > rejected_scores).mean()
    mean_margin = (chosen_scores - rejected_scores).mean()
    W = (chosen_scores > rejected_scores).sum()
    T = (chosen_scores == rejected_scores).sum()
    N = len(chosen_scores)
    concordance = (W + 0.5 * T) / N
    _ = lambda x: round(float(x), 4)
    margin_accuracy = {t: _((chosen_scores - rejected_scores >= t).mean()) for t in margin_thresholds}
    return {
            "accuracy": _(accuracy),
            "mean_margin": _(mean_margin),
            "concordance": _(concordance),
            "margin_accuracy": margin_accuracy,
            }