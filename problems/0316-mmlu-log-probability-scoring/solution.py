import numpy as np

def mmlu_log_prob_score(log_probs: list, correct_answers: list) -> dict:
    """
    Compute MMLU-style log-probability scoring metrics.
    
    Args:
        log_probs: List of lists, where each inner list contains 
                   log-probabilities for each answer choice
        correct_answers: List of correct answer indices (0-indexed)
    
    Returns:
        Dictionary with 'accuracy', 'predictions', and 'avg_correct_prob'
    """
    N = len(log_probs)
    log_probs = np.array(log_probs)
    truths = np.array(correct_answers)
    predictions = np.argmax(log_probs, axis=-1)
    probs = np.exp(log_probs) / np.exp(log_probs).sum(axis=-1, keepdims=True)
    accuracy = (predictions == truths).mean()
    avg_correct_prob = probs[np.arange(N), truths].mean()
    return {
        "accuracy": round(accuracy, 4),
        "predictions": list(predictions),
        "avg_correct_prob": round(avg_correct_prob, 4),
    }