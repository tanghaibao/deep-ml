import numpy as np

def calculate_batch_health(predictions: list, confidence_threshold: float = 0.5) -> dict:
    """
    Calculate health metrics for a batch prediction job.
    
    Args:
        predictions: list of prediction results, each a dict with 'status' and optionally 'confidence'
        confidence_threshold: threshold below which a prediction is considered low confidence
    
    Returns:
        dict with keys: 'success_rate', 'avg_confidence', 'low_confidence_rate'
        All values as percentages (0-100), rounded to 2 decimal places.
    """
    if not predictions:
        return {}
    success_rate = sum([x["status"] == "success" for x in predictions]) / len(predictions) * 100
    successes = [x for x in predictions if x["status"] == "success"]
    avg_confidence = np.mean([x["confidence"] for x in successes]) * 100 if successes else 0.0
    low_confidence_rate = np.sum([x["confidence"] < confidence_threshold for x in successes]) / len(successes) * 100 if successes el