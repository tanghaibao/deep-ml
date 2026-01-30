import numpy as np

def monitor_prediction_distribution(reference_preds: list, current_preds: list, n_bins: int = 10) -> dict:
    """
    Monitor prediction distribution changes between reference and current predictions.
    
    Args:
        reference_preds: List of reference prediction scores (floats between 0 and 1)
        current_preds: List of current prediction scores (floats between 0 and 1)
        n_bins: Number of bins for histogram comparison
    
    Returns:
        Dictionary with keys: 'mean_shift', 'std_ratio', 'js_divergence', 'drift_detected'
    """
    # Your code here
    reference = np.array(reference_preds)
    current = np.array(current_preds)
    mean_shift = current.mean() - reference.mean()
    std_ratio = current.std() / reference.std()
    
    def pdf(preds):
        bins, _ = np.histogram(preds, bins=n_bins, range=(0, 1))
        smoothed = (bins + 1) / (bins.sum() + n_bins)
        return smoothed
    
    P = pdf(reference)
    Q = pdf(current)
    e