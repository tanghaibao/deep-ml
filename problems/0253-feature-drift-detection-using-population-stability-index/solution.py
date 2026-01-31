import numpy as np

def detect_feature_drift(reference_data: list, production_data: list, num_bins: int = 10) -> dict:
    """
    Detect feature drift using Population Stability Index (PSI).
    
    Args:
        reference_data: List of feature values from reference distribution (e.g., training)
        production_data: List of feature values from production distribution
        num_bins: Number of bins for histogram comparison
    
    Returns:
        dict with 'psi', 'drift_detected', and 'drift_level'
    """
    if not reference_data:
        return {}
    ref = np.asarray(reference_data, dtype=float)
    prod = np.asarray(production_data, dtype=float)
    lo = min(ref.min(), prod.min())
    hi = max(ref.max(), prod.max())
    edges = np.linspace(lo, hi, num_bins + 1)
    reference, _ = np.histogram(reference_data, bins=edges)
    production, _ = np.histogram(production_data, bins=edges)
    eps = 0.0001
    E = reference / reference.sum()
    A = production / production.sum()
 