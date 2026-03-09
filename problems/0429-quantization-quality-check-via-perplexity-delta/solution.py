import numpy as np

def quantization_quality_check(log_probs_original: list, log_probs_quantized: list, thresholds: dict = None) -> dict:
    """
    Evaluate quantization quality by comparing perplexity before and after quantization.
    
    Args:
        log_probs_original: Token-level log-probabilities from the original (full precision) model
        log_probs_quantized: Token-level log-probabilities from the quantized model
        thresholds: Optional dict with 'excellent' and 'acceptable' percentage thresholds
    
    Returns:
        dict with perplexity metrics and quality classification
    """
    def pp(L):
        L = np.array(L)
        return np.exp(-L.mean())

    perplexity_original = pp(log_probs_original)
    perplexity_quantized = pp(log_probs_quantized)
    perplexity_delta = abs(perplexity_original - perplexity_quantized)
    relative_delta_percent = perplexity_delta / perplexity_original * 100
    if relative_delta_percent <= 1:
        quality = "excellent"
   