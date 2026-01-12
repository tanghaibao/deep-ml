import string
import numpy as np

def exact_match_score(predictions: list[str], references: list[str]) -> float:
    """
    Calculate the exact match score between predictions and references.
    
    Args:
        predictions: List of predicted strings
        references: List of reference (ground truth) strings
    
    Returns:
        Exact match score as a float between 0 and 1
    """
    if not predictions or not references:
        return 0.0
    def f(x):
        cleaned = ''.join(char.lower() for char in x.strip() if char.isalnum() or char == " ")
        return " ".join(cleaned.split())
    # Your code here
    return np.mean([float(f(a) == f(b)) for a, b in zip(predictions, references)])