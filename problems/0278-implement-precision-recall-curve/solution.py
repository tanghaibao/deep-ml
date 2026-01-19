import numpy as np

def precision_recall_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute precision-recall pairs for different probability thresholds.
    
    Args:
        y_true: List of true binary labels (0 or 1)
        y_scores: List of predicted probabilities or confidence scores
    
    Returns:
        Tuple of (precisions, recalls, thresholds) where each is a list
    """
    # Your code here
    y_true = np.array(y_true, dtype=bool)
    y_scores = np.array(y_scores)
    
    # Get unique thresholds sorted descending
    thresholds = np.unique(y_scores)[::-1]
    
    precisions = []
    recalls = []
    
    for thresh in thresholds:
        # Create boolean mask for positive predictions
        pos = y_scores >= thresh
        
        # Calculate stats
        tp = np.sum(y_true & pos)
        fp = np.sum(~y_true & pos)
        fn = np.sum(y_true & ~pos)
        
        # Precision: Handle division by zero (when no positive predictions)
        if (tp +