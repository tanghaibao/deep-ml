import numpy as np

def calculate_auc(y_true, y_scores):
    """
    Calculate the Area Under the ROC Curve (AUC).
    
    Args:
        y_true: List or array of binary ground truth labels (0 or 1)
        y_scores: List or array of predicted probabilities or confidence scores
        
    Returns:
        AUC value as a float
    """
    y_true = np.array(y_true, dtype=bool)
    y_scores = np.array(y_scores)
    def get_tpr_fpr(thresh):
        pos = y_scores >= thresh
        tp = (y_true & pos).sum()
        tn = (~y_true & ~pos).sum()
        fp = (~y_true & pos).sum()
        fn = (y_true & ~pos).sum()
        return tp / (tp + fn), fp / (fp + tn)
    tpr_, fpr_ = 0, 0
    auc = 0
    # Your code here
    for thresh in sorted(set(y_scores), reverse=True):
        tpr, fpr = get_tpr_fpr(thresh)
        auc += (tpr + tpr_) / 2 * (fpr - fpr_)
        tpr_, fpr_ = tpr, fpr
    return auc