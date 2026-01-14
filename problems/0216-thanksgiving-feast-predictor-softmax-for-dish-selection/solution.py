import torch

def thanksgiving_dish_predictor(preference_scores: torch.Tensor) -> torch.Tensor:
    """
    Predict the probability of choosing each Thanksgiving dish using softmax.
    
    Args:
        preference_scores: Tensor of preference scores for each dish
        
    Returns:
        Tensor of probabilities for each dish
    """
    # Your code here
    p = preference_scores - preference_scores.max()
    return torch.exp(p) / torch.sum(torch.exp(p))