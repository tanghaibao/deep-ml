import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def reward_model_loss(chosen_rewards: list, rejected_rewards: list, margin: float = 0.0) -> dict:
    """
    Compute reward model loss from pairwise human preferences.
    
    Args:
        chosen_rewards: Reward scores for preferred responses
        rejected_rewards: Reward scores for non-preferred responses
        margin: Minimum desired gap between chosen and rejected scores
    
    Returns:
        Dictionary with 'loss' (float) and 'accuracy' (float)
    """
    chosen_rewards = np.array(chosen_rewards)
    rejected_rewards = np.array(rejected_rewards)
    loss = -np.log(sigmoid(chosen_rewards - rejected_rewards - margin)).mean()
    accuracy = (chosen_rewards > rejected_rewards).mean()
    return {
        "loss": round(loss, 4),
        "accuracy": round(accuracy, 4)
    }
    