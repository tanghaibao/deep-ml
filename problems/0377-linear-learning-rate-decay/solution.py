import numpy as np

def linear_lr_decay(initial_lr: float, end_lr: float, num_steps: int) -> list:
    """
    Generate a linear learning rate decay schedule.
    
    Args:
        initial_lr: Starting learning rate
        end_lr: Final learning rate
        num_steps: Total number of training steps
    
    Returns:
        List of learning rates for each step
    """
    # Your code here
    rates = initial_lr + (end_lr - initial_lr) * (np.arange(num_steps) / (num_steps - 1 + 1e-12))
    return np.round(rates, 4).tolist()