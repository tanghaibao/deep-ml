import numpy as np

def cosine_annealing_warm_restarts(eta_max: float, eta_min: float, T_0: int, T_mult: int, total_epochs: int) -> list:
    """
    Compute learning rates using cosine annealing with warm restarts.
    
    Args:
        eta_max: Maximum learning rate
        eta_min: Minimum learning rate
        T_0: Number of epochs in first cycle
        T_mult: Cycle length multiplier after each restart
        total_epochs: Total number of epochs
    
    Returns:
        List of learning rates for each epoch, rounded to 4 decimal places
    """
    # Your code here
    etas = []
    Ti = T_0
    T_cur = 0
    for _ in range(total_epochs):
        eta = eta_min + 1/2 * (eta_max - eta_min) * (1 + np.cos(T_cur / Ti * np.pi)) 
        eta = round(float(eta), 4)
        if T_cur == Ti - 1:
            T_cur = 0
            Ti = int(T_mult * Ti) 
        else:
            T_cur += 1
        etas.append(eta)
    return etas