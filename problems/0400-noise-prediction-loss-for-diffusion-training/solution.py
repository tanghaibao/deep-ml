import numpy as np

def noise_prediction_loss(x_0, alpha_bar, t, epsilon, epsilon_pred):
    """
    Compute the noisy samples and noise prediction MSE loss for diffusion model training.
    
    Args:
        x_0: Clean data samples, shape (B, D)
        alpha_bar: Cumulative noise schedule, shape (T,)
        t: Timestep indices for each sample, shape (B,)
        epsilon: True Gaussian noise, shape (B, D)
        epsilon_pred: Predicted noise from model, shape (B, D)
    
    Returns:
        tuple: (x_t, loss) where x_t has shape (B, D) and loss is a scalar float
    """
    x_t = np.sqrt(alpha_bar[t][:, None]) * x_0 + np.sqrt(1 - alpha_bar[t][:, None]) * epsilon
    loss = ((epsilon - epsilon_pred) ** 2).mean()
    return x_t, loss