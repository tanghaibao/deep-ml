import torch
import torch.nn as nn

def generate_adversarial_example(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    criterion: nn.Module
) -> torch.Tensor:
    '''
    Generate an adversarial example for input x.
    
    Args:
        model: Pre-trained classifier (already in eval mode)
        x: Input image tensor, shape (1, 1, 28, 28), values in [0,1]
        y: True label, shape (1,) or scalar
        epsilon: L∞ perturbation budget
        criterion: Loss function (e.g., nn.CrossEntropyLoss())
    
    Returns:
        x_adv: Adversarial example, same shape as x, satisfying:
               - ||x_adv - x||_∞ ≤ epsilon
               - x_adv values in [0, 1]
               - model(x_adv).argmax() != y (ideally)
    '''
    # TODO: Implement your adversarial attack here
    # Hint: Use gradients of loss w.r.t. input x
    x = x.clone().detach().requires_grad_(True)
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    grad_sign = x.grad.sign()
    x_adv = x + epsilon * grad_sign
    x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv
