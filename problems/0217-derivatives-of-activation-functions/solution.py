import torch
import torch.nn as nn

def activation_derivatives(x: float) -> dict[str, float]:
    """
    Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x
    using PyTorch autograd.
    
    Args:
        x: Input value
        
    Returns:
        Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
    """
    # Your code here - use autograd!
    # Hint: Create tensors with requires_grad=True, apply activation, call .backward()
    def f(x, m):
        x = torch.tensor(x, requires_grad=True)
        y = m(x)
        y.backward()
        return x.grad.item()
    return {
        "sigmoid": f(x, nn.Sigmoid()),
        "tanh": f(x, nn.Tanh()),
        "relu": f(x, nn.ReLU())
    }
