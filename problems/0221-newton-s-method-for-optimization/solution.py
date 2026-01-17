import torch
from torch.autograd.functional import hessian, jacobian
from typing import Callable

def newtons_method_optimization(
    loss_func: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    tol: float = 1e-6,
    max_iter: int = 100
) -> torch.Tensor:
    """
    Find the minimum of a function using Newton's method with PyTorch autograd.
    Handles arbitrary input shapes by flattening/unflattening.
    """
    # Preserve original shape to return correctly later
    original_shape = x0.shape
    
    # Flatten initial guess to 1D vector. 
    # Newton's method operates on a vector of parameters.
    x = x0.clone().flatten()

    # Wrapper allows loss_func to accept a flat vector but calculate on original shape
    def flat_loss_func(x_flat):
        return loss_func(x_flat.view(original_shape))

    for i in range(max_iter):
        # Calculate Jacobian (Gradient) and Hessian
        # Note: Hessian calculation is O(N^2) relative to input size
        g = jacobian