import torch


def sgtm_step(
    params: torch.Tensor,
    grad: torch.Tensor,
    forget_mask: torch.Tensor,
    lr: float = 0.1,
    batch_type: str = "forget",
) -> torch.Tensor:
    """Perform one SGTM update step on a 1D parameter tensor.

    See the NumPy version for full details.

    Args:
        params: Current parameters, shape (d,)
        grad: Gradient for this batch, shape (d,)
        forget_mask: Mask for forget parameters, shape (d,)
        lr: Learning rate
        batch_type: One of {'forget', 'retain', 'unlabeled'}

    Returns:
        new_params: Updated parameters as a tensor
    """
    # Your implementation here (mirror the NumPy version using torch ops).
    if batch_type == "forget":
        grad[~forget_mask] = 0
    elif batch_type == "retain":
        grad[forget_mask] = 0
    params -= lr * grad
    return params