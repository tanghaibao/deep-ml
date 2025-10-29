import torch

def flow_epe_torch(pred, gt, mask=None, max_flow=None) -> torch.Tensor:
    """
    Compute mean End-Point Error (EPE) between predicted and ground-truth optical flow.

    Args:
        pred, gt: tensors or array-likes of shape (H, W, 2).
        mask: optional tensor/array broadcastable to (H, W); 1=include, 0=ignore.
        max_flow: optional float to clip per-pixel EPE.

    Returns:
        torch.Tensor scalar. Returns tensor(-1.0) on invalid input or if no valid pixels.
    """
    p = torch.as_tensor(pred, dtype=torch.float32)
    g = torch.as_tensor(gt,   dtype=torch.float32)

    # Your implementation here
    if mask is not None:
        mask = (mask > 0) & ~torch.isnan(p) & ~torch.isnan(g)
    else:
        mask = ~torch.isnan(p) & ~torch.isnan(g)
    epe = torch.sqrt((p[:, :, 0] - g[:, :, 0]) ** 2 + (p[:, :, 1] - g[:, :, 1]) ** 2)
    if max_flow is not None:
        epe = torch.minimum(epe, torch.tensor(max_flow, dtype=epe.dtype))    
    epe_mean = torch.ma