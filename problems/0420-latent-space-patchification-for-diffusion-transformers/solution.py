import numpy as np

def patchify_latent(latent: np.ndarray, patch_size: int, proj_weight: np.ndarray = None, proj_bias: np.ndarray = None) -> np.ndarray:
    """
    Convert a latent representation into a sequence of patch tokens
    for a Diffusion Transformer.

    Args:
        latent: Latent tensor of shape (C, H, W)
        patch_size: Size of each square patch (p)
        proj_weight: Optional projection matrix of shape (C*p*p, D)
        proj_bias: Optional projection bias of shape (D,)

    Returns:
        tokens: Array of shape (num_patches, patch_dim) or (num_patches, D)
                if projection is applied
    """
    C, H, W = latent.shape
    p = patch_size
    nh = H // p
    nw = W // p
    latent = latent.reshape(C, nh, p, nw, p)
    latent = latent.transpose((1, 3, 0, 2, 4))
    out = latent.reshape(nh * nw, C * p * p)
    if proj_weight is not None:
        out = out @ proj_weight
    if proj_bias is not None:
        out = out + proj_bias 
    return out