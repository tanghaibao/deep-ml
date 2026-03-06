import math

def compute_visual_tokens(image_height: int, image_width: int, patch_size: int,
                          max_resolution: int = None, add_cls_token: bool = True,
                          padding_strategy: str = 'pad') -> dict:
    """
    Compute the number of visual tokens a VLM produces for a given image.

    Args:
        image_height: Original image height in pixels.
        image_width: Original image width in pixels.
        patch_size: Side length of each square patch in pixels.
        max_resolution: If set, resize so longest side equals this value.
        add_cls_token: Whether a [CLS] token is prepended.
        padding_strategy: 'pad' to pad to patch multiple, 'truncate' to drop remainders.

    Returns:
        Dictionary with effective_height, effective_width, patches_h,
        patches_w, num_patches, total_tokens.
    """
    if max_resolution:
        s = max_resolution / max(image_height, image_width)
        image_height = round(image_height * s)
    