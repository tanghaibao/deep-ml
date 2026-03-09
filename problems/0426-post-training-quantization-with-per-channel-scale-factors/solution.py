import numpy as np

def per_channel_quantize(weight: np.ndarray, bits: int = 8) -> tuple:
    """
    Perform symmetric per-channel post-training quantization.

    Args:
        weight: Weight matrix of shape (out_channels, in_features)
        bits: Target bit-width for quantization (default: 8)

    Returns:
        Tuple of (quantized_weights, scale_factors, dequantized_weights)
        - quantized_weights: int array of shape (out_channels, in_features)
        - scale_factors: float array of shape (out_channels,)
        - dequantized_weights: float array of shape (out_channels, in_features)
    """
    Q = 2 ** (bits - 1) - 1
    eps = 1e-10
    M = np.abs(weight).max(axis=-1, keepdims=True)
    M = np.maximum(M, eps)
    S = M / Q 
    quantized_weights = (weight / S).round().clip(-Q, Q).astype(int) 
    scale_factors = S[:, 0]
    scale_factors[scale_factors < eps] = 1
    dequantized_weights = quantized_weights * S
    return quantized_weights, scale_factors, dequantized_weigh