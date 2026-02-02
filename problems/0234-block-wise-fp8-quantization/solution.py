import torch

def fp8_block_quantize(
    tensor: torch.Tensor,
    block_size: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to FP8-E4M3 format using block-wise scaling.
    
    Args:
        tensor: Input tensor of shape (N,) where N is divisible by block_size
        block_size: Number of elements per quantization block
        
    Returns:
        quantized: Quantized values of shape (N,), clipped to [-448, 448]
        scales: Per-block scale factors of shape (N // block_size,)
    """
    # Your code here
    t = tensor.reshape(-1, block_size)
    quantized = torch.zeros_like(t)
    m, _ = t.max(dim=1)
    scales = m / 448
    quantized = (t / scales).round()
    return quantized.reshape(-1), scales


def fp8_block_dequantize(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = 128
) -> torch.Tensor:
    """
    Dequantize FP8-E4M3 values back to full precision.
    
    Args:
        quantized: Quantized values of shap