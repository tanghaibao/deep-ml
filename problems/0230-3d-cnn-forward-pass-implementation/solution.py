import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3d_forward_pass_pytorch(
    input_volume: torch.Tensor,
    kernel_weight: torch.Tensor,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int] = (0, 0, 0)
) -> torch.Tensor:
	"""
	Perform 3D convolution forward pass using PyTorch.
	
	Use PyTorch's nn.Conv3d layer to perform 3D convolution.
	Note: PyTorch expects input shape (N, C, D, H, W) where N is batch size.
	
	Args:
		input_volume: Shape (C, D, H, W)
		kernel_weight: Shape (1, C, kD, kH, kW) 
		  First dim is output channels (1 for simplified version)
		stride: (stride_d, stride_h, stride_w)
		padding: (pad_d, pad_h, pad_w)
	
	Returns:
		Output tensor: Shape (1, D_out, H_out, W_out)
		
	Hints:
		- Add batch dimension to input: input.unsqueeze(0)
		- Create Conv3d layer with kernel size from kernel_weight
		- Set conv.weight parameter to kernel_weight
		- Set conv.bias to None or zeros
		- Apply convolution and remove batch dimens