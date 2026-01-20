import numpy as np

class MixedPrecision:
    def __init__(self, loss_scale=1024.0):
        # Initialize loss scaling factor
        self.loss_scale = loss_scale
    
    def forward(self, weights, inputs, targets):
        # Perform forward pass with float16, return scaled loss as float32
        preds = inputs.astype(np.float16) @ weights.astype(np.float16)
        mse = ((targets.astype(np.float16) - preds) ** 2).mean()        
        return float(mse * self.loss_scale)
    
    def backward(self, gradients):
        # Unscale gradients and check for overflow, return as float32
        if any(np.isnan(gradients)) or any(np.isinf(gradients)):
            gradients = np.zeros_like(gradients)
        return gradients.astype(np.float32) / self.loss_scale