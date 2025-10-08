import numpy as np

class DropoutLayer:
    def __init__(self, p: float):
        """Initialize the dropout layer."""
        # Your code here
        self.p = p

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass of the dropout layer."""
        # Your code here
        if training:
            self.mask = np.random.binomial(n=1, p=1 - self.p, size=x.shape)
            return x * self.mask / (1 - self.p)
        else:
            return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass of the dropout layer."""
        # Your code here
        return grad * self.mask / (1 - self.p)