import torch

class MyTransform:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (1, 28, 28) float tensor in [0,1]
        Return: transformed tensor, same shape/dtype.
        Must be non-identity and deterministic.
        """
        # TODO: implement your custom transformation logic here
        x = (x + torch.randn_like(x) * 2 + 0.5).clamp(-3, 3)
        return x
