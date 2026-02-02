import torch, torch.nn as nn

def build_model() -> nn.Module:
    """
    Return a tiny nn.Module for MNIST classification (10 classes).
    IMPORTANT: If total trainable params > 2048, final accuracy will be set to 0.
    Tip: Consider very small convs, global average pooling, and tiny linear head.
    """
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=True),  # 80 params
                nn.ReLU(),
                nn.MaxPool2d(2),                                       # 28->14

                nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=True), # 1168 params
                nn.ReLU(),

                nn.AdaptiveAvgPool2d(1),                               # -> (B,16,1,1)
            )
            self.head = nn.Linear(16, 10, bias=True)                           # 170 params

        def forward(self, x):
            x = self.features(x)
            x = x.flatten(start_dim=1)
            x = self.head(x)
            return x  # logits

    return TinyNet()
