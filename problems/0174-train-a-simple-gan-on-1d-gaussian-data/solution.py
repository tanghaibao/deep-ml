import torch
import torch.nn as nn
import torch.optim as optim

def train_gan(mean_real: float, std_real: float, 
              latent_dim: int = 1, hidden_dim: int = 16, 
              learning_rate: float = 0.001, epochs: int = 5000, 
              batch_size: int = 128, seed: int = 42):
    torch.manual_seed(seed)
    # Your PyTorch implementation here
    class Generator(nn.Module):
        def __init__(self, latent_dim, hidden_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        def forward(self, z):
            return self.net(z)

    class Discriminator(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            # IMPORTANT: output LOGITS (no sigmoid here)
            self.net = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Lin