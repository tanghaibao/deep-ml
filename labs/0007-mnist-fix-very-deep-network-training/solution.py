import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.BatchNorm1d(size)
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.net(x) + x)

class DeepNetwork(nn.Module):
    '''
    A very deep network that trains poorly!
    
    This 30-layer network achieves only ~20% accuracy due to:
    - Vanishing gradients
    - No skip connections
    - No normalization
    - Poor gradient flow
    
    Your task: Fix this network to achieve 90%+ accuracy while keeping ≥30 layers!
    
    You can:
    - Add skip/residual connections (ResNet style)
    - Add dense connections (DenseNet style)
    - Add batch normalization or layer normalization
    - Change activation functions
    - Reorganize the architecture
    - Add any architectural improvements you know
    
    You CANNOT:
    - Reduce the number of layers below 30
    '''
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        
        # Input projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        
        # 30 hidden layers (this is the deep part that needs fixing!)
        self.layers = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(15)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, num_classes)
        
        # Activation
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # Input projection
        x = self.input_layer(x)
        
        # Pass through 30 layers (problematic without skip connections!)
        for layer in self.layers:
            x = layer(x)
        
        # Output
        x = self.output_layer(x)
        
        return x
