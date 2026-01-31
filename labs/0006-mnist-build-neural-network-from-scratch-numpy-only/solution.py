import numpy as np

class NeuralNetwork:
    '''
    Build a neural network from scratch using only NumPy.
    Required architecture: 784 → 128 (ReLU) → 10 (Softmax)
    '''
    def __init__(self, input_size=784, hidden_size=128, output_size=10, lr=0.01):
        '''
        Initialize network parameters.
        Use small random initialization (e.g., Xavier/He initialization).
        '''
        self.lr = lr
        # TODO: Initialize weights and biases
        # Hint: Use np.random.randn() with proper scaling
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros(output_size)        
    
    def relu(self, x):
        return np.maximum(x, 0)

    def softmax(self, x):
        z = x - x.max(axis=1, keepdims=True)
        ex = np.exp(z)
        return ex / (ex.sum(axis=1, keepdims=True) + 1e-12)

    def forward(self, X):
        '''
        Forward pass through the network.
        
        Args:
            X: Input batch, shape (N, 784)
        
        Returns:
            probs: Class probabilities, shape (N, 10)
        
        Must cache intermediate values for backward pass!
        '''
        # TODO: Implement forward pass
        # Layer 1: X @ W1 + b1, then ReLU
        # Layer 2: hidden @ W2 + b2, then Softmax
        W1, b1, W2, b2 = self.W1, self.b1, self.W2, self.b2
        z1 = X @ W1 + b1  # (N, 128)
        h1 = self.relu(z1) # (N, 128)
        z2 = h1 @ W2 + b2  # (N, 10)
        self.cache = {"X": X, "z1": z1, "h1": h1, "z2": z2}
        return self.softmax(z2) # (N, 10)
    
    def backward(self, X, y, probs):
        '''
        Backward pass - compute gradients for all parameters.
        
        Args:
            X: Input batch, shape (N, 784)
            y: True labels, shape (N,)
            probs: Predicted probabilities from forward pass, shape (N, 10)
        
        Returns:
            loss: Scalar cross-entropy loss
        
        Must update self.W1, self.b1, self.W2, self.b2 using computed gradients!
        '''
        # TODO: Implement backward pass
        # 1. Compute loss
        N, = y.shape
        loss = -np.log(probs[np.arange(N), y] + 1e-12).mean()
        # {"X": X, "z1": z1, "h1": h1, "z2": z2}
        X, z1, h1, z2 = self.cache["X"], self.cache["z1"], self.cache["h1"], self.cache["z2"]
        W1, b1, W2, b2 = self.W1, self.b1, self.W2, self.b2

        # 2. Compute gradient of loss w.r.t. softmax output
        z2_g = probs.copy() # (N, 10)
        z2_g[np.arange(N), y] -= 1
        z2_g /= N

        # 3. Backprop through linear layer 2
        W2_g = h1.T @ z2_g # (128, 10)
        b2_g = z2_g.sum(axis=0) # (10,)
        h1_g = z2_g @ W2.T # (N, 128)

        # 4. Backprop through ReLU
        z1_g = h1_g * (z1 > 0) # (N, 128)

        # 5. Backprop through linear layer 1
        W1_g = X.T @ z1_g  # (784, 128)
        b1_g = z1_g.sum(axis=0)  # (128,)
        
        # 6. Update all parameters using gradients
        self.W1 -= self.lr * W1_g
        self.b1 -= self.lr * b1_g
        self.W2 -= self.lr * W2_g
        self.b2 -= self.lr * b2_g

        return loss
    
    def train_step(self, X, y):
        '''
        Complete training step: forward + backward + update.
        
        Args:
            X: Input batch, shape (N, 784)
            y: True labels, shape (N,)
        
        Returns:
            loss: Scalar loss value
        '''
        probs = self.forward(X)
        loss = self.backward(X, y, probs)
        return loss
    
    def predict(self, X):
        '''
        Predict class labels.
        
        Args:
            X: Input batch, shape (N, 784)
        
        Returns:
            predictions: Predicted class labels, shape (N,)
        '''
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
