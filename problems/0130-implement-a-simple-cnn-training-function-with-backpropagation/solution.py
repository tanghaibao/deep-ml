import torch

def train_simple_cnn_with_backprop(X: torch.Tensor, y: torch.Tensor, epochs: int, learning_rate: float, kernel_size: int = 3, num_filters: int = 1) -> tuple:
    '''
    Trains a simple CNN with one convolutional layer, ReLU activation, flattening, and a dense layer with softmax output using backpropagation.

    Assumes X has shape (n_samples, height, width) for grayscale images and y is one-hot encoded with shape (n_samples, num_classes).

    Parameters:
    X : torch.Tensor, input data
    y : torch.Tensor, one-hot encoded labels
    epochs : int, number of training epochs
    learning_rate : float, learning rate for weight updates
    kernel_size : int, size of the square convolutional kernel
    num_filters : int, number of filters in the convolutional layer

    Returns:
    W_conv, b_conv, W_dense, b_dense : Trained weights and biases for the convolutional and dense layers
    '''
    X = X.double()
    y = y.double()
    
    n_samples, height, width = X.shape
  