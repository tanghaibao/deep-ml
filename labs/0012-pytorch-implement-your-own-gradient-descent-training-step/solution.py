import torch
import torch.nn as nn
import torch.nn.functional as F

def train_step(model, x_batch, y_batch, lr):
    """
    Perform ONE step of gradient descent training.
    
    This is the core of deep learning! Every training loop does:
    1. Forward pass: predictions = model(inputs)
    2. Compute loss: how wrong are we?
    3. Backward pass: compute gradients (derivatives) via chain rule
    4. Update step: move parameters in direction that reduces loss
    
    The math behind step 4 (gradient descent):
        new_param = old_param - learning_rate * gradient
        
    This works because the gradient points toward INCREASING loss,
    so we go the OPPOSITE direction to decrease it.
    
    Args:
        model: nn.Module - the neural network
        x_batch: input tensor, shape (batch_size, ...)
        y_batch: target labels, shape (batch_size,)
        lr: learning rate (step size)
    
    Returns:
        loss: float - the loss value for this batch
    
    Requirements:
        - Zero gradients before backward (or they accumulate!)
        - Use F.cross_entropy for classification loss
        - Update ALL model.parameters()
        - Use torch.no_grad() when modifying parameters
        - Return loss.item() to get Python float
    
    Hints:
        - model.parameters() gives all learnable parameters
        - param.grad contains gradient after backward()
        - param.grad.zero_() clears the gradient
        - loss.backward() computes all gradients
        - Use: with torch.no_grad(): param -= lr * param.grad
    """
    # TODO: Implement gradient descent step
    #
    # Step 1: Zero all gradients
    for param in model.parameters():
        if param.grad is not None:
            param.grad.zero_()

    # Step 2: Forward pass
    logits = model(x_batch)

    # Step 3: Compute loss (use F.cross_entropy)
    loss = F.cross_entropy(logits, y_batch)

    # Step 4: Backward pass (compute gradients)
    loss.backward()

    # Step 5: Update parameters with torch.no_grad()
    with torch.no_grad():
        for param in model.parameters():
            param -= lr * param.grad

    # Step 6: Return loss as float
    
    return loss.item()
