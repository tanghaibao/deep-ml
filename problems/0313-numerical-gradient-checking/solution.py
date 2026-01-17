import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    # Your code here
    def ff(f, x, i, epsilon):
        x = np.array(x)
        x[i] += epsilon
        return f(x)
    n = len(x)
    grad = np.zeros_like(x)
    for i in range(n):
        grad[i] = (ff(f, x, i, epsilon) - ff(f, x, i, -epsilon)) / (2 * epsilon)
    relative_error = np.linalg.norm(grad - analytical_grad) / (np.linalg.norm(grad) + np.linalg.norm(analytical_grad))
    return np.round(grad, 4), relative_error