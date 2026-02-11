import numpy as np

def optimizer_step(param, grad, state, lr):
    '''
    Update a parameter using its gradient.
    
    Args:
        param: numpy array - current parameter values (any shape)
        grad: numpy array - gradient of loss w.r.t. param (same shape)
        state: dict - persists between calls, use to store any needed values
                      Example: state = {'step': 5, 'momentum': np.array([...])}
                      First call: state = {} (empty dict)
        lr: float - learning rate
    
    Returns:
        new_param: numpy array - updated parameter (must be same shape as param)
        state: dict - updated state dictionary
    '''
    # TODO: Implement your optimizer update rule
    # Hint: Think about gradient descent and how to use the gradient to update the parameter
    
    m = state.get("m", np.zeros_like(param))
    v = state.get("v", np.zeros_like(param))
    t = state.get("t", 0)
    beta1 = 0.9
    beta2 = 0.99
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    t += 1
    mh = m / (1 - beta1 ** t)
    vh = v / (1 - beta2 ** t)
    new_param = param - lr / (np.sqrt(vh) + 1e-8) * mh
    state["m"] = m
    state["v"] = v
    state["t"] = t
    return new_param, state
