import numpy as np

def relu(X):
    return np.maximum(X, 0)
    
def dense_net_block(input_data, num_layers, growth_rate, kernels, kernel_size=(3, 3)):
    # Your code here
    k, _ = kernel_size
    padding = (k - 1) // 2
    X = input_data
    for kernel in kernels:
        Xr = relu(X)
        N, H, W, c = Xr.shape
        kh, kw, c, g = kernel.shape
        Xp = np.zeros((N, H + 2 * padding, W + 2 * padding, c))
        Xp[:, padding:-padding, padding:-padding, :] = Xr
        out = np.zeros((N, H, W, g))
        for i in range(H):
            for j in range(W):
                x = Xp[:, i:i+kh, j:j+kw, :, None]
                out[:, i, j] = np.sum(x * kernel)
        X = np.concatenate((X, out), axis=-1)
    return X