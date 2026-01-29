import numpy as np

def sobel_edge_detection(image):
    """
    Apply Sobel edge detection to a grayscale image.
    
    Args:
        image: 2D list/array representing a grayscale image
               with values in range [0, 255]
    
    Returns:
        Edge magnitude image as 2D list with integer values (0-255),
        or -1 if input is invalid
    """
    # Your code here
    X = np.array(image)
    m, n = X.shape
    if m < 3 or n < 3 or np.any((X < 0) | (X > 255)):
        return -1
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = Kx.T
    Gx = np.zeros_like(X, dtype=float)
    Gy = np.zeros_like(X, dtype=float)
    X = np.pad(image, (1, 1))
    for i in range(m):
        for j in range(n):
            Gx[i, j] = np.sum(X[i:i+3, j:j+3] * Kx)
            Gy[i, j] = np.sum(X[i:i+3, j:j+3] * Ky)
    G = np.sqrt(Gx ** 2 + Gy ** 2)[1:-1, 1:-1]
    return (G / (G.max() - 1e-10) * 255).astype(int)