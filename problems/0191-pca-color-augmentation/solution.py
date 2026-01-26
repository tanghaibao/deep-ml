import numpy as np

def pca_color_augmentation(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Apply PCA color augmentation to an RGB image.
    
    Args:
        image: RGB image of shape (H, W, 3) with values in [0, 255]
        alpha: Array of 3 random coefficients for principal components
    
    Returns:
        Augmented image of shape (H, W, 3) with values clamped to [0, 255]
    """
    # Your code here
    X = image.astype(np.float32).reshape(-1, 3)
    Xb = X - X.mean(axis=0)
    n, _ = Xb.shape
    C = Xb.T @ Xb / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    distortion = eigvecs @ (np.sqrt(eigvals) * alpha)
    X_aug = X + distortion
    return X_aug.reshape(image.shape).clip(0, 255)