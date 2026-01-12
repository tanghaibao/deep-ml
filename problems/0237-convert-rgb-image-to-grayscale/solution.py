import numpy as np

def rgb_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminosity method.
    
    Args:
        image: RGB image as list or numpy array of shape (H, W, 3)
               with values in range [0, 255]
    
    Returns:
        Grayscale image as 2D list with integer values,
        or -1 if input is invalid
    """
    # Write your code here
    image = np.array(image)
    if len(image.shape) != 3:
        return -1
    h, w, c = image.shape
    if c != 3 or h == 0 or w == 0:
        return -1
    if np.any(image < 0) or np.any(image > 255):
        return -1
    return np.round(image @ np.array((0.299, 0.587, 0.114))).astype(int)