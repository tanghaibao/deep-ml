import numpy as np

def zero_pad_image(img, pad_width):
    """
    Add zero padding around a grayscale image.
    
    Args:
        img: 2D list or numpy array of pixel values
        pad_width: integer number of pixels to pad on each side
    
    Returns:
        Padded image as 2D list with integer values,
        or -1 if input is invalid
    """
    # Write your code here
    img = np.array(img)
    p = pad_width
    if len(img.shape) != 2 or pad_width < 0:
        return -1
    h, w = img.shape
    ret = np.zeros((h + 2 * p, w + 2 * p))
    ret[p : h + p, p : w + p] = img
    return ret