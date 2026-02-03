import numpy as np

def bilinear_resize(image, new_height: int, new_width: int) -> list:
    """
    Resize an image using bilinear interpolation.
    
    Args:
        image: 2D (grayscale) or 3D (RGB) array representing an image
        new_height: Target height of the resized image
        new_width: Target width of the resized image
    
    Returns:
        Resized image as a nested list with values rounded to 2 decimal places
    """
    # Your code here
    image = np.array(image, dtype=float)
    height, width = image.shape[:2]
    sy = height / new_height
    sx = width / new_width
    yy, xx = np.meshgrid(range(new_height), range(new_width))
    yy = yy.T.astype(float) * sy
    xx = xx.T.astype(float) * sx
    y0 = yy.astype(int)
    x0 = xx.astype(int)
    dy = yy - y0
    dx = xx - x0
    y1 = np.minimum(y0 + 1, height - 1)
    x1 = np.minimum(x0 + 1, width - 1)
    if len(image.shape) == 3:
        dy = dy[..., None]
        dx = dx[..., None]
    R1 = image[y0, x0] * (1 