import numpy as np

def flip_image(image, direction):
    """
    Flip an image horizontally or vertically.
    
    Args:
        image: 2D or 3D list/array representing a grayscale or RGB image
        direction: string, either 'horizontal' or 'vertical'
    
    Returns:
        Flipped image as a nested list, or -1 if input is invalid
    """
    # Your code here
    if not image:
        return -1
    image = np.array(image)
    if direction == "horizontal":
        return image[:, ::-1]
    elif direction == "vertical":
        return image[::-1, :]
    return -1