import numpy as np

def area(a):
    x1a, y1a, x2a, y2a = a
    return (x2a - x1a) * (y2a - y1a)


def iou(a, b):
    x1a, y1a, x2a, y2a = a
    x1b, y1b, x2b, y2b = b
    x1i, y1i, x2i, y2i = max(x1a, x1b), max(y1a, y1b), min(x2a, x2b), min(y2a, y2b)
    area_i = max(0, x2i - x1i) * max(0, y2i - y1i)
    res = area_i / (area(a) + area(b) - area_i)
    return res

def non_maximum_suppression(boxes, scores, iou_threshold):
    """
    Apply Non-Maximum Suppression (NMS) to bounding boxes.
    
    Args:
        boxes: Array-like of shape (N, 4) with boxes in format [x1, y1, x2, y2]
        scores: Array-like of shape (N,) with confidence scores
        iou_threshold: float, IoU threshold for suppression (0 to 1)
    
    Returns:
        List of indices of kept boxes, ordered by descending score
        Returns -1 for invalid inputs
    """
    # Your code here
    if not boxes:
        return []
    idx = np.argsort(scores).tolist()[::-1]
    res = [idx[0]]
    for b in range(1, len(id