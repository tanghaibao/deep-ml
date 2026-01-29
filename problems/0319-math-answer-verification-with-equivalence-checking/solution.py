import re
import math

def verify_math_answer(predicted: str, ground_truth: str, tolerance: float = 1e-6) -> bool:
    """
    Verify if two mathematical answers are equivalent.
    
    Args:
        predicted: The predicted answer string
        ground_truth: The ground truth answer string
        tolerance: Numerical tolerance for comparison
    
    Returns:
        True if answers are equivalent, False otherwise
    """
    def normalize(s):
        s = s.replace("sqrt", "math.sqrt").replace("pi", "math.pi").replace("e", "math.e")
        try:
            return eval(s)
        except:
            return None

    a = normalize(predicted)
    b = normalize(ground_truth)
    if not a or not b:
        return False
    if abs(a - b) < tolerance:
        return True
    return False