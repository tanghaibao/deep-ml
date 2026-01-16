import numpy as np

def check_positive_definite(matrix: list) -> dict:
    """
    Check if a matrix is positive definite and compute its eigenvalues.
    
    Args:
        matrix: A 2D list representing a square matrix
        
    Returns:
        dict with 'is_positive_definite' (bool) and 'eigenvalues' (list of floats sorted ascending)
    """
    # matrix = np.array(matrix)
    # # Your code here
    # def is_pd(matrix):
    #     k = len(matrix)
    #     for i in range(k):
    #         m = matrix[:i, :i]
    #         if np.linalg.det(m) <= 0:
    #             return False
    #     return True

    ev, _ = np.linalg.eig(matrix)
    return {
        "is_positive_definite": np.all(ev > 0), 
        "eigenvalues": sorted(np.round(ev, 4))
    }