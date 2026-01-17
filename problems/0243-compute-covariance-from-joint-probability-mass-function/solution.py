import numpy as np

def covariance_from_joint_pmf(x_values: list, y_values: list, joint_pmf: np.ndarray) -> float:
    """
    Compute the covariance of X and Y from their joint PMF.
    
    Args:
        x_values: List of possible values for X
        y_values: List of possible values for Y
        joint_pmf: 2D numpy array where joint_pmf[i][j] = P(X=x_values[i], Y=y_values[j])
    
    Returns:
        Covariance of X and Y as a float
    """
    # Your code here
    p_x = joint_pmf.sum(axis=1)
    p_y = joint_pmf.sum(axis=0)
    e_x = (np.array(x_values) * p_x).sum()
    e_y = (np.array(y_values) * p_y).sum()
    e_x_y = (np.outer(x_values, y_values) * joint_pmf).sum()
    return e_x_y - e_x * e_y