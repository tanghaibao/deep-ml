import numpy as np
from numpy.polynomial import polynomial

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x)
    """
    # Your code here
    g = np.array(g_coeffs, dtype=float)[::-1]
    h = np.array(h_coeffs, dtype=float)[::-1]
    gd = polynomial.polyder(g)
    hd = polynomial.polyder(h)
    def ev(f, x):
        return polynomial.polyval(x, f) 
    return (ev(gd, x) * ev(h, x) - ev(g, x) * ev(hd, x)) / ev(h, x) ** 2