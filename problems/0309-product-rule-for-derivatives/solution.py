import numpy as np
from numpy.polynomial import polynomial

def product_rule_derivative(f_coeffs: list, g_coeffs: list) -> list:
    """
    Compute the derivative of the product of two polynomials.
    
    Args:
        f_coeffs: Coefficients of polynomial f, where f_coeffs[i] is the coefficient of x^i
        g_coeffs: Coefficients of polynomial g, where g_coeffs[i] is the coefficient of x^i
    
    Returns:
        Coefficients of (f*g)' as a list of floats rounded to 4 decimal places
    """
    # Your code here
    f, g = len(f_coeffs), len(g_coeffs)
    def deriv(po):
        dd = np.arange(len(po))
        return (po * dd)[1:]
    f_coeffs = np.array(f_coeffs)
    g_coeffs = np.array(g_coeffs)
    fd = deriv(f_coeffs)
    gd = deriv(g_coeffs)
    def prod(a, b):
        if len(a) == 0:
            a = [0]
        if len(b) == 0:
            b = [0]
        return polynomial.polymul(a, b) 
    ans = prod(fd, g_coeffs) + prod(f_coeffs, gd)
    return ans