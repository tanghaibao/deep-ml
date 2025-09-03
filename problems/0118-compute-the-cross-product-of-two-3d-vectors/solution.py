import numpy as np

def cross_product(a, b):
    # Your code here
    a1, a2, a3 = a
    b1, b2, b3 = b
    return [a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1]