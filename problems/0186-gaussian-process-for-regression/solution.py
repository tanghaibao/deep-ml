import math  # ---------------------------------------- utf-8 encoding ---------------------------------

# This file contains Gaussian Process implementation.
import numpy as np
import math


def matern_kernel(x: np.ndarray, x_prime: np.ndarray, length_scale=1.0, nu=1.5):
    r = np.linalg.norm(x - x_prime)
    return (1 + (math.sqrt(3) * r) / length_scale) * math.exp(
        -(math.sqrt(3) * r) / length_scale
    )


def rbf_kernel(x: np.ndarray, x_prime, sigma=1.0, length_scale=1.0):
    r = np.linalg.norm(x - x_prime)
    return sigma**2 * math.exp(-0.5 * (r**2) / (length_scale**2))


def periodic_kernel(
    x: np.ndarray, x_prime: np.ndarray, sigma=1.0, length_scale=1.0, period=1.0
):
    r = np.linalg.norm(x - x_prime)
    return sigma**2 * math.exp(
        -2 * (math.sin(math.pi * r / period) ** 2) / (length_scale**2)
    )


def linear_kernel(x: np.ndarray, x_prime: np.ndarray, sigma_b=1.0, sigma_v=1.0):
    return sigma_b**2 + sigma_v**2 * np.dot(x, x_prime)


def rational_