import numpy as np
from numpy.polynomial.polynomial import Polynomial

def pdf(x, a, b):
    """
    PDF of the quadratic distribution defined on the interval (a, b)
    """
    x = np.asarray(x)
    f = np.zeros_like(x, dtype=float)
    mask = (a < x) & (x < b)
    f[mask] = 6 * (x[mask] - a) * (x[mask] - b) / (a - b)**3
    return f


def cdf(x, a, b):
    """
    CDF of the quadratic distribution defined on the interval (a, b)
    """
    x = np.asarray(x)
    F = np.zeros_like(x, dtype=float)
    F[x >= b] = 1.0
    mask = (a <= x) & (x <= b)
    F[mask] = ((a - x[mask])**2 * (a - 3*b + 2*x[mask])) / (a - b)**3
    return F


def inv(a, b, r, c):
    """
    Inverse CDF (quantile sampling) of the quadratic distribution via solving a cubic equation
    """
    u = np.random.rand(r, c)
    x = np.zeros((r, c))

    for i in range(r):
        for j in range(c):
            # Coefficients of the cubic polynomial:
            # f = - U*(a-b)^3/6 + a^3 - 3*a^2*b + 6*a*b*x - 3*a*x^2 - 3*b*x^2 + 2*x^3
            coeffs = [
                2,
                -3 * (a + b),
                6 * a * b,
                a**3 - 3 * a**2 * b - u[i, j] * (a - b)**3
            ]
            roots = np.roots(coeffs)
            # Choose root in the valid domain
            valid = roots[(roots > a) & (roots < b)]
            if len(valid) != 1:
                raise ValueError(f"Expected 1 root in ({a}, {b}), found: {valid}")
            x[i, j] = valid[0].real
    return x


def rnd(a, b, r, c):
    """
    Generates r × c samples from the quadratic distribution using inverse transform sampling.
    """
    return inv(a, b, r, c)
