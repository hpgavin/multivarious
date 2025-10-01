import numpy as np

def triangular_pdf(x, a, b, c):
    """
    Probability Density Function of the triangular distribution on [a, b] with mode c.
    Parameters are a, b, c where a < c < b
    All must be scalars.
    
    Reference:   http://en.wikipedia.org/wiki/Triangular_distribution
    """
    x = np.asarray(x)
    pdf = np.zeros_like(x, dtype=float)

    # Left side of peak
    left = (a <= x) & (x < c)
    pdf[left] = 2 * (x[left] - a) / ((b - a) * (c - a))

    # Right side of peak
    right = (c <= x) & (x <= b)
    pdf[right] = 2 * (b - x[right]) / ((b - a) * (b - c))

    # Outside [a, b] => already zero
    return pdf


def triangular_cdf(x, a, b, c):
    """
    Cumulative Distribution Function of the triangular distribution on [a, b] with mode c.
    """
    x = np.asarray(x)
    cdf = np.zeros_like(x, dtype=float)

    left = (x <= a)
    mid1 = (a < x) & (x < c)
    mid2 = (c <= x) & (x < b)
    right = (x >= b)

    cdf[mid1] = ((x[mid1] - a) ** 2) / ((b - a) * (c - a))
    cdf[mid2] = 1 - ((b - x[mid2]) ** 2) / ((b - a) * (b - c))
    cdf[right] = 1.0

    return cdf

def triangular_inv(p, a, b, c):
    """
    Inverse CDF (Quantile Function) of the triangular distribution on [a, b] with mode c.
    """
    p = np.asarray(p)
    p = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)

    Fc = (c - a) / (b - a)
    x = np.empty_like(p, dtype=float)

    below = p < Fc
    above = ~below

    x[below] = a + np.sqrt(p[below] * (b - a) * (c - a))
    x[above] = b - np.sqrt((1 - p[above]) * (b - a) * (b - c))

    return x

def triangular_rnd(a, b, c, size=(1,), seed=None):
    """
    Generate random samples from a triangular distribution.

    Parameters:
    - a, b, c: scalars (lower, upper, mode)
    - size: tuple, e.g. (1000,) or (r, c)
    - seed: int or np.random.Generator, optional

    Returns:
    - samples : np.ndarray of shape `size`
    """
    if isinstance(seed, (int, type(None))):
        rng = np.random.default_rng(seed)
    else:
        rng = seed  # assume user passed Generator

    u = rng.random(size)
    return triangular_inv(u, a, b, c)
