import numpy as np
from scipy.stats import uniform as scipy_uniform

# Scipy is a popular library for scientific computing in Python. It provides
# a wide range of statistical distributions and functions.

# -------------------------------
# Uniform Probability Density Function (PDF)
# -------------------------------
def uniform_pdf(x, a, b):
    """
    Compute the PDF of the uniform distribution on [a, b].

    Parameters:
    - x : scalar or array-like
    - a : lower bound
    - b : upper bound

    Returns:
    - pdf : float or np.ndarray
    """
    dist = scipy_uniform(loc=a, scale=b - a)  # loc refers to the mean, scale refers to the standard deviation
    return dist.pdf(x)


# -------------------------------
# Uniform Cumulative Distribution Function (CDF)
# -------------------------------
def uniform_cdf(x, a, b):
    """
    Compute the CDF of the uniform distribution on [a, b].

    Parameters:
    - x : scalar or array-like
    - a : lower bound
    - b : upper bound

    Returns:
    - cdf : float or np.ndarray
    """
    dist = scipy_uniform(loc=a, scale=b - a)
    return dist.cdf(x)


# -------------------------------
# Uniform Inverse CDF (Quantile Function)
# -------------------------------
def uniform_inv(p, a, b):
    """
    Compute the inverse CDF (quantile function) for the uniform distribution on [a, b].

    Parameters:
    - p : scalar or array-like in [0, 1]
    - a : lower bound
    - b : upper bound

    Returns:
    - x : quantiles corresponding to input probabilities p
    """
    dist = scipy_uniform(loc=a, scale=b - a)
    return dist.ppf(p)


# -------------------------------
# Uniform Random Variable Generator
# -------------------------------
def uniform_rnd(a, b, size=(1,), seed=None):
    """
    Generate random samples from a uniform distribution on [a, b].

    Parameters:
    - a : lower bound
    - b : upper bound
    - size : tuple, shape of output (e.g., (1000,), (r, c))
    - seed : int or numpy.random.Generator, optional

    Returns:
    - samples : numpy.ndarray of shape `size`
    """
    if isinstance(seed, (int, type(None))):
        rng = np.random.default_rng(seed)
    else:
        rng = seed

    return rng.uniform(low=a, high=b, size=size)
