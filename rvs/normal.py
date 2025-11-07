import numpy as np
from scipy.stats import norm as scipy_norm

# --------------------------------
# Normal Probability Density Function (PDF)
# --------------------------------
def pdf(x, mu, sigma):
    """
    Compute the PDF of the normal distribution N(mu, sigma^2).

    Parameters:
    - x : scalar or array-like
    - mu : mean
    - sigma : standard deviation

    Returns:
    - pdf : float or np.ndarray
    """
    dist = scipy_norm(loc=mu, scale=sigma)
    return dist.pdf(x)

# --------------------------------
# Normal Cumulative Distribution Function (CDF)
# --------------------------------
def cdf(x, mu, sigma):
    """
    Compute the CDF of the normal distribution N(mu, sigma^2).

    Parameters:
    - x : scalar or array-like
    - mu : mean
    - sigma : standard deviation

    Returns:
    - cdf : float or np.ndarray
    """
    dist = scipy_norm(loc=mu, scale=sigma)
    return dist.cdf(x)

# --------------------------------
# Normal Inverse CDF (Quantile Function)
# --------------------------------
def inv(p, mu, sigma):
    """
    Compute the inverse CDF (quantile function) for N(mu, sigma^2).

    Parameters:
    - p : scalar or array-like in [0, 1]
    - mu : mean
    - sigma : standard deviation

    Returns:
    - x : quantiles corresponding to input probabilities p
    """
    dist = scipy_norm(loc=mu, scale=sigma)
    return dist.ppf(p)

# --------------------------------
# Normal Random Variable Generator
# --------------------------------
def rnd(mu, sigma, size=(1,), seed=None):
    """
    Generate random samples from a normal distribution N(mu, sigma^2).

    Parameters:
    - mu : mean
    - sigma : standard deviation
    - size : tuple, shape of output (e.g., (1000,), (r, c))
    - seed : int or numpy.random.Generator, optional

    Returns:
    - samples : numpy.ndarray of shape `size`
    """
    if isinstance(seed, (int, type(None))):
        rng = np.random.default_rng(seed)
    else:
        rng = seed

    return rng.normal(loc=mu, scale=sigma, size=size)
