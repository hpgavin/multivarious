import numpy as np
from scipy.stats import norm as scipy_norm


def pdf(x, mu, sigma):
    '''
    normal.pdf

    Computes the PDF of the normal distribution N(mu, sigma²).

    Parameters:
        x : array_like or float
            Evaluation points
        mu : float
            Mean of the distribution
        sigma : float
            Standard deviation of the distribution (must be > 0)

    Output:
        f : ndarray or float
            PDF values at each point in x

    Reference:
    https://en.wikipedia.org/wiki/Normal_distribution
    '''
    dist = scipy_norm(loc=mu, scale=sigma)
    return dist.pdf(x)


def cdf(x, mu, sigma):
    '''
    normal.cdf

    Computes the CDF of the normal distribution N(mu, sigma²).

    Parameters:
        x : array_like or float
            Evaluation points
        mu : float
            Mean of the distribution
        sigma : float
            Standard deviation of the distribution (must be > 0)

    Output:
        F : ndarray or float
            CDF values at each point in x

    Reference:
    https://en.wikipedia.org/wiki/Normal_distribution
    '''

    dist = scipy_norm(loc=mu, scale=sigma)
    return dist.cdf(x)


def inv(p, mu, sigma):
    '''
    normal.inv

    Computes the inverse CDF (quantile function) of the normal distribution N(mu, sigma²).

    Parameters:
        p : array_like or float
            Probability values (must be in [0, 1])
        mu : float
            Mean of the distribution
        sigma : float
            Standard deviation of the distribution (must be > 0)

    Output:
        x : ndarray or float
            Quantile values corresponding to probabilities p

    Reference:
    https://en.wikipedia.org/wiki/Normal_distribution
    '''

    dist = scipy_norm(loc=mu, scale=sigma)
    return dist.ppf(p)


def rnd(mu, sigma, size=(1,), seed=None):
    '''
    normal.rnd
    Generates random samples from the normal distribution N(mu, sigma²).

    Parameters:
        mu : float
            Mean of the distribution
        sigma : float
            Standard deviation of the distribution (must be > 0)
        size : tuple, optional
            Shape of the output array (e.g., (1000,), (r, c)); default is (1,)
        seed : int or numpy.random.Generator, optional
            Random seed or Generator for reproducibility

    Output:
        x : ndarray
            Array of normal random samples with shape `size`

    Reference:
    https://en.wikipedia.org/wiki/Normal_distribution
    '''

    if isinstance(seed, (int, type(None))):
        rng = np.random.default_rng(seed)
    else:
        rng = seed

    X = rng.normal(loc=mu, scale=sigma, size=size)

    return X
