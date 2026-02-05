import numpy as np
from scipy.stats import norm as scipy_norm

from multivarious.utl.correlated_rvs import correlated_rvs

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


def rnd(mu, sigma, N, R=None):
    '''
    normal.rnd
    Generates correlated random samples from the normal distribution N(mu, sigma²).

    Parameters:
        mu : float (n,)
            Mean of the distribution
        sigma : float (n,)
            Standard deviation of the distribution (must be > 0)
        N : int
            number of observations of each of the n random variables 
        R : float (n,n) - optional
            correlation matrix

    Output:
        X : ndarray
            Array of normal random samples with shape `size`

    Reference:
    https://en.wikipedia.org/wiki/Normal_distribution
    '''
    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    mu = np.atleast_1d(mu).astype(float)
    sigma = np.atleast_1d(sigma).astype(float)

    # Determine number of random variables
    n = len(mu)

    # Validate that all parameter arrays have the same length
    if not (len(mu) == n and len(sigma) == n):
        raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got mu:{len(mu)}, sigma:{len(sigma)}"

    _, Y, _ = correlated_rvs(R,n,N)

    X = mu + sigma*Y

    return X
