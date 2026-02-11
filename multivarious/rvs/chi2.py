## chi-squared distribution
# github.com/hpgavin/multivarious ... rvs/chi2

import numpy as np
from scipy.stats import norm as scipy_normal

from multivarious.utl.correlated_rvs import correlated_rvs


def _ppp_(x, k):
    """
    Validate and preprocess input parameters for consistency and correctness.

    INPUTS:
        x : array_like
            Evaluation points
        k : float or array_like
            Degrees of freedom (must be > 0)

    OUTPUTS:
        x : ndarray
            Evaluation points as array
        k : ndarray
            Degrees of freedom as column array
        n : int
            Number of random variables
        m : ndarray
            Wilson-Hilferty transformation mean
        s : ndarray
            Wilson-Hilferty transformation standard deviation
    """ 

    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    x = np.atleast_1d(x).astype(float)
    k = np.atleast_1d(k).reshape(-1,1).astype(float)
    n = len(k)   
        
    # Validate parameter values 
    if np.any(k <= 0):
        raise ValueError("chi2: all k values must be greater than zero")

    # Wilson-Hilferty approximation parameters
    m = 1 - 2 / (9 * k)         # mean of cube-root-transformed variable
    s = np.sqrt(2 / (9 * k))    # std dev of cube-root-transformed variable

    return x, k, n, m, s


def pdf(x, k):
    """
    chi2.pdf

    Computes the PDF of the Chi-squared distribution using the 
    Wilson-Hilferty transformation.

    INPUTS:
        x : array_like
            Evaluation points
        k : float or array_like, shape (n,)
            Degrees of freedom (must be > 0)

    OUTPUTS:
        f : ndarray, shape (n, N)
            Approximate PDF values at each point in x for each of n random variables

    Notes
    -----
    Uses Wilson-Hilferty transformation to approximate chi-squared distribution
    via normal distribution: Z = (X/k)^(1/3) ~ N(m, s²)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Chi-squared_distribution
    """
    
    x, k, n, m, s = _ppp_(x, k)
    
    # Transform x into z-space: Z = (X / k)^{1/3}
    z = (x / k) ** (1/3)

    # Approximate PDF using normal distribution
    f = scipy_normal.pdf(z, m, s)

    return f


def cdf(x, k):
    """
    chi2.cdf

    Approximates the CDF of the chi-squared distribution using the
    Wilson-Hilferty transformation, which maps chi2(k) into a normal distribution.

    INPUTS:
        x : array_like
            Evaluation points
        k : float or array_like, shape (n,)
            Degrees of freedom (must be > 0)
 
    OUTPUTS:
        F : ndarray, shape (n, N)
            Approximate cumulative probability evaluated at x for each of n random variables

    Notes
    -----
    Uses Wilson-Hilferty transformation: Z = (X/k)^(1/3) ~ N(m, s²)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Chi-squared_distribution
    """

    x, k, n, m, s = _ppp_(x, k)

    # Apply transformation: (X/k)^(1/3)
    z = (x / k) ** (1 / 3)

    # Apply normal CDF using transformed variable
    F = scipy_normal.cdf(z, loc=m, scale=s)

    return F


def inv(p, k):
    """
    chi2.inv
 
    Approximates the inverse CDF (quantile function) of the chi-squared distribution
    using the Wilson-Hilferty transformation.
 
    INPUTS:
        p : array_like
            Non-exceedance probabilities (values between 0 and 1)
        k : float or array_like, shape (n,)
            Degrees of freedom (must be > 0)
 
    OUTPUTS:
        x : ndarray
            Quantile values such that Prob[X ≤ x] = p

    Notes
    -----
    Uses Wilson-Hilferty transformation: x = k * z³ where z ~ N(m, s²)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Chi-squared_distribution
    """

    _, k, n, m, s = _ppp_(0, k)

    p = np.asarray(p, dtype=float)

    # Inverse normal CDF
    z = scipy_normal.ppf(p, loc=m, scale=s)

    # Apply inverse transformation: x = k * z³
    x = k * z**3

    return x


def rnd(k, N, R=None, seed=None):
    """
    chi2.rnd
    
    Generate N observations of n correlated (or uncorrelated) chi2 random variables
    via the Wilson-Hilferty transformation (a good approximation).

    INPUTS:
        k : float or array_like, shape (n,)
            Degrees of freedom (must be > 0)
        N : int
            Number of observations per random variable
        R : ndarray, shape (n, n), optional
            Correlation matrix for generating correlated samples.
            If None, generates uncorrelated samples.
        seed : int, optional
            Random seed for reproducibility
 
    OUTPUTS:
        X : ndarray, shape (n, N) or shape (N,) if n=1
            Random samples from Chi-squared(k) distribution.
            Each row corresponds to one random variable.
            Each column corresponds to one sample.

    Notes
    -----
    Uses Wilson-Hilferty transformation via correlated normal variates.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Chi-squared_distribution
    """

    _, k, n, m, s = _ppp_(0, k)

    _, Y, _ = correlated_rvs(R, n, N, seed)
   
    # Apply transformation
    X = np.zeros((n, N))
    for i in range(n):
        X[i, :] = k[i] * (m[i] + s[i] * Y[i, :]) ** 3

    if n == 1:
        X = X.flatten()

    return X
