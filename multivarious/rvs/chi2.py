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

    Reference
    ---------
    https://en.wikipedia.org/wiki/Chi-squared_distribution#Asymptotic_properties
    """ 

    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    x = np.atleast_1d(x).astype(float)
    k = np.atleast_1d(k).reshape(-1,1).astype(float)
    n = len(k)   
    N = len(x)
        
    # Validate parameter values 
    if np.any(k <= 0):
        raise ValueError("chi2: all k values must be greater than zero")

    # Wilson-Hilferty approximation parameters
    m = 1 - 2 / (9 * k)         # mean of cube-root-transformed variable
    s = np.sqrt(2 / (9 * k))    # std dev of cube-root-transformed variable

    return x, k, n, m, s, N


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
    https://en.wikipedia.org/wiki/Chi-squared_distribution#Asymptotic_properties
    """
    
    x, k, n, m, s, N = _ppp_(x, k)
    
    # Transform x into z-space: Z = (X / k)^{1/3}
    z = ( (x / k) ** (1/3) - m ) / s

    # Approximate PDF using normal distribution
    f = scipy_normal.pdf(z, 0, 1) * (np.sqrt(2)*s)

    if n == 1 and f.shape[0] == 1:
        f = f.flatten()

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
    https://en.wikipedia.org/wiki/Chi-squared_distribution#Asymptotic_properties
    """

    x, k, n, m, s, N = _ppp_(x, k)

    # Apply transformation: (X/k)^(1/3)
    z = ( (x / k) ** (1/3) - m ) / s

    # Apply normal CDF using transformed variable
    F = scipy_normal.cdf(z, 0, 1)

    if n == 1 and F.shape[0] == 1:
        F = F.flatten()

    return F


def inv(F, k):
    """
    chi2.inv
 
    Approximates the inverse CDF (quantile function) of the chi-squared distribution
    using the Wilson-Hilferty transformation.
 
    INPUTS:
        F : array_like
            Non-exceedance probabilities (values between 0 and 1)
        k : float or array_like, shape (n,)
            Degrees of freedom (must be > 0)
 
    OUTPUTS:
        x : ndarray
            Quantile values such that Prob[X ≤ x] = F

    Notes
    -----
    Uses Wilson-Hilferty transformation: x = k * z³ where z ~ N(m, s²)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Chi-squared_distribution#Asymptotic_properties
    """

    _, k, n, m, s, _ = _ppp_(0, k)

    F = np.atleast_2d(F).astype(float)
    F = np.clip(F, np.finfo(float).eps, 1 - np.finfo(float).eps)
    N = F.shape[1]    

    x = np.zeros((n,N)) 

    # Inverse normal CDF
    z = scipy_normal.ppf(F, m, s)

    print('F shape', F.shape)
    print('x shape', x.shape)
    print('z shape', z.shape)

    # Inverse transformation
    for i in range(n):
        x[i, :] = k[i] * z[i, :] ** 3

    x [ x <= 0 ] = 1e-12

    if n == 1 and x.shape[0] == 1:
        x = x.flatten()

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
    https://en.wikipedia.org/wiki/Chi-squared_distribution#Asymptotic_properties
    """

    _, k, n, m, s, _ = _ppp_(0, k)

    _, _, U = correlated_rvs(R, n, N, seed)
   
    # Apply transformation
    X = inv(U, k)  

    return X
