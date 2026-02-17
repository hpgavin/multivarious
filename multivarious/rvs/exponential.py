## exponential distribution
# github.com/hpgavin/multivarious ... rvs/exponential

import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs


def _ppp_(x, meanX):
    """
    Validate and preprocess input parameters for consistency and correctness.

    INPUTS:
        x : array_like
            Evaluation points
        meanX : float or array_like 
            Mean(s) of the distribution (must be > 0)

    OUTPUTS:
        x : ndarray
            Evaluation points as array
        meanX : ndarray
            Means as column array
        n : int
            Number of random variables
    """ 

    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    x = np.atleast_1d(x).astype(float)
    meanX = np.atleast_1d(meanX).reshape(-1,1).astype(float)
    n = len(meanX)   
        
    # Validate parameter values 
    if np.any(meanX <= 0):
        raise ValueError("exponential: all meanX values must be positive")

    # Prevent negative or zero values (log not defined)
    x = np.where(x < 0, 0.01, x)

    return x, meanX, n


def pdf(x, meanX):
    """
    exponential.pdf

    Computes the Probability Density Function (PDF) of the exponential distribution.

    INPUTS:
        x : array_like
            Evaluation points (must be x >= 0)
        meanX : float or array_like, shape (n,)
            Mean(s) of the exponential distribution (must be > 0)
 
    OUTPUTS:
        f : ndarray, shape (n, N)
            PDF values at each point in x for each of n random variables
 
    Notes
    -----
    f(x) = (1/meanX) * exp(-x / meanX) for x >= 0

    Reference
    ---------
    https://en.wikipedia.org/wiki/Exponential_distribution
    """

    x, meanX, n = _ppp_(x, meanX)

    f = np.exp(-x / meanX) / meanX

    if n == 1 and f.shape[0] == 1:
        f = f.flatten()
    
    return f


def cdf(x, meanX):
    """
    exponential.cdf

    Computes the Cumulative Distribution Function (CDF) of the exponential distribution.

    INPUTS:
        x : array_like
            Evaluation points (x >= 0)
        meanX : float or array_like, shape (n,)
            Mean(s) of the exponential distribution (must be > 0)
 
    OUTPUTS:
        F : ndarray, shape (n, N)
            CDF values at each point in x for each of n random variables
 
    Notes
    -----
    F(x) = 1 - exp(-x / meanX) for x >= 0

    Reference
    ---------
    https://en.wikipedia.org/wiki/Exponential_distribution
    """

    x, meanX, n = _ppp_(x, meanX)

    F = 1.0 - np.exp(-x / meanX)

    if n == 1 and F.shape[0] == 1:
        F = F.flatten()
    
    return F


def inv(P, meanX):
    """
    exponential.inv

    Computes the inverse CDF (quantile function) of the exponential distribution.

    INPUTS:
        P : array_like
            Probability values (0 <= P <= 1)
        meanX : float or array_like, shape (n,)
            Mean(s) of the exponential distribution (must be > 0)

    OUTPUTS:
        X : ndarray
            Quantiles corresponding to probabilities P

    Notes
    -----
    X = -meanX * log(1 - P)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Exponential_distribution
    """
    
    _, meanX, n = _ppp_(0, meanX)

    # Clip invalid P values 
    P = np.where(P < 0, 0.0, P)
    P = np.where(P > 1, 1.0, P)

    x = -meanX * np.log(1 - P)
    
    if n == 1 and x.shape[0] == 1:
        x = x.flatten()
    
    return x


def rnd(meanX, N, R=None, seed=None):
    """
    exponential.rnd

    Generate random samples from an exponential distribution with mean meanX.

    INPUTS:
        meanX : float or array_like, shape (n,)
            Mean(s) of the distribution (must be > 0)
        N : int
            Number of observations per random variable
        R : ndarray, shape (n, n), optional
            Correlation matrix for generating correlated samples.
            If None, generates uncorrelated samples.
        seed : int, optional
            Random seed for reproducibility
 
    OUTPUTS:
        X : ndarray, shape (n, N) or shape (N,) if n=1
            Random samples from the exponential distribution.
            Each row corresponds to one random variable.
            Each column corresponds to one sample.

    Notes
    -----
    Uses inverse CDF method: x = -meanX * log(U) where U ~ Uniform(0,1)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Exponential_distribution
    """

    _, meanX, n = _ppp_(0, meanX)

    _, _, U = correlated_rvs(R, n, N, seed)

    # Inverse transform: x = -meanX * log(U)
    X = -meanX * np.log(U)

    if n == 1:
        X = X.flatten()

    return X
