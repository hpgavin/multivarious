## rayleigh distribution
# github.com/hpgavin/multivarious ... rvs/rayleigh

import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs


def _ppp_(x, meanX):
    """
    Validate and preprocess input parameters for consistency and correctness.

    INPUTS:
        x : array_like
            Evaluation points
        meanX : float or array_like
            Mean(s) of the Rayleigh distribution (must be > 0)

    OUTPUTS:
        x : ndarray
            Evaluation points as array
        meanX : ndarray
            Means as column array
        modeX : ndarray
            Mode parameters (σ) as column array
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
        raise ValueError("rayleigh: all meanX values must be greater than zero")

    # Replace non-positive values to prevent invalid evaluation
    x[x <= 0] = np.sum(meanX)/(n*1e3)

    # Convert mean meanX to modeX using Rayleigh identity
    modeX = meanX * np.sqrt(2 / np.pi)

    return x, meanX, modeX, n


def pdf(x, meanX):
    """
    rayleigh.pdf

    Computes the PDF of the Rayleigh distribution using the mean parameter meanX.

    INPUTS:
        x : array_like
            Evaluation points (must be ≥ 0)
        meanX : float or array_like, shape (n,)
            Mean(s) of the Rayleigh distribution (must be > 0)

    OUTPUTS:
        f : ndarray, shape (n, N)
            PDF values at each point in x for each of n random variables

    Notes
    -----
    The Rayleigh distribution with scale parameter σ (mode) has:
    f(x) = (x/σ²) exp(-x²/(2σ²)) for x ≥ 0
    where σ = mean · √(2/π)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Rayleigh_distribution
    """

    x, meanX, modeX, n = _ppp_(x, meanX)

    # Apply the Rayleigh PDF formula
    f = (x / modeX**2) * np.exp(-0.5 * (x / modeX)**2)

    if n == 1 and f.shape[0] == 1:
        f = f.flatten()

    return f


def cdf(x, meanX):
    """
    rayleigh.cdf
    
    Computes the CDF of the Rayleigh distribution using the mean parameter meanX.

    INPUTS:
        x : array_like
            Evaluation points (must be ≥ 0)
        meanX : float or array_like, shape (n,)
            Mean(s) of the Rayleigh distribution (must be > 0)

    OUTPUTS:
        F : ndarray, shape (n, N)
            CDF values at each point in x for each of n random variables

    Notes
    -----
    F(x) = 1 - exp(-x²/(2σ²)) for x ≥ 0

    Reference
    ---------
    https://en.wikipedia.org/wiki/Rayleigh_distribution
    """

    x, meanX, modeX, n = _ppp_(x, meanX)

    # Apply the Rayleigh CDF formula
    F = 1.0 - np.exp(-0.5 * (x / modeX)**2)

    if n == 1 and F.shape[0] == 1:
        F = F.flatten()

    return F


def inv(F, meanX):
    """
    rayleigh.inv

    Computes the inverse CDF (quantile function) of the Rayleigh distribution
    using the mean parameter meanX.

    INPUTS:
        F : array_like
            Non-exceedance probabilities (0 ≤ F ≤ 1)
        meanX : float or array_like, shape (n,)
            Mean(s) of the Rayleigh distribution (must be > 0)

    OUTPUTS:
        x : ndarray
            Quantile values corresponding to probabilities F

    Notes
    -----
    x = σ√(-2 ln(1-F)) where σ = mean · √(2/π)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Rayleigh_distribution
    """

    _, meanX, modeX, n = _ppp_(0, meanX)

    F = np.atleast_2d(F).astype(float)
    F = np.clip(F, np.finfo(float).eps, 1 - np.finfo(float).eps)
    N = F.shape[1]    

    # Compute the inverse CDF formula
    x = modeX * np.sqrt(-2.0 * np.log(1 - F))

    if n == 1 and x.shape[0] == 1:
        x = x.flatten()

    return x


def rnd(meanX, N, R=None, seed=None):
    """
    rayleigh.rnd

    Generates random samples from the Rayleigh distribution using the mean
    parameter meanX.

    INPUTS:
        meanX : float or array_like, shape (n,)
            Mean(s) of the Rayleigh distribution (must be > 0)
        N : int
            Number of observations per random variable
        R : ndarray, shape (n, n), optional
            Correlation matrix for generating correlated samples.
            If None, generates uncorrelated samples.
        seed : int, optional
            Random seed for reproducibility

    OUTPUTS:
        X : ndarray, shape (n, N) or shape (N,) if n=1
            Random samples drawn from the Rayleigh distribution.
            Each row corresponds to one random variable.
            Each column corresponds to one sample.

    Notes
    -----
    Uses inverse transform method: x = σ√(-2 ln(u)) where u ~ Uniform(0,1)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Rayleigh_distribution
    """

    _, meanX, modeX, n = _ppp_(0, meanX)

    _, _, U = correlated_rvs(R, n, N, seed)

    # Inverse transform sampling
    X = inv(U, meanX) 

    return X
