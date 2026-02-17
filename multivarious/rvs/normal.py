## normal distribution
# github.com/hpgavin/multivarious ... rvs/normal

import numpy as np
from scipy.special import erf as scipy_erf
from scipy.special import erfinv as scipy_erfinv

from multivarious.utl.correlated_rvs import correlated_rvs


def _ppp_(x, meanX=0.0, sdvnX=1.0):
    """
    Validate and preprocess input parameters for consistency and correctness.

    INPUTS:
        x : array_like
            Evaluation points
        meanX : float or array_like
            Mean(s) of the distribution
        sdvnX : float or array_like
            Standard deviation(s) of the distribution (must be > 0)

    OUTPUTS:
        x : ndarray
            Evaluation points as row array
        meanX : ndarray
            Means as column array
        sdvnX : ndarray
            Standard deviations as column array
        n : int
            Number of random variables
    """ 

    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    x = np.atleast_1d(x).reshape(1,-1).astype(float)

    meanX = np.atleast_1d(meanX).reshape(-1,1).astype(float)
    sdvnX = np.atleast_1d(sdvnX).reshape(-1,1).astype(float)
    n = len(meanX)   
        
    # Validate parameter dimensions 
    if not (len(meanX) == n and len(sdvnX) == n):
        raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got meanX:{len(meanX)}, sdvnX:{len(sdvnX)}") 

    # Validate parameter values 
    if np.any(sdvnX <= 0):
        raise ValueError("normal: sdvnX must be > 0")

    return x, meanX, sdvnX, n


def pdf(x, meanX=0.0, sdvnX=1.0):
    """
    normal.pdf

    Computes the PDF of the normal distribution N(meanX, sdvnX²).

    INPUTS:
        x : array_like
            Evaluation points
        meanX : float or array_like, shape (n,)
            Mean(s) of the distribution
        sdvnX : float or array_like, shape (n,)
            Standard deviation(s) of the distribution (must be > 0)

    OUTPUTS:
        f : ndarray, shape (n, N)
            PDF values at each point in x for each of n random variables

    Notes
    -----
    The normal (Gaussian) distribution with mean μ and standard deviation σ:
    f(x) = (1/√(2πσ²)) exp(-(x-μ)²/(2σ²))

    Reference
    ---------
    https://en.wikipedia.org/wiki/Normal_distribution
    """
 
    x, meanX, sdvnX, n = _ppp_(x, meanX, sdvnX)

    z = (x - meanX) / sdvnX

    f = 1.0 / np.sqrt(2 * np.pi*sdvnX**2) * np.exp(-(z**2.0) / 2.0)

    if n == 1 and f.shape[0] == 1:
        f = f.flatten()

    return f 


def cdf(x, params=[0.0, 1.0]):
    """
    normal.cdf

    Computes the CDF of the normal distribution N(meanX, sdvnX²).

    INPUTS:
        x : array_like
            Evaluation points
        meanX : float or array_like, shape (n,)
            Mean(s) of the distribution
        sdvnX : float or array_like, shape (n,)
            Standard deviation(s) of the distribution (must be > 0)

    OUTPUTS:
        F : ndarray, shape (n, N)
            CDF values at each point in x for each of n random variables

    Notes
    -----
    F(x) = (1 + erf((x-μ)/(σ√2)))/2

    Reference
    ---------
    https://en.wikipedia.org/wiki/Normal_distribution
    """

    meanX, sdvnX = params

    _, meanX, sdvnX, n = _ppp_(0, meanX, sdvnX)

    z = (x - meanX) / sdvnX

    F = (1.0 + scipy_erf(z / np.sqrt(2.0))) / 2.0

    if n == 1 and x.shape[0] == 1:
        F = F.flatten()

    return F


def inv(F, meanX=0.0, sdvnX=1.0):
    """
    normal.inv

    Computes the inverse CDF (quantile function) of the normal distribution N(meanX, sdvnX²).

    INPUTS:
        F : array_like
            Probability values (must be in [0, 1])
        meanX : float or array_like, shape (n,)
            Mean(s) of the distribution
        sdvnX : float or array_like, shape (n,)
            Standard deviation(s) of the distribution (must be > 0)

    OUTPUTS:
        x : ndarray
            Quantile values corresponding to probabilities F

    Notes
    -----
    x = μ + σ√2 · erfinv(2F - 1)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Normal_distribution
    """

    _, meanX, sdvnX, n = _ppp_(0, meanX, sdvnX)

    F = np.atleast_2d(F).astype(float)
    F = np.clip(F, np.finfo(float).eps, 1 - np.finfo(float).eps)
    N = F.shape[1]    
    
    # Compute normal quantile using inverse CDF formula
    z = np.sqrt(2) * scipy_erfinv(2 * F - 1) 
    x = meanX + sdvnX * z

    if n == 1 and x.shape[0] == 1:
        x = x.flatten()

    return x


def rnd(meanX=0.0, sdvnX=1.0, N=1, R=None, seed=None):
    """
    normal.rnd
    
    Generates correlated random samples from the normal distribution N(meanX, sdvnX²).

    INPUTS:
        meanX : float or array_like, shape (n,)
            Mean(s) of the distribution
        sdvnX : float or array_like, shape (n,)
            Standard deviation(s) of the distribution (must be > 0)
        N : int
            Number of observations per random variable
        R : ndarray, shape (n, n), optional
            Correlation matrix for generating correlated samples.
            If None, generates uncorrelated samples.
        seed : int, optional
            Random seed for reproducibility

    OUTPUTS:
        X : ndarray, shape (n, N) or shape (N,) if n=1
            Array of normal random samples.
            Each row corresponds to one random variable.
            Each column corresponds to one sample.

    Notes
    -----
    Uses eigenvalue decomposition of correlation matrix to generate 
    correlated standard normal variates, then transforms to desired
    mean and standard deviation.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Normal_distribution
    """

    _, meanX, sdvnX, n = _ppp_(0, meanX, sdvnX)

    # Correlated standard normal variables (n,N)
    _, _, U = correlated_rvs(R, n, N, seed)

    X = inv(U, meanX, sdvnX)

    return X
