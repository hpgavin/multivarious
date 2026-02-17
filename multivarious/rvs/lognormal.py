## lognormal distribution
# github.com/hpgavin/multivarious ... rvs/lognormal

import numpy as np
from scipy.special import erf as scipy_erf
from scipy.special import erfinv as scipy_erfinv

from multivarious.utl.correlated_rvs import correlated_rvs


def _ppp_(x, mednX, covnX):
    """
    Validate and preprocess input parameters for consistency and correctness.

    INPUTS:
        x : array_like
            Evaluation points
        mednX : float or array_like
            Median(s) of the lognormal distribution (must be > 0)
        covnX : float or array_like
            Coefficient(s) of variation (must be > 0)

    OUTPUTS:
        x : ndarray
            Evaluation points as array (with x<=0 replaced by 0.01)
        mednX : ndarray
            Medians as column array
        covnX : ndarray
            Coefficients of variation as column array
        n : int
            Number of random variables
    """ 

    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    x = np.atleast_1d(x).astype(float)

    mednX = np.atleast_1d(mednX).reshape(-1,1).astype(float)
    covnX = np.atleast_1d(covnX).reshape(-1,1).astype(float)
    n = len(mednX)   
        
    # Validate parameter dimensions 
    if not (len(mednX) == n and len(covnX) == n):
        raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got mednX:{len(mednX)}, covnX:{len(covnX)}")

    # Validate parameter values 
    if np.any(mednX <= 0):
        raise ValueError("lognormal: mednX must be > 0")
    if np.any(covnX <= 0):
        raise ValueError("lognormal: covnX must be > 0")

    # Replace invalid x <= 0 with 0.01 to avoid log(0)
    x = np.where(x <= 0, 0.01, x)
    
    return x, mednX, covnX, n


def pdf(x, mednX, covnX):
    """
    lognormal.pdf
 
    Computes the Probability Density Function (PDF) of the lognormal distribution.
 
    INPUTS:
        x : array_like
            Evaluation points. Must be > 0 since lognormal is defined for x > 0.
        mednX : float or array_like, shape (n,)
            Median(s) of the lognormal distribution
        covnX : float or array_like, shape (n,)
            Coefficient(s) of variation: covnX = std(X) / mean(X)
 
    OUTPUTS:
        f : ndarray, shape (n, N)
            PDF values at each point in x for each of n random variables
 
    Notes
    -----
    If X ~ Lognormal(mednX, covnX), then log(X) ~ Normal with:
        mean = log(mednX)
        variance = log(1 + covnX²)
    
    f(x) = (1/(x·√(2πV))) exp(-(log(x/mednX))²/(2V))
    where V = log(1 + covnX²)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Log-normal_distribution
    """

    x, mednX, covnX, n = _ppp_(x, mednX, covnX) 

    # Compute variance of log(X)
    VlnX = np.log(1 + covnX**2)
    
    # Compute using Lognormal PDF formula
    f = (1.0 / (x * np.sqrt(2.0 * np.pi * VlnX))) * np.exp(-0.5 * (np.log(x / mednX))**2.0 / VlnX)
 
    if n == 1 and f.shape[0] == 1:
        f = f.flatten()
    
    return f


def cdf(x, params):
    """ 
    lognormal.cdf
 
    Computes the CDF of a lognormal distribution.
 
    INPUTS:
        x : array_like
            Evaluation points
        params : array_like [mednX, covnX]
            mednX : float or array_like
                Median(s) of X
            covnX : float or array_like
                Coefficient(s) of variation of X
    
    OUTPUTS: 
        F : ndarray, shape (n, N)
            CDF values at each point in x for each of n random variables

    Notes
    -----
    F(x) = (1 + erf((log(x) - log(mednX)) / √(2V))) / 2
    where V = log(1 + covnX²)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Log-normal_distribution
    """ 
    
    mednX, covnX = params

    x, mednX, covnX, n = _ppp_(x, mednX, covnX) 

    # Compute variance of log(X)
    VlnX = np.log(1 + covnX**2)
    
    # Lognormal CDF formula
    F = 0.5 * (1 + scipy_erf((np.log(x) - np.log(mednX)) / np.sqrt(2 * VlnX)))
    
    if n == 1 and F.shape[0] == 1:
        F = F.flatten()
    
    return F


def inv(F, mednX, covnX):
    """
    lognormal.inv
 
    Computes the inverse CDF (quantile function) for lognormal distribution.
 
    INPUTS:
        F : array_like
            Probability values (must be in (0,1))
        mednX : float or array_like, shape (n,)
            Median(s) of the lognormal distribution
        covnX : float or array_like, shape (n,)
            Coefficient(s) of variation
 
    OUTPUTS:
        x : ndarray
            Quantile values such that P(X <= x) = F
   
    Notes
    -----
    x = exp(log(mednX) + √(2V) · erfinv(2F - 1))
    where V = log(1 + covnX²)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Log-normal_distribution
    """

    _, mednX, covnX, n = _ppp_(0, mednX, covnX) 

    F = np.atleast_2d(F).astype(float)
    F = np.clip(F, np.finfo(float).eps, 1 - np.finfo(float).eps)
    N = F.shape[1]    

    # Compute lognormal quantile using inverse CDF formula
    VlnX = np.log(1 + covnX**2)  # Variance of log(X)

    x = np.exp(np.log(mednX) + np.sqrt(2 * VlnX) * scipy_erfinv(2 * F - 1)) 

    if n == 1 and x.shape[0] == 1:
        x = x.flatten()
    
    return x


def rnd(mednX, covnX, N, R=None, seed=None):
    """
    lognormal.rnd
 
    Generate N observations of correlated (or uncorrelated) lognormal random variables.
 
    INPUTS:
        mednX : float or array_like, shape (n,)
            Median(s) of the lognormal distribution
        covnX : float or array_like, shape (n,)
            Coefficient(s) of variation
        N : int
            Number of observations per random variable
        R : ndarray, shape (n, n), optional
            Correlation matrix for generating correlated samples.
            If None, generates uncorrelated samples.
        seed : int, optional
            Random seed for reproducibility
 
    OUTPUTS:
        X : ndarray, shape (n, N) or shape (N,) if n=1
            Correlated lognormal random samples.
            Each row corresponds to one random variable.
            Each column corresponds to one observation.
  
    Notes
    -----
    Method (Gaussian Copula):
        1. Generate correlated standard normal samples Y ~ N(0, R)
        2. Transform to lognormal: X = exp(log(mednX) + Y * √V)
        where V = log(1 + covnX²)
    
    If X is a lognormal random variable, then log(X) is normally 
    distributed with mean log(mednX) and variance log(1+covnX²)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Log-normal_distribution
    """
    
    _, mednX, covnX, n = _ppp_(0, mednX, covnX) 

    _, _, U = correlated_rvs(R, n, N, seed)

    # Transform to lognormal: x = exp(log(mednX) + Y * sqrt(VlnX))
    X = inv( U, mednX, covnX )
    
    return X
