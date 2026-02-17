## laplace distribution
# github.com/hpgavin/multivarious ... rvs/laplace

import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs


def _ppp_(x, meanX, sdvnX):
    """
    Validate and preprocess input parameters for consistency and correctness.

    INPUTS:
        x : array_like
            Evaluation points
        meanX : float or array_like
            Mean(s) (location parameter) of the distribution
        sdvnX : float or array_like
            Standard deviation(s) (scale parameter) of the distribution (must be > 0)

    OUTPUTS:
        x : ndarray
            Evaluation points as array
        meanX : ndarray
            Means as column array
        sdvnX : ndarray
            Standard deviations as column array
        n : int
            Number of random variables
    """ 

    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    x = np.atleast_1d(x).astype(float)

    meanX = np.atleast_1d(meanX).reshape(-1,1).astype(float)
    sdvnX = np.atleast_1d(sdvnX).reshape(-1,1).astype(float)
    n = len(meanX)   
    N = len(x)
        
    # Validate parameter dimensions 
    if not (len(meanX) == n and len(sdvnX) == n):
        raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got meanX:{len(meanX)}, sdvnX:{len(sdvnX)}")

    # Validate parameter values 
    if np.any(sdvnX <= 0):
        raise ValueError("laplace: all sdvnX values must be greater than zero")

    return x, meanX, sdvnX, n, N


def pdf(x, meanX, sdvnX):
    """
    laplace.pdf

    Computes the PDF of the Laplace distribution with mean (location)
    parameter meanX and standard deviation (scale) parameter sdvnX.

    INPUTS:
        x : array_like
            Evaluation points
        meanX : float or array_like, shape (n,)
            Mean(s) (location parameter)
        sdvnX : float or array_like, shape (n,)
            Standard deviation(s) (scale parameter) (must be > 0)

    OUTPUTS:
        f : ndarray, shape (n, N)
            PDF values at each point in x for each of n random variables

    Notes
    -----
    f(x) = (1/(√2·σ)) exp(-√2|x-μ|/σ)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Laplace_distribution
    """

    x, meanX, sdvnX, n, N = _ppp_(x, meanX, sdvnX)

    sr2 = np.sqrt(2)
    f = (1 / (sr2 * sdvnX)) * np.exp(-sr2 * np.abs(x - meanX) / sdvnX)

    if n == 1 and f.shape[0] == 1:
        f = f.flatten()
    
    return f


def cdf(x, params):
    """
    laplace.cdf

    Computes the CDF of the Laplace distribution with parameters meanX and sdvnX.

    INPUTS:
        x : array_like
            Evaluation points
        params : array_like [meanX, sdvnX]
            meanX : float or array_like
                Mean(s) (location parameter)
            sdvnX : float or array_like
                Standard deviation(s) (scale parameter) (must be > 0)

    OUTPUTS:
        F : ndarray, shape (n, N)
            CDF values at each point in x for each of n random variables

    Notes
    -----
    F(x) = 0.5·exp(√2(x-μ)/σ) for x ≤ μ
    F(x) = 1 - 0.5·exp(-√2(x-μ)/σ) for x > μ

    Reference
    ---------
    https://en.wikipedia.org/wiki/Laplace_distribution
    """
    
    meanX, sdvnX = params

    x, meanX, sdvnX, n, N = _ppp_(x, meanX, sdvnX)

    sr2 = np.sqrt(2)

    F = np.zeros((n,N))

    for i in range(n): 
        mask = x <= meanX[i]
        F[i,mask] =       0.5 * np.exp( sr2 * (x[mask] - meanX[i]) / sdvnX[i])
        mask = ~mask
        F[i,mask] = 1.0 - 0.5 * np.exp(-sr2 * (x[mask] - meanX[i]) / sdvnX[i])
    
    if n == 1 and F.shape[0] == 1:
        F = F.flatten()
    
    return F


def inv(F, meanX, sdvnX):
    """
    laplace.inv

    Computes the inverse CDF (quantile function) of the Laplace distribution
    with mean (location) meanX and standard deviation (scale) sdvnX.

    INPUTS:
        F : array_like
            Non-exceedance probabilities (must be in [0, 1])
        meanX : float or array_like, shape (n,)
            Mean(s) (location parameter)
        sdvnX : float or array_like, shape (n,)
            Standard deviation(s) (scale parameter) (must be > 0)

    OUTPUTS:
        x : ndarray
            Quantile values corresponding to probabilities F

    Notes
    -----
    x = μ + (σ/√2)·ln(2F) for F ≤ 0.5
    x = μ - (σ/√2)·ln(2(1-F)) for F > 0.5

    Reference
    ---------
    https://en.wikipedia.org/wiki/Laplace_distribution
    """
    
    _, meanX, sdvnX, n, N = _ppp_(F, meanX, sdvnX)


    F = np.atleast_2d(F).astype(float)
    F = np.clip(F, np.finfo(float).eps, 1 - np.finfo(float).eps)
    N = F.shape[1]    

    sr2 = np.sqrt(2)

    x = np.zeros((n,N))
  
    for i in range(n): 
        mask = F[i,:] <= 0.5 
        x[i,mask] = meanX[i] + sdvnX[i] / sr2 * np.log(2 * F[i,mask]) 
        mask = ~mask
        x[i,mask] = meanX[i] - sdvnX[i] / sr2 * np.log(2 * (1 - F[i,mask]))

#   if x.size > 1 else x.item()
    if n == 1 and x.shape[0] == 1:
        x = x.flatten()
    
    return x


def rnd(meanX, sdvnX, N, R=None, seed=None):
    """
    laplace.rnd

    Generates random samples from the Laplace distribution using the
    inverse transform sampling method.

    INPUTS:
        meanX : float or array_like, shape (n,)
            Mean(s) (location parameter)
        sdvnX : float or array_like, shape (n,)
            Standard deviation(s) (scale parameter) (must be > 0)
        N : int
            Number of observations per random variable
        R : ndarray, shape (n, n), optional
            Correlation matrix for generating correlated samples.
            If None, generates uncorrelated samples.
        seed : int, optional
            Random seed for reproducibility

    OUTPUTS:
        X : ndarray, shape (n, N) or shape (N,) if n=1
            Random samples drawn from the Laplace distribution.
            Each row corresponds to one random variable.
            Each column corresponds to one sample.

    Notes
    -----
    Uses inverse transform method with correlated uniform variates.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Laplace_distribution
    """
    
    _, meanX, sdvnX, n, _ = _ppp_(0, meanX, sdvnX)

    _, _, U = correlated_rvs(R, n, N, seed)

    X = inv(U, meanX, sdvnX)

    return X
