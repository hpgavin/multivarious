import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs


# generic pre processing of parameters (ppp) 

def _ppp_(x, meanX):
    '''
    Validate and preprocess input parameters for consistency and correctness.

    Parameters:
        x : array_like
            Evaluation points
        a : float
            Minimum of the distribution
        b : float
            Maximum of the distribution (must be > a)
        q : float
            First shape parameter
        p : float
            Second shape parameter
    ''' 

    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    x = np.atleast_1d(x).astype(float)
    meanX = np.atleast_1d(meanX).reshape(-1,1).astype(float)
    n = len(meanX)   
        
    # Validate parameter values 
    if np.any(meanX <= 0):
        raise ValueError("rayleigh: all meanX values must be greater than zero")

    # Replace non-positive values to prevent invalid evaluation
    x[x <= 0] =  np.sum(meanX)/(n*1e3)

    # Convert mean meanX to modeX using Rayleigh identity
    modeX = meanX * np.sqrt(2 / np.pi)

    return x, meanX, modeX, n


def pdf(x, meanX):
    '''
    rayleigh.pdf

    Computes the PDF of the Rayleigh distribution using the mean parameter meanX.

    Input:
        X : array_like
            Evaluation points
        meanX : float
            Mean of the Rayleigh distribution (must be > 0)

    Output:
        f : ndarray
            PDF values at each point in X

    Reference:
    https://en.wikipedia.org/wiki/Rayleigh_distribution
    '''

    x, meanX, modeX, n = _ppp_(x, meanX)

    # Apply the Rayleigh PDF formula
    f = (x / modeX**2) * np.exp(-0.5 * (x / modeX)**2)

    return f


def cdf(x, meanX):
    '''
    rayleigh.cdf
    Computes the CDF of the Rayleigh distribution using the mean parameter meanX.

    Input:
        X : array_like
            Evaluation points
        meanX : float
            Mean of the Rayleigh distribution (must be > 0)

    Output:
        F : ndarray
            CDF values at each point in X

    Reference:
    https://en.wikipedia.org/wiki/Rayleigh_distribution
    '''

    x, meanX, modeX, n = _ppp_(x, meanX)

    # Apply the Rayleigh CDF formula
    F = 1.0 - np.exp(-0.5 * (X / modeX)**2)

    return F


def inv(P, meanX):
    '''
    rayleigh.inv

    Computes the inverse CDF (quantile function) of the Rayleigh distribution
    using the mean parameter meanX.

    INPUT:
        P : array_like
            Non-exceedance probabilities (0 ≤ P ≤ 1)
        meanX : float
            Mean of the Rayleigh distribution (must be > 0)

    OUTPUT:
        x : ndarray
            Quantile values corresponding to probabilities P

    Reference:
    https://en.wikipedia.org/wiki/Rayleigh_distribution
    '''

    _, meanX, modeX, n = _ppp_(0, meanX)

    P = np.asarray(P, dtype=float).copy()

    # Clamp values: ensure P stays in [0, 1] 
    P[P <= 0] = 0.0
    P[P >= 1] = 1.0

    # Compute the inverse CDF formula
    x = modeX * np.sqrt(-2.0 * np.log(1 - P))

    return x

def rnd(meanX, N, R=None, seed=None):
    '''
    rayleigh.rnd

    Generates random samples from the Rayleigh distribution using the mean
    parameter meanX. Samples can be generated either by passing a matrix of
    uniform random numbers or by specifying the output dimensions.

    Input:
        meanX : float (n,1)
        N : int, optional
            Number of observations of n rayleigh random variables
        R : float (n,n) , optional
            correlation matrix

    Output:
        X : ndarray
            Random samples drawn from the Rayleigh distribution

    Reference
    ----------
    https://en.wikipedia.org/wiki/Rayleigh_distribution
    '''

    _, meanX, modeX, n = _ppp_(0, meanX)

    _, _, U = correlated_rvs( R, n, N, seed )

    # Inverse transform sampling
    X = modeX * np.sqrt(-2.0 * np.log(U))

    if n == 1:
        X = X.flatten()

    return X
