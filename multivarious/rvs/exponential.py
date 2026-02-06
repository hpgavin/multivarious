# exponential distribution
# github.com/hpgavin/multivarious ... rvs/exponential

import numpy as np
from scipy.stats import norm

from multivarious.utl.correlated_rvs import correlated_rvs


def pdf(x, meanX):
    '''
    exponential.pdf

    Computes the Probability Density Function (PDF) of the exponential distribution.

    INPUTS:
      x    = evaluation points (must be x >= 0)
      meanX  = mean of the exponential distribution
 
    OUTPUT:
      f    = PDF evaluated at x
 
    FORMULA:
      f(x) = (1/meanX) * exp(-x / meanX), for x >= 0
    '''
    x = np.asarray(x, dtype=float)

    # Prevent negative or zero values (log not defined)
    x = np.where(x < 0, 0.01, x)

    f = np.exp(-x / meanX) / meanX
    
    return f


def cdf(x, meanX):
    '''
    exponential.cdf

    Computes the Cumulative Distribution Function (CDF) of the exponential.

    INPUTS:
      x    = values at which to evaluate the CDF (x >= 0)
      meanX  = mean of the exponential distribution
 
    OUTPUT:
      F    = CDF values at each x
 
    FORMULA:
      F(x) = 1 - exp(-x / meanX), for x >= 0
    '''
    x = np.asarray(x, dtype=float)

    # Prevent issues with x <= 0
    x = np.where(x <= 0, 0.01, x)

    F = 1.0 - np.exp(-x / meanX)
    return F


def inv(P, meanX):
    '''
    exponential.inv

    Computes the inverse CDF (quantile function) of the exponential distribution.

    INPUTS:
      P    = probability values (0 <= P <= 1)
      meanX  = mean of the exponential distribution

    OUTPUT:
      X    = quantiles corresponding to P

    FORMULA:
      X = -meanX * log(1 - P)
    '''
    P = np.asarray(P, dtype=float)

    # Clip invalid P values just like in MATLAB
    P = np.where(P < 0, 0.0, P)
    P = np.where(P > 1, 1.0, P)

    X = -meanX * np.log(1 - P)
    return X


def rnd(meanX, N, R=None, seed=None):
    '''
    exponential.rnd

    Generate random samples from an exponential distribution with mean meanX.

    INPUTS:
        meanX  = mean (n,) one component for each r.v.
        N = number of values for each variable in the sample (columns)
        R = correlation matrix (n x n) - not yet implemented
 
    OUTPUT:
        X    = random samples shaped (n, N)

    METHOD:
        Use inverse CDF method: x = -meanX * log(U), where U ~ Uniform(0,1)
    '''

    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    meanX = np.atleast_1d(meanX).astype(float)

    print("meanX:\n", meanX)

    # Check parameter validity
    if np.any(meanX <= 0) or np.any(np.isinf(meanX)):
        raise ValueError(f"exponential.rnd: meanX must be > 0 and finite")

    n = len(meanX)

    _, _, U = correlated_rvs( R, n, N, seed )

    # Inverse transform: x = -meanX * log(U)
    X = -meanX * np.log(U)

    if n == 1:
        X = X.flatten()

    return X
