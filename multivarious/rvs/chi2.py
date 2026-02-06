# chi-squared distribution
# github.com/hpgavin/multivarious ... rvs/chi2

import numpy as np
from scipy.stats import norm as scipy_norm

from multivarious.utl.correlated_rvs import correlated_rvs

def pdf(x, k):
    '''
    chi2.pdf

    Computes the PDF of the Chi-squared distribution using the 
    Wilson-Hilferty transformation.

    INPUTS:
      x : array-like
          Points to evaluate the PDF
      k : float
          Degrees of freedom (must be positive)

    OUTPUT:
      f : array-like
          Approximate PDF values at x
    '''
    x = np.asarray(x, dtype=float)
    
    # Wilson-Hilferty transformation parameters
    m = 1 - 2 / (9 * k)             # Approximate mean of Z = (X/k)^{1/3}
    s = np.sqrt(2 / (9 * k))        # Approximate std. dev. of Z

    # Transform x into z-space: Z = (X / k)^{1/3}
    z = (x / k) ** (1/3)

    # Approximate PDF using normal distribution
    f = scipy_norm.pdf(z, m, s)

    return f


def cdf(x, k):
    '''
    chi2.cdf

    Approximates the CDF of the chi-squared distribution using the
    Wilson-Hilferty transformation, which maps chi2(k) into a normal distribution.

    INPUTS:
      x = evaluation points
      k = degrees of freedom (must be > 0)
 
    OUTPUT:
      F = approximate cumulative probability evaluated at x
    '''
    x = np.asarray(x, dtype=float)
    if k <= 0:
        raise ValueError(" chi2.cdf: Degrees of freedom k must be > 0")

    # Wilson-Hilferty approximation parameters
    m = 1 - 2 / (9 * k)         # mean of cube-root-transformed variable
    s = np.sqrt(2 / (9 * k))    # std dev of cube-root-transformed variable

    # Apply transformation: (X/k)^(1/3)
    z = (x / k) ** (1 / 3)

    # Apply normal CDF using transformed variable
    F = scipy_norm.cdf(z, loc=m, scale=s)

    return F


def inv(p, k):
    '''
    chi2.inv
 
    Approximates the inverse CDF (quantile function) of the chi-squared distribution
    using the Wilson-Hilferty transformation.
 
    INPUTS:
      p = non-exceedance probabilities (values between 0 and 1)
      k = degrees of freedom (must be > 0)
 
    OUTPUT:
      x = quantile values such that Prob[X ≤ x] = p
    '''
    p = np.asarray(p, dtype=float)
    if k <= 0:
        raise ValueError(" chi2.inv: Degrees of freedom k must be > 0")

    # Wilson-Hilferty transformation parameters
    m = 1 - 2 / (9 * k)         # mean of cube-root-transformed variable
    s = np.sqrt(2 / (9 * k))    # std dev of cube-root-transformed variable

    # Inverse normal CDF
    z = scipy_norm.ppf(p, loc=m, scale=s)

    # Apply inverse transformation: x = k * z³
    x = k * z**3

    return x


def rnd(k, N, R=None, seed=None):
    '''
    chi2.rnd
    Generate N observations of n correlated (or uncorrelated) chi2 random var's
    via the Wilson-Hilferty transformation (a good approximation).

    INPUTS:
      k = degrees of freedom (must be > 0), (n,) one component for each r.v.
      N = number of values for each random variable in the sample (columns)
      R = correlation matrix (n x n) - not yet implemented
 
    OUTPUT:
      X = random samples from Chi-squared(k), shape (n, N)
    '''

    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    k = np.atleast_1d(k).astype(int)

    # Validate k is not negative 
    if np.any(k <= 0) or np.any(np.isinf(k)):
        raise ValueError(" chi2.rnd: Degrees of freedom k must be > 0")

    n = len(k) 

    # Wilson-Hilferty transformation parameters
    m = 1 - 2 / (9 * k)
    s = np.sqrt(2 / (9 * k))

    R, Y, U = correlated_rvs( R, n, N, seed )
   
    # Apply transformation
    X = np.zeros((n, N))
    for i in range(n):
        X[i, :] = k[i] * ( m[i] + s[i] * Y[i, :]) ** 3

    if n == 1:
        X = X.flatten()

    return X
