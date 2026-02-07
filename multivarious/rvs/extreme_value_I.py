# multivarious/distributions/extreme_value_I.py

import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs


# Euler-Mascheroni constant
GAMMA = 0.57721566490153286060651209008240243104215933593992

def _meanX_covnX_to_loc_scale(meanX, covnX):
    """Helper: Convert (meanX, covnX) to (location, scale) for Gumbel."""
    sigma = meanX * covnX
    scale = np.sqrt(6) * sigma / np.pi
    loc = meanX - scale * GAMMA
    return loc, scale


def pdf(x, meanX, covnX):
    """
    PDF of Extreme Value Type I (Gumbel) distribution, param'd by (meanX, covnX).
    """
    loc, scale = _meanX_covnX_to_loc_scale(meanX, covnX)
    z = (x - loc) / scale
    exp_z = np.exp(-z)
    return exp_z * np.exp(-exp_z) / scale


def cdf(x, params):
    """
    CDF of Extreme Value Type I (Gumbel) distribution, param'd by array [meanX, covnX].
    """
    
    meanX, covnX = params
    
    loc, scale = _meanX_covnX_to_loc_scale(meanX, covnX)
    z = (x - loc) / scale
    return np.exp(-np.exp(-z))


def inv(p, meanX, covnX):
    """
    Inverse CDF (quantile) of Extreme Value Type I (Gumbel) distribution.
    """
    loc, scale = _meanX_covnX_to_loc_scale(meanX, covnX)
    return loc - scale * np.log(-np.log(p))


def rnd(meanX, covnX, N, R=None, seed=None):
    """
    Generate samples from the Extreme Value Type I distribution.

    Args:
        meanX : mean (scalar or array-like)
        covnX : coefficient of variation (scalar or array-like)
        N   : number of values of each random variable 
        R   : correlation matrix (n,n) 

    Returns:
        X : random samples of shape (n,N) or shape of r
    """
    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    meanX = np.atleast_1d(meanX).astype(float)
    covnX = np.atleast_1d(covnX).astype(float)

    n = len(meanX)

    if not (len(meanX) == n and len(covnX) == n):  
       raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got meanX:{len(meanX)}, covnX:{len(covnX)}")

    if np.any(np.asarray(meanX) <= 0):
        raise ValueError("meanX must be > 0")
    if np.any(np.asarray(covnX) <= 0):
        raise ValueError("covnX must be > 0")

    loc, scale = _meanX_covnX_to_loc_scale(meanX, covnX)

    _, _, U = correlated_rvs( R, n, N, seed )

    # Apply transformation
    X = loc - scale * np.log(-np.log(U))

    if n == 1:
        X = X.flatten()

    return X
