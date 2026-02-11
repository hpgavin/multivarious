# multivarious/distributions/extreme_value_I.py

import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs


# Euler-Mascheroni constant
GAMMA = 0.57721566490153286060651209008240243104215933593992

def _ppp_(x, meanX, covnX):
    '''
    Validate and preprocess input parameters for consistency and correctness.

    Parameters:
        x : array_like
            Evaluation points
        meanX : float
            Minimum of the distribution
        covnX : float
    ''' 

    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    x = np.atleast_1d(x).astype(float)


    meanX = np.atleast_2d(meanX).reshape(-1,1).astype(float)
    covnX = np.atleast_2d(covnX).reshape(-1,1).astype(float)
    n = len(meanX)   
        
    # Validate parameter dimensions 
    if not (len(meanX) == n and len(covnX) == n):
        raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got meanX:{len(meanX)}, covnX:{len(covnX)}, q:{len(q)}, p:{len(p)}")

    # Validate parameter values 
    if np.any(meanX <= 0):
        raise ValueError("extreme_value_I: meanX must be > 0")
    if np.any(covnX <= 0):
        raise ValueError("extreme_value_I: covnX must be > 0")

    sigma = meanX * covnX
    scale = np.sqrt(6.0) * sigma / np.pi
    loctn = meanX - scale * GAMMA

    return x, meanX, covnX, loctn, scale, n


def pdf(x, meanX, covnX):
    """
    PDF of Extreme Value Type I (Gumbel) distribution, param'd by (meanX, covnX).
    """

    x, _, _, loctn, scale, _ = _ppp_(x, meanX, covnX)

    z = (x - loctn) / scale
    exp_z = np.exp(-z)
    return exp_z * np.exp(-exp_z) / scale


def cdf(x, params):
    """
    CDF of Extreme Value Type I (Gumbel) distribution, param'd by array [meanX, covnX].
    """
    
    meanX, covnX = params
    
    x, _, _, loctn, scale, _ = _ppp_(x, meanX, covnX)

    z = (x - loc) / scale

    return np.exp(-np.exp(-z))


def inv(p, meanX, covnX):
    """
    Inverse CDF (quantile) of Extreme Value Type I (Gumbel) distribution.
    """

    _, _, _, loctn, scale, _ = _ppp_(x, meanX, covnX)

    x = loctn - scale * np.log(-np.log(p))

    return x


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
    _, _, _, loctn, scale, n  = _ppp_(0, meanX, covnX)

    # Correlated standard uniform values (n,N)
    _, _, U = correlated_rvs( R, n, N, seed )

    # Apply transformation
    X = loctn - scale * np.log(-np.log(U))

    if n == 1:
        X = X.flatten()

    return X
