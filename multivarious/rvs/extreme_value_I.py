# multivarious/distributions/extreme_value_I.py

import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs


# Euler-Mascheroni constant
GAMMA = 0.57721566490153286060651209008240243104215933593992

def _mu_cv_to_loc_scale(mu, cv):
    """Helper: Convert (mu, cv) to (location, scale) for Gumbel."""
    sigma = mu * cv
    scale = np.sqrt(6) * sigma / np.pi
    loc = mu - scale * GAMMA
    return loc, scale


def pdf(x, mu, cv):
    """
    PDF of Extreme Value Type I (Gumbel) distribution, param'd by (mu, cv).
    """
    loc, scale = _mu_cv_to_loc_scale(mu, cv)
    z = (x - loc) / scale
    exp_z = np.exp(-z)
    return exp_z * np.exp(-exp_z) / scale


def cdf(x, params):
    """
    CDF of Extreme Value Type I (Gumbel) distribution, param'd by [mu, cv].
    """
    mu, cv = params
    loc, scale = _mu_cv_to_loc_scale(mu, cv)
    z = (x - loc) / scale
    return np.exp(-np.exp(-z))


def inv(p, mu, sigma):
    """
    Inverse CDF (quantile) of Extreme Value Type I (Gumbel) distribution.
    """
    scale = np.sqrt(6) * sigma / np.pi
    loc = mu - scale * GAMMA
    return loc - scale * np.log(-np.log(p))


def rnd(muX, cvX, N, R=None):
    """
    Generate samples from the Extreme Value Type I distribution.

    Args:
        muX : mean (scalar or array-like)
        cvX : coefficient of variation (scalar or array-like)
        N   : number of values of each random variable 
        R   : correlation matrix (n,n) 

    Returns:
        X : random samples of shape (n,N) or shape of r
    """
    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    muX = np.atleast_1d(muX).astype(float)
    cvX = np.atleast_1d(cvX).astype(float)

    n = len(muX)

    if not (len(muX) == n and len(cvX) == n):  
       raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got muX:{len(muX)}, cvX:{len(cvX)}")

    if np.any(np.asarray(muX) <= 0):
        raise ValueError("muX must be > 0")
    if np.any(np.asarray(cvX) <= 0):
        raise ValueError("cvX must be > 0")

    _, _, U = correlated_rvs(R,n,N)

    sigma = cvX * muX
    scale = np.sqrt(6) * sigma / np.pi
    loc = muX - scale * GAMMA

    X = loc - scale * np.log(-np.log(U))

    if n == 1:
        X = X.flatten()

    return
