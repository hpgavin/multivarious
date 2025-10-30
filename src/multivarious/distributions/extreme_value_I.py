# multivarious/distributions/extreme_value_I.py

import numpy as np

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


def rnd(muX, cvX, r, c=None):
    """
    Generate samples from the Extreme Value Type I distribution.

    Args:
        muX : mean (scalar or array-like)
        cvX : coefficient of variation (scalar or array-like)
        r   : rows OR uniform sample matrix
        c   : cols (optional)

    Returns:
        x : random samples of shape (r,c) or shape of r
    """
    if np.any(np.asarray(muX) <= 0):
        raise ValueError("muX must be > 0")
    if np.any(np.asarray(cvX) <= 0):
        raise ValueError("cvX must be > 0")

    if c is None:
        u = np.asarray(r)
    else:
        u = np.random.rand(r, c)

    muX = np.broadcast_to(muX, u.shape)
    cvX = np.broadcast_to(cvX, u.shape)
    sigma = cvX * muX
    scale = np.sqrt(6) * sigma / np.pi
    loc = muX - scale * GAMMA

    return loc - scale * np.log(-np.log(u))
