import numpy as np
from scipy.stats import norm

# ------------------------------------------------------
# LOGNORMAL DISTRIBUTION (Custom Parameterization)
# ------------------------------------------------------
# This implementation uses:
# - medX: median of the lognormal variable X
# - covX: coefficient of variation of X (std(X) / mean(X))
#
# MATLAB version uses this parameterization instead of shape/scale.
# ------------------------------------------------------

def _convert_med_cov_to_log_params(medX, covX):
    """
    Convert (medX, covX) to parameters of the underlying normal distribution.

    Returns:
    - mu_lnX : mean of log(X)
    - sigma_lnX : std dev of log(X)
    """
    sigma_lnX = np.sqrt(np.log(1 + covX**2))  # log-space std dev
    mu_lnX = np.log(medX)                     # log-space mean is log(median)
    return mu_lnX, sigma_lnX

# ------------------------------------------------------
# PDF: Probability Density Function
# ------------------------------------------------------
def lognormal_pdf(x, medX, covX):
    """
    Compute PDF of lognormal distribution from median and cov.

    Parameters:
    - x : values to evaluate (scalar or array)
    - medX : median of X
    - covX : coefficient of variation of X

    Returns:
    - pdf : evaluated PDF values
    """
    mu, sigma = _convert_med_cov_to_log_params(medX, covX)
    with np.errstate(divide='ignore'):
        pdf = (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
    pdf[x <= 0] = 0  # ensure support is x > 0
    return pdf

# ------------------------------------------------------
# CDF: Cumulative Distribution Function
# ------------------------------------------------------
def lognormal_cdf(x, medX, covX):
    """
    Compute CDF of lognormal distribution.

    Parameters:
    - x : values to evaluate (scalar or array)
    - medX : median of X
    - covX : coefficient of variation of X

    Returns:
    - cdf : evaluated CDF values
    """
    mu, sigma = _convert_med_cov_to_log_params(medX, covX)
    cdf = norm.cdf((np.log(x) - mu) / sigma)
    cdf[x <= 0] = 0  # CDF is 0 for x <= 0
    return cdf

# ------------------------------------------------------
# INV: Inverse CDF (Quantile Function)
# ------------------------------------------------------
def lognormal_inv(p, medX, covX):
    """
    Compute inverse CDF (quantile) of lognormal distribution.

    Parameters:
    - p : probabilities in [0, 1]
    - medX : median of X
    - covX : coefficient of variation of X

    Returns:
    - quantiles corresponding to p
    """
    mu, sigma = _convert_med_cov_to_log_params(medX, covX)
    return np.exp(mu + sigma * norm.ppf(p))

# ------------------------------------------------------
# RND: Random Variates from Lognormal Distribution
# ------------------------------------------------------
def lognormal_rnd(medX, covX, size=(1,), seed=None):
    """
    Generate random samples from lognormal distribution.

    Parameters:
    - medX : median of X
    - covX : coefficient of variation of X
    - size : tuple indicating shape of output
    - seed : random seed or generator

    Returns:
    - samples : array of shape `size`
    """
    mu, sigma = _convert_med_cov_to_log_params(medX, covX)
    rng = np.random.default_rng(seed)
    return rng.lognormal(mean=mu, sigma=sigma, size=size)
