## extreme_value_I distribution (Gumbel)
# github.com/hpgavin/multivarious ... rvs/extreme_value_I

import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs


# Euler-Mascheroni constant
GAMMA = 0.57721566490153286060651209008240243104215933593992


def _ppp_(x, meanX, covnX):
    """
    Validate and preprocess input parameters for consistency and correctness.

    INPUTS:
        x : array_like
            Evaluation points
        meanX : float or array_like
            Mean(s) of the distribution (must be > 0)
        covnX : float or array_like
            Coefficient(s) of variation (must be > 0)

    OUTPUTS:
        x : ndarray
            Evaluation points as array
        meanX : ndarray
            Means as column array
        covnX : ndarray
            Coefficients of variation as column array
        loctn : ndarray
            Location parameters (μ) as column array
        scale : ndarray
            Scale parameters (σ) as column array
        n : int
            Number of random variables
    """ 

    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    x = np.atleast_1d(x).astype(float)

    meanX = np.atleast_1d(meanX).reshape(-1,1).astype(float)
    covnX = np.atleast_1d(covnX).reshape(-1,1).astype(float)
    n = len(meanX)   
        
    # Validate parameter dimensions 
    if not (len(meanX) == n and len(covnX) == n):
        raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got meanX:{len(meanX)}, covnX:{len(covnX)}")

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
    extreme_value_I.pdf
    
    Computes the PDF of Extreme Value Type I (Gumbel) distribution, 
    parameterized by mean and coefficient of variation.

    INPUTS:
        x : array_like
            Evaluation points
        meanX : float or array_like, shape (n,)
            Mean(s) of the distribution (must be > 0)
        covnX : float or array_like, shape (n,)
            Coefficient(s) of variation (must be > 0)

    OUTPUTS:
        f : ndarray, shape (n, N)
            PDF values at each point in x for each of n random variables

    Notes
    -----
    The Gumbel distribution has PDF:
    f(x) = (1/σ) exp(-z - exp(-z)) where z = (x-μ)/σ
    with μ = mean - σ·γ and σ = √6·stddev/π
    where γ is the Euler-Mascheroni constant ≈ 0.5772

    Reference
    ---------
    https://en.wikipedia.org/wiki/Gumbel_distribution
    """

    x, _, _, loctn, scale, n = _ppp_(x, meanX, covnX)

    z = (x - loctn) / scale
    exp_z = np.exp(-z)
    f = exp_z * np.exp(-exp_z) / scale

    if n == 1 and f.shape[0] == 1:
        f = f.flatten()

    return f


def cdf(x, params):
    """
    extreme_value_I.cdf
    
    Computes the CDF of Extreme Value Type I (Gumbel) distribution, 
    parameterized by array [meanX, covnX].

    INPUTS:
        x : array_like
            Evaluation points
        params : array_like [meanX, covnX]
            meanX : float or array_like
                Mean(s) of the distribution (must be > 0)
            covnX : float or array_like
                Coefficient(s) of variation (must be > 0)

    OUTPUTS:
        F : ndarray, shape (n, N)
            CDF values at each point in x for each of n random variables

    Notes
    -----
    F(x) = exp(-exp(-z)) where z = (x-μ)/σ

    Reference
    ---------
    https://en.wikipedia.org/wiki/Gumbel_distribution
    """
    
    meanX, covnX = params
    
    x, _, _, loctn, scale, _ = _ppp_(x, meanX, covnX)

    z = (x - loctn) / scale

    F = np.exp(-np.exp(-z))

    if n == 1 and F.shape[0] == 1:
        F = F.flatten()

    return F


def inv(F, meanX, covnX):
    """
    extreme_value_I.inv
    
    Computes the inverse CDF (quantile function) of Extreme Value Type I 
    (Gumbel) distribution.

    INPUTS:
        F : array_like
            Probability values (must be in (0, 1))
        meanX : float or array_like, shape (n,)
            Mean(s) of the distribution (must be > 0)
        covnX : float or array_like, shape (n,)
            Coefficient(s) of variation (must be > 0)

    OUTPUTS:
        x : ndarray
            Quantile values corresponding to probabilities F

    Notes
    -----
    x = μ - σ ln(-ln(F))

    Reference
    ---------
    https://en.wikipedia.org/wiki/Gumbel_distribution
    """

    _, _, _, loctn, scale, _ = _ppp_(0, meanX, covnX)

    x = loctn - scale * np.log(-np.log(F))

    if n == 1 and x.shape[0] == 1:
        x = x.flatten()

    return x


def rnd(meanX, covnX, N, R=None, seed=None):
    """
    extreme_value_I.rnd
    
    Generate samples from the Extreme Value Type I (Gumbel) distribution.

    INPUTS:
        meanX : float or array_like, shape (n,)
            Mean(s) of the distribution (must be > 0)
        covnX : float or array_like, shape (n,)
            Coefficient(s) of variation (must be > 0)
        N : int
            Number of observations per random variable
        R : ndarray, shape (n, n), optional
            Correlation matrix for generating correlated samples.
            If None, generates uncorrelated samples.
        seed : int, optional
            Random seed for reproducibility

    OUTPUTS:
        X : ndarray, shape (n, N) or shape (N,) if n=1
            Random samples from the Gumbel distribution.
            Each row corresponds to one random variable.
            Each column corresponds to one sample.

    Notes
    -----
    Uses inverse transform method: x = μ - σ ln(-ln(u)) where u ~ Uniform(0,1)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Gumbel_distribution
    """
    
    _, _, _, loctn, scale, n = _ppp_(0, meanX, covnX)

    # Correlated standard uniform values (n,N)
    _, _, U = correlated_rvs(R, n, N, seed)

    # Apply transformation
    X = loctn - scale * np.log(-np.log(U))

    if n == 1 and X.shape[0] == 1:
        X = X.flatten()

    return X
