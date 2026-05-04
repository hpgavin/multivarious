#! /usr/bin/env -S python3 -i
## extreme_value_I distribution (Gumbel)
# github.com/hpgavin/multivarious ... rvs/extreme_value_I

import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs


# Euler-Mascheroni constant
GAMMA = 0.57721566490153286060651209008240243104215933593992


def _validate_(meanX, covnX):
    """
    Validate and preprocess Gumbel distribution parameters.

    Converts meanX and covnX to (n, 1) column arrays for broadcasting against
    a (1, N) row array of evaluation points, producing (n, N) output.
    Derives the location parameter loctn and scale parameter scale.

    INPUTS
        meanX : float or array_like   mean(s),                  must be > 0
        covnX : float or array_like   coefficient(s) of variation, must be > 0

    OUTPUTS
        meanX : ndarray, shape (n, 1)
        covnX : ndarray, shape (n, 1)
        loctn : ndarray, shape (n, 1)   location mu  = meanX - scale * GAMMA
        scale : ndarray, shape (n, 1)   scale    sigma = sqrt(6) * meanX*covnX / pi
    """
    meanX = np.asarray(meanX, dtype=float).reshape(-1, 1)  # (n, 1)
    covnX = np.asarray(covnX, dtype=float).reshape(-1, 1)  # (n, 1)

    if meanX.shape != covnX.shape:
        raise ValueError(f"extreme_value_I: meanX and covnX must have the same length. "
                         f"Got meanX:{meanX.size}, covnX:{covnX.size}")
    if np.any(meanX <= 0):
        raise ValueError("extreme_value_I: meanX must be > 0")
    if np.any(covnX <= 0):
        raise ValueError("extreme_value_I: covnX must be > 0")

    scale = np.sqrt(6.0) * meanX * covnX / np.pi   # (n, 1) sigma
    loctn = meanX - scale * GAMMA                   # (n, 1) mu

    return meanX, covnX, loctn, scale


def pdf(x, meanX, covnX):
    """
    extreme_value_I.pdf

    Computes the PDF of the Extreme Value Type I (Gumbel) distribution,
    parameterized by mean and coefficient of variation.

    INPUTS
        x     : float or array_like, shape (N,)   evaluation points
        meanX : float or array_like, shape (n,)   mean(s),                  > 0
        covnX : float or array_like, shape (n,)   coefficient(s) of variation, > 0

    OUTPUTS
        f : ndarray, shape (n, N)   PDF values; singleton axes are squeezed

    Notes
    -----
    f(x) = (1/sigma) * exp(-z - exp(-z))   where z = (x - mu) / sigma
    mu    = meanX - sigma * GAMMA
    sigma = sqrt(6) * meanX * covnX / pi

    Reference
    ---------
    https://en.wikipedia.org/wiki/Gumbel_distribution
    """
    meanX, covnX, loctn, scale = _validate_(meanX, covnX)    # (n, 1)
    x = np.asarray(x, dtype=float).reshape( 1, -1)          # (1, N)

    z = (x - loctn) / scale                                   # (n, N) standardized
    f = np.exp(-z - np.exp(-z)) / scale                      # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [meanX, x].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([meanX, x]) if v.size == 1)

    return np.squeeze(f, axis=squeeze_axes)


def cdf(x, params):
    """
    extreme_value_I.cdf

    Computes the CDF of the Extreme Value Type I (Gumbel) distribution.

    INPUTS
        x      : float or array_like, shape (N,)   evaluation points
        params : tuple (meanX, covnX)
            meanX : float or array_like, shape (n,)   mean(s),                  > 0
            covnX : float or array_like, shape (n,)   coefficient(s) of variation, > 0

    OUTPUTS
        F : ndarray, shape (n, N)   CDF values; singleton axes are squeezed

    Notes
    -----
    F(x) = exp(-exp(-z))   where z = (x - mu) / sigma

    Reference
    ---------
    https://en.wikipedia.org/wiki/Gumbel_distribution
    """
    meanX, covnX = params
    meanX, covnX, loctn, scale = _validate_(meanX, covnX)    # (n, 1)
    x = np.asarray(x, dtype=float).reshape( 1, -1)          # (1, N)

    z = (x - loctn) / scale                                   # (n, N) standardized
    F = np.exp(-np.exp(-z))                                   # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [meanX, x].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([meanX, x]) if v.size == 1)

    return np.squeeze(F, axis=squeeze_axes)


def inv(F, meanX, covnX):
    """
    extreme_value_I.inv

    Computes the inverse CDF (quantile function) of the Extreme Value Type I
    (Gumbel) distribution.

    INPUTS
        F     : float or array_like, shape (N,)   probability values in (0, 1)
        meanX : float or array_like, shape (n,)   mean(s),                  > 0
        covnX : float or array_like, shape (n,)   coefficient(s) of variation, > 0

    OUTPUTS
        x : ndarray, shape (n, N)   quantile values; singleton axes are squeezed

    Notes
    -----
    x = mu - sigma * log(-log(F))

    Reference
    ---------
    https://en.wikipedia.org/wiki/Gumbel_distribution
    """
    meanX, covnX, loctn, scale = _validate_(meanX, covnX)    # (n, 1)
    if F.ndim <= 1:
        F = F.reshape(1, -1)   # (1, N) - shared F grid for all n variables
    F = np.clip(F, np.finfo(float).eps, 1.0 - np.finfo(float).eps)

    x = loctn - scale * np.log(-np.log(F))                   # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [meanX, F].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([meanX, F]) if v.size == 1)

    return np.squeeze(x, axis=squeeze_axes)


def rnd(meanX, covnX, N, R=None, seed=None):
    """
    extreme_value_I.rnd

    Generate random samples from the Extreme Value Type I (Gumbel) distribution.

    INPUTS
        meanX : float or array_like, shape (n,)   mean(s),                  > 0
        covnX : float or array_like, shape (n,)   coefficient(s) of variation, > 0
        N     : int                                number of samples per variable
        R     : ndarray, shape (n, n), optional    correlation matrix;
                if None, generates uncorrelated samples
        seed  : int or None                        random seed for reproducibility

    OUTPUTS
        X : ndarray, shape (n, N)   Gumbel random samples;
            each row is one random variable, each column one sample;
            singleton axes are squeezed

    Notes
    -----
    Uses the inverse transform method: X = mu - sigma * log(-log(U))
    Inlined rather than calling inv() because U is already (n, N)
    from correlated_rvs, while inv() expects F as (1, N).

    Reference
    ---------
    https://en.wikipedia.org/wiki/Gumbel_distribution
    """
    if N is None or N < 1:
        raise ValueError("extreme_value_I.rnd: N must be greater than zero")

    meanX, covnX, loctn, scale = _validate_(meanX, covnX)    # (n, 1)
    n = meanX.size

    # Generate n correlated uniform [0, 1] variates, shape (n, N)
    _, _, U = correlated_rvs(R, n, N, seed)

    # Inline the inverse transform rather than calling inv(), because U is
    # already (n, N) from correlated_rvs, while inv() expects F as (1, N).
    X = loctn - scale * np.log(-np.log(U))                   # (n, N)

    # Squeeze singleton axes from the (n, N) output
    squeeze_axes = tuple(np.where(np.asarray([n, N]) == 1)[0])

    return np.squeeze(X, axis=squeeze_axes)
