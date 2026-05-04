#! /usr/bin/env -S python3 -i
## rayleigh distribution
# github.com/hpgavin/multivarious ... rvs/rayleigh

import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs


def _validate_(meanX):
    """
    Validate and preprocess Rayleigh distribution parameters.

    Converts meanX to an (n, 1) column array for broadcasting against
    a (1, N) row array of evaluation points, producing (n, N) output.
    Derives the scale parameter modeX (sigma) from meanX.

    INPUTS
        meanX : float or array_like   mean(s) of the distribution, must be > 0

    OUTPUTS
        meanX : ndarray, shape (n, 1)
        modeX : ndarray, shape (n, 1)   scale parameter sigma = meanX * sqrt(2/pi)
    """
    meanX = np.asarray(meanX, dtype=float).reshape(-1, 1)  # (n, 1)

    if np.any(meanX <= 0):
        raise ValueError("rayleigh: all meanX values must be greater than zero")

    # Convert mean to scale parameter sigma via the Rayleigh identity
    modeX = meanX * np.sqrt(2.0 / np.pi)                   # (n, 1)

    return meanX, modeX


def pdf(x, meanX):
    """
    rayleigh.pdf

    Computes the PDF of the Rayleigh distribution.

    INPUTS
        x     : float or array_like, shape (N,)   evaluation points (x >= 0)
        meanX : float or array_like, shape (n,)   mean(s), must be > 0

    OUTPUTS
        f : ndarray, shape (n, N)   PDF values; singleton axes are squeezed

    Notes
    -----
    The Rayleigh distribution with scale parameter sigma (mode) has:
        f(x) = (x / sigma^2) * exp(-x^2 / (2 sigma^2))   for x >= 0
    where sigma = meanX * sqrt(2 / pi)

    Values x <= 0 are replaced by eps to avoid division by zero.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Rayleigh_distribution
    """
    meanX, modeX = _validate_(meanX)                        # (n, 1)
    x = np.asarray(x, dtype=float).reshape( 1, -1)        # (1, N)
    x = np.where(x <= 0, np.finfo(float).eps, x)          # guard against x <= 0

    f = (x / modeX**2) * np.exp(-0.5 * (x / modeX)**2)   # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [meanX, x].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([meanX, x]) if v.size == 1)

    return np.squeeze(f, axis=squeeze_axes)


def cdf(x, meanX):
    """
    rayleigh.cdf

    Computes the CDF of the Rayleigh distribution.

    INPUTS
        x     : float or array_like, shape (N,)   evaluation points (x >= 0)
        meanX : float or array_like, shape (n,)   mean(s), must be > 0

    OUTPUTS
        F : ndarray, shape (n, N)   CDF values; singleton axes are squeezed

    Notes
    -----
    F(x) = 1 - exp(-x^2 / (2 sigma^2))   for x >= 0
    where sigma = meanX * sqrt(2 / pi)

    Values x <= 0 are replaced by eps to avoid undefined results.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Rayleigh_distribution
    """
    meanX, modeX = _validate_(meanX)                        # (n, 1)
    x = np.asarray(x, dtype=float).reshape( 1, -1)        # (1, N)
    x = np.where(x <= 0, np.finfo(float).eps, x)          # guard against x <= 0

    F = 1.0 - np.exp(-0.5 * (x / modeX)**2)               # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [meanX, x].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([meanX, x]) if v.size == 1)

    return np.squeeze(F, axis=squeeze_axes)


def inv(F, meanX):
    """
    rayleigh.inv

    Computes the inverse CDF (quantile function) of the Rayleigh distribution.

    INPUTS
        F     : float or array_like, shape (N,)   probability values in [0, 1]
        meanX : float or array_like, shape (n,)   mean(s), must be > 0

    OUTPUTS
        x : ndarray, shape (n, N)   quantile values; singleton axes are squeezed

    Notes
    -----
    x = sigma * sqrt(-2 * log(1 - F))
    where sigma = meanX * sqrt(2 / pi)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Rayleigh_distribution
    """
    meanX, modeX = _validate_(meanX)                        # (n, 1)
    F = np.asarray(F, dtype=float).reshape( 1, -1)        # (1, N)
    F = np.clip(F, 0.0, 1.0 - np.finfo(float).eps)       # guard against log(0)

    x = modeX * np.sqrt(-2.0 * np.log(1.0 - F))           # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [meanX, F].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([meanX, F]) if v.size == 1)

    return np.squeeze(x, axis=squeeze_axes)


def rnd(meanX, N, R=None, seed=None):
    """
    rayleigh.rnd

    Generate random samples from the Rayleigh distribution.

    INPUTS
        meanX : float or array_like, shape (n,)   mean(s), must be > 0
        N     : int                                number of samples per variable
        R     : ndarray, shape (n, n), optional    correlation matrix;
                if None, generates uncorrelated samples
        seed  : int or None                        random seed for reproducibility

    OUTPUTS
        X : ndarray, shape (n, N)   Rayleigh random samples;
            each row is one random variable, each column one sample;
            singleton axes are squeezed

    Notes
    -----
    Uses the inverse transform method: X = sigma * sqrt(-2 * log(U))
    where U ~ Uniform(0, 1) and sigma = meanX * sqrt(2 / pi).
    Since 1 - U ~ Uniform(0, 1) as well, log(U) and log(1-U) are
    equivalent in distribution, so the simpler form log(U) is used directly.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Rayleigh_distribution
    """
    if N is None or N < 1:
        raise ValueError("rayleigh.rnd: N must be greater than zero")

    meanX, modeX = _validate_(meanX)                       # (n, 1)
    n = meanX.size

    # Generate n correlated uniform [0, 1] variates, shape (n, N)
    _, _, U = correlated_rvs(R, n, N, seed)

    # Inline the inverse transform rather than calling inv(), because U is
    # already (n, N) from correlated_rvs, while inv() expects F as (1, N).
    X = modeX * np.sqrt(-2.0 * np.log(U))                 # (n, N)

    # Squeeze singleton axes from the (n, N) output
    squeeze_axes = tuple(np.where(np.asarray([n, N]) == 1)[0])

    return np.squeeze(X, axis=squeeze_axes)
