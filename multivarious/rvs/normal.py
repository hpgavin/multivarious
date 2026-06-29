#! /usr/bin/env -S python3 -i
## normal distribution
# github.com/hpgavin/multivarious ... rvs/normal

import numpy as np
from scipy.special import erf    as scipy_erf
from scipy.special import erfinv as scipy_erfinv

from multivarious.utl.correlated_rvs import correlated_rvs


def _validate_(meanX, sdvnX):
    """
    Validate and preprocess normal distribution parameters.

    Converts meanX and sdvnX to (n, 1) column arrays for broadcasting
    against a (1, N) row array of evaluation points, producing (n, N) output.

    INPUTS
        meanX : float or array_like   mean(s) of the distribution
        sdvnX : float or array_like   standard deviation(s), must be > 0

    OUTPUTS
        meanX : ndarray, shape (n, 1)
        sdvnX : ndarray, shape (n, 1)
    """
    meanX = np.asarray(meanX, dtype=float).reshape(-1, 1)  # (n, 1)
    sdvnX = np.asarray(sdvnX, dtype=float).reshape(-1, 1)  # (n, 1)

    if meanX.shape != sdvnX.shape:
        raise ValueError(f"normal: meanX and sdvnX must have the same length. "
                         f"Got meanX:{meanX.size}, sdvnX:{sdvnX.size}")
    if np.any(sdvnX <= 0):
        raise ValueError("normal: sdvnX must be > 0")

    return meanX, sdvnX


def pdf(x, meanX=0.0, sdvnX=1.0):
    """
    normal.pdf

    Computes the PDF of the normal distribution N(meanX, sdvnX^2).

    INPUTS
        x     : float or array_like, shape (N,)   evaluation points
        meanX : float or array_like, shape (n,)   mean(s)
        sdvnX : float or array_like, shape (n,)   standard deviation(s), > 0

    OUTPUTS
        f : ndarray, shape (n, N)   PDF values; singleton axes are squeezed

    Notes
    -----
    f(x) = (1 / sqrt(2 pi sigma^2)) * exp(-(x - mu)^2 / (2 sigma^2))

    Reference
    ---------
    https://en.wikipedia.org/wiki/Normal_distribution
    """
    meanX, sdvnX = _validate_(meanX, sdvnX)                 # (n, 1)
    x = np.asarray(x, dtype=float).reshape( 1, -1)         # (1, N)

    z = (x - meanX) / sdvnX                                 # (n, N) standardized

    f = np.exp(-0.5 * z**2) / (sdvnX * np.sqrt(2.0 * np.pi))  # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [meanX, x].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([meanX, x]) if v.size == 1)

    return np.squeeze(f, axis=squeeze_axes)


def cdf(x, params=(0.0, 1.0)):
    """
    normal.cdf

    Computes the CDF of the normal distribution N(meanX, sdvnX^2).

    INPUTS
        x      : float or array_like, shape (N,)   evaluation points
        params : tuple (meanX, sdvnX)
            meanX : float or array_like, shape (n,)   mean(s)
            sdvnX : float or array_like, shape (n,)   standard deviation(s), > 0

    OUTPUTS
        F : ndarray, shape (n, N)   CDF values; singleton axes are squeezed

    Notes
    -----
    F(x) = (1 + erf((x - mu) / (sigma * sqrt(2)))) / 2

    Reference
    ---------
    https://en.wikipedia.org/wiki/Normal_distribution
    """
    meanX, sdvnX = params
    meanX, sdvnX = _validate_(meanX, sdvnX)                 # (n, 1)
    x = np.asarray(x, dtype=float).reshape( 1, -1)         # (1, N)

    z = (x - meanX) / sdvnX                                 # (n, N) standardized

    F = (1.0 + scipy_erf(z / np.sqrt(2.0))) / 2.0          # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [meanX, x].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([meanX, x]) if v.size == 1)

    return np.squeeze(F, axis=squeeze_axes)


def inv(F, meanX=0.0, sdvnX=1.0):
    """
    normal.inv

    Computes the inverse CDF (quantile function) of the normal distribution
    N(meanX, sdvnX^2).

    INPUTS
        F     : float or array_like, shape (N,)   probability values in [0, 1]
        meanX : float or array_like, shape (n,)   mean(s)
        sdvnX : float or array_like, shape (n,)   standard deviation(s), > 0

    OUTPUTS
        x : ndarray, shape (n, N)   quantile values; singleton axes are squeezed

    Notes
    -----
    x = mu + sigma * sqrt(2) * erfinv(2F - 1)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Normal_distribution
    """
    meanX, sdvnX = _validate_(meanX, sdvnX)         # (n, 1)
    F = np.asarray(F, dtype=float)
    if F.ndim <= 1:
        F = F.reshape(1, -1)   # (1, N) - shared F grid for all n variables
    F = np.clip(F, np.finfo(float).eps, 1.0 - np.finfo(float).eps)

    z = np.sqrt(2.0) * scipy_erfinv(2.0 * F - 1.0)  # (n, N) standard normal quantile
    x = meanX + sdvnX * z                           # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [meanX, F].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([meanX, F]) if v.size == 1)

    return np.squeeze(x, axis=squeeze_axes)


def rnd(meanX=0.0, sdvnX=1.0, N=1, R=None, seed=None):
    """
    normal.rnd

    Generate correlated random samples from the normal distribution N(meanX, sdvnX^2).

    INPUTS
        meanX : float or array_like, shape (n,)   mean(s)
        sdvnX : float or array_like, shape (n,)   standard deviation(s), > 0
        N     : int                                number of samples per variable
        R     : ndarray, shape (n, n), optional    correlation matrix;
                if None, generates uncorrelated samples
        seed  : int or None                        random seed for reproducibility

    OUTPUTS
        X : ndarray, shape (n, N)   normal random samples;
            each row is one random variable, each column one sample;
            singleton axes are squeezed

    Notes
    -----
    Uses inverse transform method with correlated uniform variates,
    then transforms to the desired mean and standard deviation.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Normal_distribution
    """
    if N is None or N < 1:
        raise ValueError("normal.rnd: N must be greater than zero")

    meanX, sdvnX = _validate_(meanX, sdvnX)                 # (n, 1)
    n = meanX.size

    # Generate n correlated uniform [0, 1] variates, shape (n, N)
    _, _, U = correlated_rvs(R, n, N, seed)

    # Inline the inverse transform rather than calling inv(), because U is
    # already (n, N) from correlated_rvs, while inv() expects F as (1, N).
    z = np.sqrt(2.0) * scipy_erfinv(2.0 * U - 1.0)         # (n, N) standard normal quantile
    X = meanX + sdvnX * z                                   # (n, N)

    # Squeeze singleton axes from the (n, N) output
    squeeze_axes = tuple(np.where(np.asarray([n, N]) == 1)[0])

    return np.squeeze(X, axis=squeeze_axes)
