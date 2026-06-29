#! /usr/bin/env -S python3 -i
## laplace distribution
# github.com/hpgavin/multivarious ... rvs/laplace

import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs


def _validate_(meanX, sdvnX):
    """
    Validate and preprocess Laplace distribution parameters.

    Converts meanX and sdvnX to (n, 1) column arrays for broadcasting
    against a (1, N) row array of evaluation points, producing (n, N) output.

    INPUTS
        meanX : float or array_like   mean(s) / location parameter(s)
        sdvnX : float or array_like   standard deviation(s) / scale parameter(s), > 0

    OUTPUTS
        meanX : ndarray, shape (n, 1)
        sdvnX : ndarray, shape (n, 1)
    """
    meanX = np.asarray(meanX, dtype=float).reshape(-1, 1)  # (n, 1)
    sdvnX = np.asarray(sdvnX, dtype=float).reshape(-1, 1)  # (n, 1)

    if meanX.shape != sdvnX.shape:
        raise ValueError(f"laplace: meanX and sdvnX must have the same length. "
                         f"Got meanX:{meanX.size}, sdvnX:{sdvnX.size}")
    if np.any(sdvnX <= 0):
        raise ValueError("laplace: all sdvnX values must be > 0")

    return meanX, sdvnX


def pdf(x, meanX, sdvnX):
    """
    laplace.pdf

    Computes the PDF of the Laplace distribution.

    INPUTS
        x     : float or array_like, shape (N,)   evaluation points
        meanX : float or array_like, shape (n,)   mean(s) / location parameter(s)
        sdvnX : float or array_like, shape (n,)   standard deviation(s) / scale, > 0

    OUTPUTS
        f : ndarray, shape (n, N)   PDF values; singleton axes are squeezed

    Notes
    -----
    f(x) = (1 / (sqrt(2) * sigma)) * exp(-sqrt(2) * |x - mu| / sigma)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Laplace_distribution
    """
    meanX, sdvnX = _validate_(meanX, sdvnX)                  # (n, 1)
    x = np.asarray(x, dtype=float).reshape( 1, -1)          # (1, N)

    sr2 = np.sqrt(2.0)
    f = np.exp(-sr2 * np.abs(x - meanX) / sdvnX) \
        / (sr2 * sdvnX)                                       # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [meanX, x].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([meanX, x]) if v.size == 1)

    return np.squeeze(f, axis=squeeze_axes)


def cdf(x, params):
    """
    laplace.cdf

    Computes the CDF of the Laplace distribution.

    INPUTS
        x      : float or array_like, shape (N,)   evaluation points
        params : tuple (meanX, sdvnX)
            meanX : float or array_like, shape (n,)   mean(s) / location parameter(s)
            sdvnX : float or array_like, shape (n,)   standard deviation(s) / scale, > 0

    OUTPUTS
        F : ndarray, shape (n, N)   CDF values; singleton axes are squeezed

    Notes
    -----
    F(x) = 0.5 * exp( sqrt(2) * (x - mu) / sigma)   for x <= mu
    F(x) = 1 - 0.5 * exp(-sqrt(2) * (x - mu) / sigma)   for x > mu

    Reference
    ---------
    https://en.wikipedia.org/wiki/Laplace_distribution
    """
    meanX, sdvnX = params
    meanX, sdvnX = _validate_(meanX, sdvnX)                  # (n, 1)
    x = np.asarray(x, dtype=float).reshape( 1, -1)          # (1, N)

    sr2 = np.sqrt(2.0)
    z   = sr2 * (x - meanX) / sdvnX                          # (n, N) standardized

    F_left  =       0.5 * np.exp( z)                         # (n, N) branch values
    F_right = 1.0 - 0.5 * np.exp(-z)

    F = np.where(x <= meanX, F_left, F_right)                # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [meanX, x].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([meanX, x]) if v.size == 1)

    return np.squeeze(F, axis=squeeze_axes)


def inv(F, meanX, sdvnX):
    """
    laplace.inv

    Computes the inverse CDF (quantile function) of the Laplace distribution.

    INPUTS
        F     : float or array_like, shape (N,)   probability values in [0, 1]
        meanX : float or array_like, shape (n,)   mean(s) / location parameter(s)
        sdvnX : float or array_like, shape (n,)   standard deviation(s) / scale, > 0

    OUTPUTS
        x : ndarray, shape (n, N)   quantile values; singleton axes are squeezed

    Notes
    -----
    x = mu + (sigma / sqrt(2)) * log(2F)       for F <= 0.5
    x = mu - (sigma / sqrt(2)) * log(2(1 - F)) for F >  0.5

    Reference
    ---------
    https://en.wikipedia.org/wiki/Laplace_distribution
    """
    meanX, sdvnX = _validate_(meanX, sdvnX)                  # (n, 1)
    if F.ndim <= 1:
        F = F.reshape(1, -1)   # (1, N) - shared F grid for all n variables
    F = np.clip(F, np.finfo(float).eps, 1.0 - np.finfo(float).eps)

    sr2 = np.sqrt(2.0)
    x_left  = meanX + sdvnX / sr2 * np.log(2.0 * F)         # (n, N) branch values
    x_right = meanX - sdvnX / sr2 * np.log(2.0 * (1.0 - F))

    x = np.where(F <= 0.5, x_left, x_right)                  # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [meanX, F].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([meanX, F]) if v.size == 1)

    return np.squeeze(x, axis=squeeze_axes)


def rnd(meanX, sdvnX, N, R=None, seed=None):
    """
    laplace.rnd

    Generate random samples from the Laplace distribution.

    INPUTS
        meanX : float or array_like, shape (n,)   mean(s) / location parameter(s)
        sdvnX : float or array_like, shape (n,)   standard deviation(s) / scale, > 0
        N     : int                                number of samples per variable
        R     : ndarray, shape (n, n), optional    correlation matrix;
                if None, generates uncorrelated samples
        seed  : int or None                        random seed for reproducibility

    OUTPUTS
        X : ndarray, shape (n, N)   Laplace random samples;
            each row is one random variable, each column one sample;
            singleton axes are squeezed

    Notes
    -----
    Uses the inverse transform method with correlated uniform variates.
    The inverse transform is inlined rather than calling inv(), because U is
    already (n, N) from correlated_rvs, while inv() expects F as (1, N).

    Reference
    ---------
    https://en.wikipedia.org/wiki/Laplace_distribution
    """
    if N is None or N < 1:
        raise ValueError("laplace.rnd: N must be greater than zero")

    meanX, sdvnX = _validate_(meanX, sdvnX)                  # (n, 1)
    n = meanX.size

    # Generate n correlated uniform [0, 1] variates, shape (n, N)
    _, _, U = correlated_rvs(R, n, N, seed)

    # Inline the inverse transform rather than calling inv(), because U is
    # already (n, N) from correlated_rvs, while inv() expects F as (1, N).
    sr2 = np.sqrt(2.0)
    X_left  = meanX + sdvnX / sr2 * np.log(2.0 * U)         # (n, N) branch values
    X_right = meanX - sdvnX / sr2 * np.log(2.0 * (1.0 - U))

    X = np.where(U <= 0.5, X_left, X_right)                  # (n, N)

    # Squeeze singleton axes from the (n, N) output
    squeeze_axes = tuple(np.where(np.asarray([n, N]) == 1)[0])

    return np.squeeze(X, axis=squeeze_axes)
