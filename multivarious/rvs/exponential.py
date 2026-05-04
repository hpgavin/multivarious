#! /usr/bin/env -S python3 -i
## exponential distribution
# github.com/hpgavin/multivarious ... rvs/exponential

import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs


def _validate_(meanX):
    """
    Validate and preprocess exponential distribution parameters.

    Converts meanX to an (n, 1) column array for broadcasting against
    a (1, N) row array of evaluation points, producing (n, N) output.

    INPUTS
        meanX : float or array_like   mean(s) of the distribution, must be > 0

    OUTPUTS
        meanX : ndarray, shape (n, 1)
    """
    meanX = np.asarray(meanX, dtype=float).reshape(-1, 1)  # (n, 1)

    if np.any(meanX <= 0):
        raise ValueError("exponential: all meanX values must be positive")

    return meanX


def pdf(x, meanX):
    """
    exponential.pdf

    Computes the PDF of the exponential distribution.

    INPUTS
        x     : float or array_like, shape (N,)   evaluation points (x >= 0)
        meanX : float or array_like, shape (n,)   mean(s), must be > 0

    OUTPUTS
        f : ndarray, shape (n, N)   PDF values; singleton axes are squeezed

    Notes
    -----
    f(x) = (1 / meanX) * exp(-x / meanX)   for x >= 0

    Values x < 0 are replaced by 0.01 to avoid undefined results.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Exponential_distribution
    """
    meanX = _validate_(meanX)                              # (n, 1)
    x = np.asarray(x, dtype=float).reshape( 1, -1)         # (1, N)
    x = np.where(x < 0, 0.01, x)                           # guard against x < 0

    f = np.exp(-x / meanX) / meanX                         # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [meanX, x].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([meanX, x]) if v.size == 1)

    return np.squeeze(f, axis=squeeze_axes)


def cdf(x, meanX):
    """
    exponential.cdf

    Computes the CDF of the exponential distribution.

    INPUTS
        x     : float or array_like, shape (N,)   evaluation points (x >= 0)
        meanX : float or array_like, shape (n,)   mean(s), must be > 0

    OUTPUTS
        F : ndarray, shape (n, N)   CDF values; singleton axes are squeezed

    Notes
    -----
    F(x) = 1 - exp(-x / meanX)   for x >= 0

    Values x < 0 are replaced by 0.01 to avoid undefined results.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Exponential_distribution
    """
    meanX = _validate_(meanX)                              # (n, 1)
    x = np.asarray(x, dtype=float).reshape( 1, -1)         # (1, N)
    x = np.where(x < 0, 0.01, x)                           # guard against x < 0

    F = 1.0 - np.exp(-x / meanX)                           # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [meanX, x].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([meanX, x]) if v.size == 1)

    return np.squeeze(F, axis=squeeze_axes)


def inv(P, meanX):
    """
    exponential.inv

    Computes the inverse CDF (quantile function) of the exponential distribution.

    INPUTS
        P     : float or array_like, shape (N,)   probability values in [0, 1]
        meanX : float or array_like, shape (n,)   mean(s), must be > 0

    OUTPUTS
        x : ndarray, shape (n, N)   quantile values; singleton axes are squeezed

    Notes
    -----
    x = -meanX * log(1 - P)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Exponential_distribution
    """
    meanX = _validate_(meanX)                             # (n, 1)
    P = np.asarray(P, dtype=float).reshape( 1, -1)        # (1, N)
    P = np.clip(P, 0.0, 1.0 - np.finfo(float).eps)        # guard against log(0)

    x = -meanX * np.log(1.0 - P)                          # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [meanX, P].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([meanX, P]) if v.size == 1)

    return np.squeeze(x, axis=squeeze_axes)


def rnd(meanX, N, R=None, seed=None):
    """
    exponential.rnd

    Generate random samples from the exponential distribution.

    INPUTS
        meanX : float or array_like, shape (n,)   mean(s), must be > 0
        N     : int                                number of samples per variable
        R     : ndarray, shape (n, n), optional    correlation matrix;
                if None, generates uncorrelated samples
        seed  : int or None                        random seed for reproducibility

    OUTPUTS
        X : ndarray, shape (n, N)   exponential random samples;
            each row is one random variable, each column one sample;
            singleton axes are squeezed

    Notes
    -----
    Uses the inverse transform method: X = -meanX * log(U), U ~ Uniform(0, 1).
    Since 1 - U ~ Uniform(0, 1) as well, log(U) and log(1-U) are equivalent
    in distribution, so the simpler form log(U) is used directly.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Exponential_distribution
    """
    if N is None or N < 1:
        raise ValueError("exponential.rnd: N must be greater than zero")

    meanX = _validate_(meanX)                             # (n, 1)
    n = meanX.size

    # Generate n correlated uniform [0, 1] variates, shape (n, N)
    _, _, U = correlated_rvs(R, n, N, seed)

    # Inverse transform: X = -meanX * log(U); broadcasts (n,1) over (n,N)
    X = -meanX * np.log(U)                                # (n, N)

    # Squeeze singleton axes from the (n, N) output
    squeeze_axes = tuple(np.where(np.asarray([n, N]) == 1)[0])

    return np.squeeze(X, axis=squeeze_axes)
