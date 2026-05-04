#! /usr/bin/env -S python3 -i
## lognormal distribution
# github.com/hpgavin/multivarious ... rvs/lognormal

import numpy as np
from scipy.special import erf    as scipy_erf
from scipy.special import erfinv as scipy_erfinv

from multivarious.utl.correlated_rvs import correlated_rvs


def _validate_(mednX, covnX):
    """
    Validate and preprocess lognormal distribution parameters.

    Converts mednX and covnX to (n, 1) column arrays for broadcasting
    against a (1, N) row array of evaluation points, producing (n, N) output.

    INPUTS
        mednX : float or array_like   median(s) of the distribution, must be > 0
        covnX : float or array_like   coefficient(s) of variation, must be > 0

    OUTPUTS
        mednX : ndarray, shape (n, 1)
        covnX : ndarray, shape (n, 1)
    """
    mednX = np.asarray(mednX, dtype=float).reshape(-1, 1)  # (n, 1)
    covnX = np.asarray(covnX, dtype=float).reshape(-1, 1)  # (n, 1)

    if mednX.shape != covnX.shape:
        raise ValueError(f"lognormal: mednX and covnX must have the same length. "
                         f"Got mednX:{mednX.size}, covnX:{covnX.size}")
    if np.any(mednX <= 0):
        raise ValueError("lognormal: mednX must be > 0")
    if np.any(covnX <= 0):
        raise ValueError("lognormal: covnX must be > 0")

    return mednX, covnX


def pdf(x, mednX, covnX):
    """
    lognormal.pdf

    Computes the PDF of the lognormal distribution.

    INPUTS
        x     : float or array_like, shape (N,)   evaluation points (x > 0)
        mednX : float or array_like, shape (n,)   median(s), must be > 0
        covnX : float or array_like, shape (n,)   coefficient(s) of variation, > 0

    OUTPUTS
        f : ndarray, shape (n, N)   PDF values; singleton axes are squeezed

    Notes
    -----
    If X ~ Lognormal(mednX, covnX), then log(X) ~ Normal with
        mean     = log(mednX)
        variance = V = log(1 + covnX^2)

    f(x) = (1 / (x * sqrt(2 pi V))) * exp(-(log(x/mednX))^2 / (2V))

    Values x <= 0 are replaced by 0.01 to avoid log(0).

    Reference
    ---------
    https://en.wikipedia.org/wiki/Log-normal_distribution
    """
    mednX, covnX = _validate_(mednX, covnX)          # (n, 1)
    x = np.asarray(x, dtype=float).reshape( 1, -1)   # (1, N)
    x = np.where(x <= 0, 0.01, x)                    # guard against log(0)

    VlnX = np.log(1.0 + covnX**2)                    # (n, 1) variance of log(X)

    f = np.exp(-0.5 * (np.log(x / mednX))**2 / VlnX) \
        / (x * np.sqrt(2.0 * np.pi * VlnX))                # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [mednX, x].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([mednX, x]) if v.size == 1)

    return np.squeeze(f, axis=squeeze_axes)


def cdf(x, params):
    """
    lognormal.cdf

    Computes the CDF of the lognormal distribution.

    INPUTS
        x      : float or array_like, shape (N,)   evaluation points (x > 0)
        params : tuple (mednX, covnX)
            mednX : float or array_like, shape (n,)   median(s), must be > 0
            covnX : float or array_like, shape (n,)   coefficient(s) of variation, > 0

    OUTPUTS
        F : ndarray, shape (n, N)   CDF values; singleton axes are squeezed

    Notes
    -----
    F(x) = (1 + erf((log(x) - log(mednX)) / sqrt(2V))) / 2
    where V = log(1 + covnX^2)

    Values x <= 0 are replaced by 0.01 to avoid log(0).

    Reference
    ---------
    https://en.wikipedia.org/wiki/Log-normal_distribution
    """
    mednX, covnX = params
    mednX, covnX = _validate_(mednX, covnX)          # (n, 1)
    x = np.asarray(x, dtype=float).reshape( 1, -1)   # (1, N)
    x = np.where(x <= 0, 0.01, x)                    # guard against log(0)

    VlnX = np.log(1.0 + covnX**2)                    # (n, 1) variance of log(X)

    F = 0.5 * (1.0 + scipy_erf(
            (np.log(x) - np.log(mednX)) / np.sqrt(2.0 * VlnX)))  # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [mednX, x].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([mednX, x]) if v.size == 1)

    return np.squeeze(F, axis=squeeze_axes)


def inv(F, mednX, covnX):
    """
    lognormal.inv

    Computes the inverse CDF (quantile function) of the lognormal distribution.

    INPUTS
        F     : float or array_like, shape (N,)   probability values in (0, 1)
        mednX : float or array_like, shape (n,)   median(s), must be > 0
        covnX : float or array_like, shape (n,)   coefficient(s) of variation, > 0

    OUTPUTS
        x : ndarray, shape (n, N)   quantile values; singleton axes are squeezed

    Notes
    -----
    x = exp(log(mednX) + sqrt(2V) * erfinv(2F - 1))
    where V = log(1 + covnX^2)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Log-normal_distribution
    """
    mednX, covnX = _validate_(mednX, covnX)          # (n, 1)
    if F.ndim <= 1:
        F = F.reshape(1, -1)   # (1, N) - shared F grid for all n variables
    F = np.clip(F, np.finfo(float).eps, 1.0 - np.finfo(float).eps)

    VlnX = np.log(1.0 + covnX**2)                    # (n, 1) variance of log(X)

    x = np.exp(np.log(mednX) + np.sqrt(2.0 * VlnX) \
        * scipy_erfinv(2.0 * F - 1.0))               # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [mednX, F].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([mednX, F]) if v.size == 1)

    return np.squeeze(x, axis=squeeze_axes)


def rnd(mednX, covnX, N, R=None, seed=None):
    """
    lognormal.rnd

    Generate correlated random samples from the lognormal distribution.

    INPUTS
        mednX : float or array_like, shape (n,)   median(s), must be > 0
        covnX : float or array_like, shape (n,)   coefficient(s) of variation, > 0
        N     : int                                number of samples per variable
        R     : ndarray, shape (n, n), optional    correlation matrix;
                if None, generates uncorrelated samples
        seed  : int or None                        random seed for reproducibility

    OUTPUTS
        X : ndarray, shape (n, N)   lognormal random samples;
            each row is one random variable, each column one sample;
            singleton axes are squeezed

    Notes
    -----
    Uses the Gaussian copula method:
        1. Generate correlated uniform variates U via correlated_rvs
        2. Apply the lognormal inverse transform:
           X = exp(log(mednX) + sqrt(2V) * erfinv(2U - 1))
           where V = log(1 + covnX^2)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Log-normal_distribution
    """
    if N is None or N < 1:
        raise ValueError("lognormal.rnd: N must be greater than zero")

    mednX, covnX = _validate_(mednX, covnX)                 # (n, 1)
    n = mednX.size

    # Generate n correlated uniform [0, 1] variates, shape (n, N)
    _, _, U = correlated_rvs(R, n, N, seed)

    # Inline the inverse transform rather than calling inv(), because U is
    # already (n, N) from correlated_rvs, while inv() expects F as (1, N).
    VlnX = np.log(1.0 + covnX**2)                          # (n, 1) variance of log(X)
    X = np.exp(np.log(mednX) + np.sqrt(2.0 * VlnX) \
        * scipy_erfinv(2.0 * U - 1.0))                     # (n, N)

    # Squeeze singleton axes from the (n, N) output
    squeeze_axes = tuple(np.where(np.asarray([n, N]) == 1)[0])

    return np.squeeze(X, axis=squeeze_axes)
