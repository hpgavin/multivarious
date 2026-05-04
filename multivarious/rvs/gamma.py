#! /usr/bin/env -S python3 -i
## gamma distribution
# github.com/hpgavin/multivarious ... rvs/gamma

import numpy as np
from scipy.special import gammaln, gammainc, gammaincinv

from multivarious.utl.correlated_rvs import correlated_rvs


def _validate_(meanX, covnX):
    """
    Validate and preprocess gamma distribution parameters.

    Converts meanX and covnX to (n, 1) column arrays for broadcasting against
    a (1, N) row array of evaluation points, producing (n, N) output.
    Derives the shape parameter k and scale parameter theta from meanX and covnX.

    INPUTS
        meanX : float or array_like   mean(s) of the distribution,            must be > 0
        covnX : float or array_like   coefficient(s) of variation (std/mean), must be > 0

    OUTPUTS
        meanX : ndarray, shape (n, 1)
        covnX : ndarray, shape (n, 1)
        k     : ndarray, shape (n, 1)   shape parameter alpha = 1 / covnX^2
        theta : ndarray, shape (n, 1)   scale parameter beta  = covnX^2 * meanX
    """
    meanX = np.asarray(meanX, dtype=float).reshape(-1, 1)  # (n, 1)
    covnX = np.asarray(covnX, dtype=float).reshape(-1, 1)  # (n, 1)

    if meanX.shape != covnX.shape:
        raise ValueError(f"gamma: meanX and covnX must have the same length. "
                         f"Got meanX:{meanX.size}, covnX:{covnX.size}")
    if np.any(meanX <= 0):
        raise ValueError("gamma: meanX must be > 0")
    if np.any(covnX <= 0):
        raise ValueError("gamma: covnX must be > 0")

    k     = 1.0 / covnX**2       # (n, 1) shape parameter alpha
    theta = covnX**2 * meanX     # (n, 1) scale parameter beta

    return meanX, covnX, k, theta


def pdf(x, meanX, covnX):
    """
    gamma.pdf

    Computes the PDF of the gamma distribution parameterized by its mean
    and coefficient of variation.

    INPUTS
        x     : float or array_like, shape (N,)   evaluation points (x > 0)
        meanX : float or array_like, shape (n,)   mean(s),                > 0
        covnX : float or array_like, shape (n,)   coefficient(s) of variation, > 0

    OUTPUTS
        f : ndarray, shape (n, N)   PDF values; singleton axes are squeezed

    Notes
    -----
    f(x) = x^(k-1) * exp(-x/theta) / (Gamma(k) * theta^k)
    where k = 1/covnX^2 and theta = covnX^2 * meanX.

    Computed in log space for numerical stability, then exponentiated.
    Values x <= 0 are replaced by eps before evaluation.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Gamma_distribution
    """
    meanX, covnX, k, theta = _validate_(meanX, covnX)       # (n, 1)
    x = np.asarray(x, dtype=float).reshape( 1, -1)         # (1, N)
    x = np.where(x <= 0, np.finfo(float).eps, x)           # guard against x <= 0

    # Log-space computation for numerical stability, then exponentiate
    log_f = (k - 1) * np.log(x) - x / theta \
            - k * np.log(theta) - gammaln(k)                # (n, N)
    f = np.exp(log_f)                                        # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [meanX, x].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([meanX, x]) if v.size == 1)

    return np.squeeze(f, axis=squeeze_axes)


def cdf(x, params):
    """
    gamma.cdf

    Computes the CDF of the gamma distribution parameterized by its mean
    and coefficient of variation.

    INPUTS
        x      : float or array_like, shape (N,)   evaluation points (x > 0)
        params : tuple (meanX, covnX)
            meanX : float or array_like, shape (n,)   mean(s),                > 0
            covnX : float or array_like, shape (n,)   coefficient(s) of variation, > 0

    OUTPUTS
        F : ndarray, shape (n, N)   CDF values; singleton axes are squeezed

    Notes
    -----
    F(x) = gammainc(k, x/theta)
    where gammainc is the regularized lower incomplete gamma function,
    k = 1/covnX^2, and theta = covnX^2 * meanX.

    gammainc supports NumPy broadcasting, so no loop over n is needed.
    Values x <= 0 return F = 0.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Gamma_distribution
    """
    meanX, covnX = params
    meanX, covnX, k, theta = _validate_(meanX, covnX)       # (n, 1)
    x = np.asarray(x, dtype=float).reshape( 1, -1)         # (1, N)

    # gammainc broadcasts (n,1) k against (n,N) x/theta → (n,N)
    xp = np.where(x <= 0, np.finfo(float).eps, x)          # guard against x <= 0
    F  = np.where(x <= 0, 0.0, gammainc(k, xp / theta))    # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [meanX, x].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([meanX, x]) if v.size == 1)

    return np.squeeze(F, axis=squeeze_axes)


def inv(F, meanX, covnX):
    """
    gamma.inv

    Computes the inverse CDF (quantile function) of the gamma distribution.

    INPUTS
        F     : float or array_like, shape (N,)   probability values in [0, 1]
        meanX : float or array_like, shape (n,)   mean(s),                > 0
        covnX : float or array_like, shape (n,)   coefficient(s) of variation, > 0

    OUTFUTS
        x : ndarray, shape (n, N)   quantile values; singleton axes are squeezed

    Notes
    -----
    Solves gammainc(k, x/theta) = F exactly via:
        x = theta * gammaincinv(k, F)
    where gammaincinv is the inverse of the regularized lower incomplete gamma
    function (scipy.special.gammaincinv).

    gammaincinv supports NumPy broadcasting, so no iterative solver is needed.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Gamma_distribution
    """
    meanX, covnX, k, theta = _validate_(meanX, covnX)       # (n, 1)
    if F.ndim <= 1:
        F = F.reshape(1, -1)   # (1, N) - shared F grid for all n variables
    F = np.clip(F, np.finfo(float).eps, 1.0 - np.finfo(float).eps)

    # gammaincinv broadcasts (n,1) k against (1,N) F → (n,N)
    x = theta * gammaincinv(k, F)                           # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [meanX, F].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([meanX, F]) if v.size == 1)

    return np.squeeze(x, axis=squeeze_axes)


def rnd(meanX, covnX, N, R=None, seed=None):
    """
    gamma.rnd

    Generate random samples from the gamma distribution.

    INPUTS
        meanX : float or array_like, shape (n,)   mean(s),                > 0
        covnX : float or array_like, shape (n,)   coefficient(s) of variation, > 0
        N     : int                                number of samples per variable
        R     : ndarray, shape (n, n), optional    correlation matrix;
                if None, generates uncorrelated samples
        seed  : int or None                        random seed for reproducibility

    OUTPUTS
        X : ndarray, shape (n, N)   gamma random samples;
            each row is one random variable, each column one sample;
            singleton axes are squeezed

    Notes
    -----
    Uses the inverse transform method with correlated uniform variates.
    The inverse transform is inlined rather than calling inv(), because U is
    already (n, N) from correlated_rvs, while inv() expects P as (1, N).

    gammaincinv broadcasts (n,1) k against (n,N) U → (n,N), so no loop
    over n is needed.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Gamma_distribution
    """
    if N is None or N < 1:
        raise ValueError("gamma.rnd: N must be greater than zero")

    meanX, covnX, k, theta = _validate_(meanX, covnX)       # (n, 1)
    n = meanX.size

    # Generate n correlated uniform [0, 1] variates, shape (n, N)
    _, _, U = correlated_rvs(R, n, N, seed)

    # Inline the inverse transform rather than calling inv(), because U is
    # already (n, N) from correlated_rvs, while inv() expects P as (1, N).
    # gammaincinv broadcasts (n,1) k against (n,N) U → (n,N)
    X = theta * gammaincinv(k, U)                           # (n, N)

    # Squeeze singleton axes from the (n, N) output
    squeeze_axes = tuple(np.where(np.asarray([n, N]) == 1)[0])

    return np.squeeze(X, axis=squeeze_axes)
