#! /usr/bin/env -S python3 -i
## chi-squared distribution
# github.com/hpgavin/multivarious ... rvs/chi2

import numpy as np
from scipy.stats import norm as scipy_normal

from multivarious.utl.correlated_rvs import correlated_rvs


def _validate_(k):
    """
    Validate and preprocess chi-squared distribution parameters.

    Converts k to an (n, 1) column array for broadcasting against
    a (1, N) row array of evaluation points, producing (n, N) output.
    Derives the Wilson-Hilferty transformation parameters m and s from k.

    INPUTS
        k : float or array_like   degrees of freedom, must be > 0

    OUTPUTS
        k : ndarray, shape (n, 1)
        m : ndarray, shape (n, 1)   WH mean:   1 - 2/(9k)
        s : ndarray, shape (n, 1)   WH std dev: sqrt(2/(9k))

    Reference
    ---------
    https://en.wikipedia.org/wiki/Chi-squared_distribution#Asymptotic_properties
    """
    k = np.asarray(k, dtype=float).reshape(-1, 1)  # (n, 1)

    if np.any(k <= 0):
        raise ValueError("chi2: all k values must be > 0")

    # Wilson-Hilferty approximation parameters
    m = 1.0 - 2.0 / (9.0 * k)          # (n, 1) mean of cube-root-transformed variable
    s = np.sqrt(2.0 / (9.0 * k))       # (n, 1) std dev of cube-root-transformed variable

    return k, m, s


def pdf(x, k):
    """
    chi2.pdf

    Approximates the PDF of the chi-squared distribution using the
    Wilson-Hilferty (WH) transformation.

    INPUTS
        x : float or array_like, shape (N,)   evaluation points (x > 0)
        k : float or array_like, shape (n,)   degrees of freedom, must be > 0

    OUTPUTS
        f : ndarray, shape (n, N)   approximate PDF values; singleton axes are squeezed

    Notes
    -----
    The WH transformation maps (X/k)^(1/3) ~ N(m, s^2), so:
        z = ((x/k)^(1/3) - m) / s
        f(x) ≈ phi(z, 0, 1) * sqrt(2) * s
    where phi is the standard normal PDF.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Chi-squared_distribution#Asymptotic_properties
    """
    k, m, s = _validate_(                               # (n, 1)
    x = np.asarray(x, dtype=float).reshape( 1, -1)      # (1, N)
    x = np.where(x <= 0, np.finfo(float).eps, x)        # guard against x <= 0

    z = ((x / k) ** (1.0 / 3.0) - m) / s                # (n, N) standardized

    # scipy_normal.pdf broadcasts (n,N) z with scalar loc=0, scale=1
    f = scipy_normal.pdf(z, 0, 1) * np.sqrt(2.0) * s    # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [k, x].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([k, x]) if v.size == 1)

    return np.squeeze(f, axis=squeeze_axes)


def cdf(x, k):
    """
    chi2.cdf

    Approximates the CDF of the chi-squared distribution using the
    Wilson-Hilferty (WH) transformation.

    INPUTS
        x : float or array_like, shape (N,)   evaluation points (x > 0)
        k : float or array_like, shape (n,)   degrees of freedom, must be > 0

    OUTPUTS
        F : ndarray, shape (n, N)   approximate CDF values; singleton axes are squeezed

    Notes
    -----
    The WH transformation maps (X/k)^(1/3) ~ N(m, s^2), so:
        z = ((x/k)^(1/3) - m) / s
        F(x) ≈ Phi(z)
    where Phi is the standard normal CDF.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Chi-squared_distribution#Asymptotic_properties
    """
    k, m, s = _validate_(k)                            # (n, 1)
    x = np.asarray(x, dtype=float).reshape( 1, -1)     # (1, N)
    x = np.where(x <= 0, np.finfo(float).eps, x)       # guard against x <= 0

    z = ((x / k) ** (1.0 / 3.0) - m) / s               # (n, N) standardized

    # scipy_normal.cdf broadcasts (n,N) z with scalar loc=0, scale=1
    F = scipy_normal.cdf(z, 0, 1)                      # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [k, x].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([k, x]) if v.size == 1)

    return np.squeeze(F, axis=squeeze_axes)


def inv(F, k):
    """
    chi2.inv

    Approximates the inverse CDF (quantile function) of the chi-squared
    distribution using the Wilson-Hilferty (WH) transformation.

    INPUTS
        F : float or array_like, shape (N,)   probability values in [0, 1]
        k : float or array_like, shape (n,)   degrees of freedom, must be > 0

    OUTPUTS
        x : ndarray, shape (n, N)   quantile values; singleton axes are squeezed

    Notes
    -----
    Inverts the WH transformation:
        z = scipy_normal.ppf(F, m, s)   →   x = k * z^3

    scipy_normal.ppf broadcasts (n,1) m, s against (1,N) F → (n,N),
    so no loop over n is needed.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Chi-squared_distribution#Asymptotic_properties
    """
    k, m, s = _validate_(k)                           # (n, 1)
    F = np.asarray(F, dtype=float).reshape( 1, -1)    # (1, N)
    F = np.clip(F, np.finfo(float).eps, 1.0 - np.finfo(float).eps)

    # scipy_normal.ppf broadcasts (n,1) m, s against (1,N) F → (n,N)
    z = scipy_normal.ppf(F, m, s)                     # (n, N)
    x = k * z**3                                      # (n, N)
    x = np.where(x <= 0, 1e-12, x)                    # guard against x <= 0

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [k, F].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([k, F]) if v.size == 1)

    return np.squeeze(x, axis=squeeze_axes)


def rnd(k, N, R=None, seed=None):
    """
    chi2.rnd

    Generate random samples from the chi-squared distribution via the
    Wilson-Hilferty (WH) transformation.

    INPUTS
        k    : float or array_like, shape (n,)   degrees of freedom, must be > 0
        N    : int                                number of samples per variable
        R    : ndarray, shape (n, n), optional    correlation matrix;
               if None, generates uncorrelated samples
        seed : int or None                        random seed for reproducibility

    OUTPUTS
        X : ndarray, shape (n, N)   chi-squared random samples;
            each row is one random variable, each column one sample;
            singleton axes are squeezed

    Notes
    -----
    Uses the inverse transform method with correlated uniform variates.
    The inverse transform is inlined rather than calling inv(), because U is
    already (n, N) from correlated_rvs, while inv() expects F as (1, N).

    scipy_normal.ppf broadcasts (n,1) m, s against (n,N) U → (n,N),
    so no loop over n is needed.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Chi-squared_distribution#Asymptotic_properties
    """
    if N is None or N < 1:
        raise ValueError("chi2.rnd: N must be greater than zero")

    k, m, s = _validate_(k)                            # (n, 1)
    n = k.size

    # Generate n correlated uniform [0, 1] variates, shape (n, N)
    _, _, U = correlated_rvs(R, n, N, seed)

    # Inline the inverse transform rather than calling inv(), because U is
    # already (n, N) from correlated_rvs, while inv() expects F as (1, N).
    # scipy_normal.ppf broadcasts (n,1) m, s against (n,N) U → (n,N)
    z = scipy_normal.ppf(U, m, s)                      # (n, N)
    X = k * z**3                                       # (n, N)
    X = np.where(X <= 0, 1e-12, X)                     # guard against X <= 0

    # Squeeze singleton axes from the (n, N) output
    squeeze_axes = tuple(np.where(np.asarray([n, N]) == 1)[0])

    return np.squeeze(X, axis=squeeze_axes)
