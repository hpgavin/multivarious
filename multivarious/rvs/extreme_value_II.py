#! /usr/bin/env -S python3 -i
## extreme_value_II distribution (Fréchet)
# github.com/hpgavin/multivarious ... rvs/extreme_value_II

import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs


def _validate_(m, s, k):
    """
    Validate and preprocess Fréchet distribution parameters.

    Converts m, s, k to (n, 1) column arrays for broadcasting against
    a (1, N) row array of evaluation points, producing (n, N) output.

    INPUTS
        m : float or array_like   location parameter(s),  must be > 0
        s : float or array_like   scale parameter(s),     must be > 0
        k : float or array_like   shape parameter(s),     must be > 0

    OUTPUTS
        m : ndarray, shape (n, 1)
        s : ndarray, shape (n, 1)
        k : ndarray, shape (n, 1)
    """
    m = np.asarray(m, dtype=float).reshape(-1, 1)  # (n, 1)
    s = np.asarray(s, dtype=float).reshape(-1, 1)  # (n, 1)
    k = np.asarray(k, dtype=float).reshape(-1, 1)  # (n, 1)

    if not (m.shape == s.shape == k.shape):
        raise ValueError(f"extreme_value_II: m, s, k must have the same length. "
                         f"Got m:{m.size}, s:{s.size}, k:{k.size}")
    if np.any(m <= 0):
        raise ValueError("extreme_value_II: m must be > 0")
    if np.any(s <= 0):
        raise ValueError("extreme_value_II: s must be > 0")
    if np.any(k <= 0):
        raise ValueError("extreme_value_II: k must be > 0")

    return m, s, k


def pdf(x, m, s, k):
    """
    extreme_value_II.pdf

    Computes the PDF of the Extreme Value Type II (Fréchet) distribution.

    INPUTS
        x : float or array_like, shape (N,)   evaluation points
        m : float or array_like, shape (n,)   location parameter(s), > 0
        s : float or array_like, shape (n,)   scale parameter(s),    > 0
        k : float or array_like, shape (n,)   shape parameter(s),    > 0

    OUTPUTS
        f : ndarray, shape (n, N)   PDF values; singleton axes are squeezed

    Notes
    -----
    f(x) = (k/s) * z^(-1-k) * exp(-z^(-k))   for x > m,   0 otherwise
    where z = (x - m) / s

    Reference
    ---------
    https://en.wikipedia.org/wiki/Fr%C3%A9chet_distribution
    """
    m, s, k = _validate_(m, s, k)                            # (n, 1)
    x = np.asarray(x, dtype=float).reshape( 1, -1)          # (1, N)

    inside = x > m                                            # (n, N) boolean mask
    z  = np.where(inside, (x - m) / s, 1.0)                 # (n, N) clip to avoid /0

    f_in = (k / s) * z**(-1.0 - k) * np.exp(-z**(-k))      # (n, N) formula values
    f = np.where(inside, f_in, 0.0)                          # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [m, x].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([m, x]) if v.size == 1)

    return np.squeeze(f, axis=squeeze_axes)


def cdf(x, params):
    """
    extreme_value_II.cdf

    Computes the CDF of the Extreme Value Type II (Fréchet) distribution.

    INPUTS
        x      : float or array_like, shape (N,)   evaluation points
        params : tuple (m, s, k)
            m : float or array_like, shape (n,)   location parameter(s), > 0
            s : float or array_like, shape (n,)   scale parameter(s),    > 0
            k : float or array_like, shape (n,)   shape parameter(s),    > 0

    OUTPUTS
        F : ndarray, shape (n, N)   CDF values; singleton axes are squeezed

    Notes
    -----
    F(x) = exp(-z^(-k))   for x > m,   0 otherwise
    where z = (x - m) / s

    Reference
    ---------
    https://en.wikipedia.org/wiki/Fr%C3%A9chet_distribution
    """
    m, s, k = params
    m, s, k = _validate_(m, s, k)                            # (n, 1)
    x = np.asarray(x, dtype=float).reshape( 1, -1)          # (1, N)

    inside = x > m                                            # (n, N) boolean mask
    z  = np.where(inside, (x - m) / s, 1.0)                 # clip to avoid /0

    F = np.where(inside, np.exp(-z**(-k)), 0.0)              # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [m, x].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([m, x]) if v.size == 1)

    return np.squeeze(F, axis=squeeze_axes)


def inv(F, m, s, k):
    """
    extreme_value_II.inv

    Computes the inverse CDF (quantile function) of the Extreme Value Type II
    (Fréchet) distribution.

    INPUTS
        F : float or array_like, shape (N,)   probability values in (0, 1)
        m : float or array_like, shape (n,)   location parameter(s), > 0
        s : float or array_like, shape (n,)   scale parameter(s),    > 0
        k : float or array_like, shape (n,)   shape parameter(s),    > 0

    OUTPUTS
        x : ndarray, shape (n, N)   quantile values; singleton axes are squeezed

    Notes
    -----
    x = m + s * (-log(F))^(-1/k)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Fr%C3%A9chet_distribution
    """
    m, s, k = _validate_(m, s, k)                            # (n, 1)
    F = np.asarray(F, dtype=float).reshape( 1, -1)          # (1, N)
    F = np.clip(F, np.finfo(float).eps, 1.0 - np.finfo(float).eps)

    x = m + s * (-np.log(F))**(-1.0 / k)                    # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [m, F].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([m, F]) if v.size == 1)

    return np.squeeze(x, axis=squeeze_axes)


def rnd(m, s, k, N, R=None, seed=None):
    """
    extreme_value_II.rnd

    Generate random samples from the Extreme Value Type II (Fréchet) distribution.

    INPUTS
        m    : float or array_like, shape (n,)   location parameter(s), > 0
        s    : float or array_like, shape (n,)   scale parameter(s),    > 0
        k    : float or array_like, shape (n,)   shape parameter(s),    > 0
        N    : int                                number of samples per variable
        R    : ndarray, shape (n, n), optional    correlation matrix;
               if None, generates uncorrelated samples
        seed : int or None                        random seed for reproducibility

    OUTPUTS
        X : ndarray, shape (n, N)   Fréchet random samples;
            each row is one random variable, each column one sample;
            singleton axes are squeezed

    Notes
    -----
    Uses the inverse transform method: X = m + s * (-log(U))^(-1/k)
    Inlined rather than calling inv() because U is already (n, N)
    from correlated_rvs, while inv() expects F as (1, N).

    Reference
    ---------
    https://en.wikipedia.org/wiki/Fr%C3%A9chet_distribution
    """
    if N is None or N < 1:
        raise ValueError("extreme_value_II.rnd: N must be greater than zero")

    m, s, k = _validate_(m, s, k)                            # (n, 1)
    n = m.size

    # Generate n correlated uniform [0, 1] variates, shape (n, N)
    _, _, U = correlated_rvs(R, n, N, seed)

    # Inline the inverse transform rather than calling inv(), because U is
    # already (n, N) from correlated_rvs, while inv() expects F as (1, N).
    X = m + s * (-np.log(U))**(-1.0 / k)                    # (n, N)

    # Squeeze singleton axes from the (n, N) output
    squeeze_axes = tuple(np.where(np.asarray([n, N]) == 1)[0])

    return np.squeeze(X, axis=squeeze_axes)
