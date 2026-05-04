#! /usr/bin/env -S python3 -i
## generalized extreme value distribution
# github.com/hpgavin/multivarious ... rvs/gev

import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs


def _validate_(m, s, k):
    """
    Validate and preprocess GEV distribution parameters.

    Converts m, s, k to (n, 1) column arrays for broadcasting against
    a (1, N) row array of evaluation points, producing (n, N) output.

    INPUTS
        m : float or array_like   location parameter(s)
        s : float or array_like   scale parameter(s),  must be > 0
        k : float or array_like   shape parameter(s)   (k=0: Gumbel,
                                                         k>0: Fréchet,
                                                         k<0: Weibull)

    OUTPUTS
        m : ndarray, shape (n, 1)
        s : ndarray, shape (n, 1)
        k : ndarray, shape (n, 1)
    """
    m = np.asarray(m, dtype=float).reshape(-1, 1)  # (n, 1)
    s = np.asarray(s, dtype=float).reshape(-1, 1)  # (n, 1)
    k = np.asarray(k, dtype=float).reshape(-1, 1)  # (n, 1)

    if not (m.shape == s.shape == k.shape):
        raise ValueError(f"gev: m, s, k must have the same length. "
                         f"Got m:{m.size}, s:{s.size}, k:{k.size}")
    if np.any(s <= 0):
        raise ValueError("gev: s must be > 0")

    return m, s, k


def pdf(x, m, s, k):
    """
    gev.pdf

    Computes the PDF of the Generalized Extreme Value (GEV) distribution.

    INPUTS
        x : float or array_like, shape (N,)   evaluation points
        m : float or array_like, shape (n,)   location parameter(s)
        s : float or array_like, shape (n,)   scale parameter(s),  > 0
        k : float or array_like, shape (n,)   shape parameter(s)

    OUTPUTS
        f : ndarray, shape (n, N)   PDF values; singleton axes are squeezed

    Notes
    -----
    t(x) = 1 + k*(x-m)/s
    f(x) = (1/s) * t^(-1-1/k) * exp(-t^(-1/k))   where t > 0
    f(x) = eps where t <= 0 (outside support)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution
    """
    m, s, k = _validate_(m, s, k)                             # (n, 1)
    x = np.asarray(x, dtype=float).reshape( 1, -1)           # (1, N)

    z    = (x - m) / s                                        # (n, N) standardized
    kzp1 = k * z + 1.0                                        # (n, N) t(x)

    f_in = (1.0 / s) * kzp1**(-1.0 - 1.0 / k) \
           * np.exp(-kzp1**(-1.0 / k))                        # (n, N) formula values
    f = np.where(kzp1 <= 0, np.finfo(float).eps, np.real(f_in))  # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [m, x].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([m, x]) if v.size == 1)

    return np.squeeze(f, axis=squeeze_axes)


def cdf(x, params):
    """
    gev.cdf

    Computes the CDF of the Generalized Extreme Value (GEV) distribution.

    INPUTS
        x      : float or array_like, shape (N,)   evaluation points
        params : tuple (m, s, k)
            m : float or array_like, shape (n,)   location parameter(s)
            s : float or array_like, shape (n,)   scale parameter(s),  > 0
            k : float or array_like, shape (n,)   shape parameter(s)

    OUTPUTS
        F : ndarray, shape (n, N)   CDF values; singleton axes are squeezed

    Notes
    -----
    t(x) = 1 + k*(x-m)/s
    F(x) = exp(-t^(-1/k))   where t > 0
    F(x) = eps               where t <= 0

    Reference
    ---------
    https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution
    """
    m, s, k = params
    m, s, k = _validate_(m, s, k)                             # (n, 1)
    x = np.asarray(x, dtype=float).reshape( 1, -1)           # (1, N)

    z    = (x - m) / s                                        # (n, N) standardized
    kzp1 = k * z + 1.0                                        # (n, N) t(x)

    F_in = np.exp(-kzp1**(-1.0 / k))                         # (n, N) formula values
    F = np.where(kzp1 <= 0, np.finfo(float).eps, np.real(F_in))  # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [m, x].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([m, x]) if v.size == 1)

    return np.squeeze(F, axis=squeeze_axes)


def inv(F, m, s, k):
    """
    gev.inv

    Computes the inverse CDF (quantile function) of the GEV distribution.

    INPUTS
        F : float or array_like, shape (N,)   probability values in (0, 1)
        m : float or array_like, shape (n,)   location parameter(s)
        s : float or array_like, shape (n,)   scale parameter(s),  > 0
        k : float or array_like, shape (n,)   shape parameter(s)

    OUTPUTS
        x : ndarray, shape (n, N)   quantile values; singleton axes are squeezed

    Notes
    -----
    x = m + (s/k) * ((-log(F))^(-k) - 1)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution
    """
    m, s, k = _validate_(m, s, k)                             # (n, 1)
    F = np.asarray(F, dtype=float).reshape( 1, -1)           # (1, N)
    F = np.clip(F, np.finfo(float).eps, 1.0 - np.finfo(float).eps)

    x = m + (s / k) * ((-np.log(F))**(-k) - 1.0)            # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [m, F].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([m, F]) if v.size == 1)

    return np.squeeze(x, axis=squeeze_axes)


def rnd(m, s, k, N, R=None, seed=None):
    """
    gev.rnd

    Generate random samples from the Generalized Extreme Value (GEV) distribution.

    INPUTS
        m    : float or array_like, shape (n,)   location parameter(s)
        s    : float or array_like, shape (n,)   scale parameter(s),  > 0
        k    : float or array_like, shape (n,)   shape parameter(s)
        N    : int                                number of samples per variable
        R    : ndarray, shape (n, n), optional    correlation matrix;
               if None, generates uncorrelated samples
        seed : int or None                        random seed for reproducibility

    OUTPUTS
        X : ndarray, shape (n, N)   GEV random samples;
            each row is one random variable, each column one sample;
            singleton axes are squeezed

    Notes
    -----
    Uses the inverse transform method: X = m + (s/k) * ((-log(U))^(-k) - 1)
    Inlined rather than calling inv() because U is already (n, N)
    from correlated_rvs, while inv() expects F as (1, N).

    Reference
    ---------
    https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution
    """
    if N is None or N < 1:
        raise ValueError("gev.rnd: N must be greater than zero")

    m, s, k = _validate_(m, s, k)                             # (n, 1)
    n = m.size

    # Generate n correlated uniform [0, 1] variates, shape (n, N)
    _, _, U = correlated_rvs(R, n, N, seed)

    # Inline the inverse transform rather than calling inv(), because U is
    # already (n, N) from correlated_rvs, while inv() expects F as (1, N).
    X = m + (s / k) * ((-np.log(U))**(-k) - 1.0)            # (n, N)

    # Squeeze singleton axes from the (n, N) output
    squeeze_axes = tuple(np.where(np.asarray([n, N]) == 1)[0])

    return np.squeeze(X, axis=squeeze_axes)
