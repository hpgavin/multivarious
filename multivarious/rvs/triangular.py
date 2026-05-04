#! /usr/bin/env -S python3 -i
## triangular distribution
# github.com/hpgavin/multivarious ... rvs/triangular

import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs


def _validate_(a, b, c):
    """
    Validate and preprocess triangular distribution parameters.

    Converts a, b, c to (n, 1) column arrays for broadcasting against
    a (1, N) row array of evaluation points, producing (n, N) output.

    INPUTS
        a : float or array_like   lower bound(s)
        b : float or array_like   upper bound(s), must be >= a element-wise
        c : float or array_like   mode(s), must satisfy a <= c <= b

    OUTPUTS
        a : ndarray, shape (n, 1)
        b : ndarray, shape (n, 1)
        c : ndarray, shape (n, 1)
    """
    a = np.asarray(a, dtype=float).reshape(-1, 1)  # (n, 1)
    b = np.asarray(b, dtype=float).reshape(-1, 1)  # (n, 1)
    c = np.asarray(c, dtype=float).reshape(-1, 1)  # (n, 1)

    if not (a.shape == b.shape == c.shape):
        raise ValueError(f"triangular: a, b, c must have the same length. "
                         f"Got a:{a.size}, b:{b.size}, c:{c.size}")
    if np.any(a > b):
        raise ValueError("triangular: all a values must be <= b")
    if np.any(a > c):
        raise ValueError("triangular: all a values must be <= c")
    if np.any(c > b):
        raise ValueError("triangular: all c values must be <= b")

    return a, b, c


def pdf(x, a, b, c):
    """
    triangular.pdf

    Computes the PDF of the triangular distribution on [a, b] with mode c.

    INPUTS
        x : float or array_like, shape (N,)   evaluation points
        a : float or array_like, shape (n,)   lower bound(s)
        b : float or array_like, shape (n,)   upper bound(s), must be >= a
        c : float or array_like, shape (n,)   mode(s), must satisfy a <= c <= b

    OUTPUTS
        f : ndarray, shape (n, N)   PDF values; singleton axes are squeezed

    Notes
    -----
    np.where evaluates both branches everywhere before applying the mask,
    so c == a or c == b may raise a runtime divide-by-zero warning,
    though the masked-out values are not used in the result.

    Reference
    ---------
    http://en.wikipedia.org/wiki/Triangular_distribution
    """
    a, b, c = _validate_(a, b, c)                   # (n, 1)
    x = np.asarray(x, dtype=float).reshape( 1, -1)  # (1, N)

    left  = (a <= x) & (x <  c)                     # (n, N) boolean mask
    right = (c <= x) & (x <= b)                     # (n, N) boolean mask

    f_left  = 2 * (x - a) / ((b - a) * (c - a))     # (n, N) left-branch values
    f_right = 2 * (b - x) / ((b - a) * (b - c))     # (n, N) right-branch values

    f = np.where(left, f_left, np.where(right, f_right, 0.0))  # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [a, x].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([a, x]) if v.size == 1)

    return np.squeeze(f, axis=squeeze_axes)


def cdf(x, params):
    """
    triangular.cdf

    Computes the CDF of the triangular distribution on [a, b] with mode c.

    INPUTS
        x      : float or array_like, shape (N,)   evaluation points
        params : tuple (a, b, c)
            a : float or array_like, shape (n,)   lower bound(s)
            b : float or array_like, shape (n,)   upper bound(s), must be >= a
            c : float or array_like, shape (n,)   mode(s), a <= c <= b

    OUTPUTS
        F : ndarray, shape (n, N)   CDF values; singleton axes are squeezed

    Notes
    -----
    np.where evaluates both branches everywhere before applying the mask,
    so c == a or c == b may raise a runtime divide-by-zero warning,
    though the masked-out values are not used in the result.

    Reference
    ---------
    http://en.wikipedia.org/wiki/Triangular_distribution
    """
    a, b, c = params
    a, b, c = _validate_(a, b, c)                   # (n, 1)
    x = np.asarray(x, dtype=float).reshape( 1, -1)  # (1, N)

    mid1  = (a <  x) & (x <  c)                     # (n, N) boolean masks
    mid2  = (c <= x) & (x <  b)
    right = (x >= b)

    F_mid1 =     (x - a)**2 / ((b - a) * (c - a))   # (n, N) branch values
    F_mid2 = 1 - (b - x)**2 / ((b - a) * (b - c))

    F = np.where(right, 1.0,
        np.where(mid2,  F_mid2,
        np.where(mid1,  F_mid1, 0.0)))              # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [a, x].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([a, x]) if v.size == 1)

    return np.squeeze(F, axis=squeeze_axes)


def inv(F, a, b, c):
    """
    triangular.inv

    Computes the inverse CDF (quantile function) of the triangular distribution
    on [a, b] with mode c.

    INPUTS
        F : float or array_like, shape (N,)   probability values in [0, 1]
        a : float or array_like, shape (n,)   lower bound(s)
        b : float or array_like, shape (n,)   upper bound(s), must be >= a
        c : float or array_like, shape (n,)   mode(s), must satisfy a <= c <= b

    OUTPUTS
        x : ndarray, shape (n, N)   quantile values; singleton axes are squeezed

    Reference
    ---------
    http://en.wikipedia.org/wiki/Triangular_distribution
    """
    a, b, c = _validate_(a, b, c)                    # (n, 1)
    if F.ndim <= 1:
        F = F.reshape(1, -1)   # (1, N) - shared F grid for all n variables
    F = np.clip(F, np.finfo(float).eps, 1.0 - np.finfo(float).eps)

    Fc = (c - a) / (b - a)                           # (n, 1) CDF value at the mode

    x_below = a + np.sqrt( F      * (b - a) * (c - a))  # (n, N) branch values
    x_above = b - np.sqrt((1 - F) * (b - a) * (b - c))

    x = np.where(F < Fc, x_below, x_above)          # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [a, F].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([a, F]) if v.size == 1)

    return np.squeeze(x, axis=squeeze_axes)


def rnd(a, b, c, N, R=None, seed=None):
    """
    triangular.rnd

    Generate random samples from the triangular distribution on [a, b] with mode c.

    INPUTS
        a    : float or array_like, shape (n,)   lower bound(s)
        b    : float or array_like, shape (n,)   upper bound(s), must be >= a
        c    : float or array_like, shape (n,)   mode(s), must satisfy a <= c <= b
        N    : int                                number of samples per variable
        R    : ndarray, shape (n, n), optional    correlation matrix;
               if None, generates uncorrelated samples
        seed : int or None                        random seed for reproducibility

    OUTPUTS
        X : ndarray, shape (n, N)   triangular random samples;
            each row is one random variable, each column one sample;
            singleton axes are squeezed

    Reference
    ---------
    http://en.wikipedia.org/wiki/Triangular_distribution
    """
    if N is None or N < 1:
        raise ValueError("triangular.rnd: N must be greater than zero")

    a, b, c = _validate_(a, b, c)                 # (n, 1)
    n = a.size

    # Generate n correlated uniform [0, 1] variates, shape (n, N)
    _, _, U = correlated_rvs(R, n, N, seed)

    # Inline the inverse transform rather than calling inv(), because U is
    # already (n, N) from correlated_rvs, while inv() expects F as (1, N).
    Fc = (c - a) / (b - a)                        # (n, 1) CDF value at the mode

    X_below = a + np.sqrt( U      * (b - a) * (c - a))  # (n, N) branch values
    X_above = b - np.sqrt((1 - U) * (b - a) * (b - c))

    X = np.where(U < Fc, X_below, X_above)        # (n, N)

    # Squeeze singleton axes from the (n, N) output
    squeeze_axes = tuple(np.where(np.asarray([n, N]) == 1)[0])

    return np.squeeze(X, axis=squeeze_axes)
