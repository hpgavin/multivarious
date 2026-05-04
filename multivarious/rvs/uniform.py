#! /usr/bin/env -S python3 -i
## uniform distribution
# github.com/hpgavin/multivarious ... rvs/uniform

import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs


def _validate_(a, b):
    """
    Validate and preprocess uniform distribution parameters.

    Converts a and b to (n, 1) column arrays for broadcasting against
    a (1, N) row array of evaluation points, producing (n, N) output.

    INPUTS
        a : float or array_like   lower bound(s)
        b : float or array_like   upper bound(s), must be > a element-wise

    OUTPUTS
        a : ndarray, shape (n, 1)   lower bounds as column array
        b : ndarray, shape (n, 1)   upper bounds as column array
    """
    a = np.asarray(a, dtype=float).reshape(-1, 1)  # (n, 1)
    b = np.asarray(b, dtype=float).reshape(-1, 1)  # (n, 1)

    if a.shape != b.shape:
        raise ValueError(f"uniform: a and b must have the same length. "
                         f"Got a:{a.size}, b:{b.size}")
    if np.any(b <= a):
        raise ValueError("uniform: all b values must be greater than "
                         "corresponding a values")
    return a, b


def pdf(x, a, b):
    """
    uniform.pdf

    Computes the PDF of the uniform distribution on [a, b].

    INPUTS
        x : float or array_like, shape (N,)   evaluation points
        a : float or array_like, shape (n,)   lower bound(s)
        b : float or array_like, shape (n,)   upper bound(s), must be > a

    OUTPUTS
        f : ndarray, shape (n, N)   PDF values; singleton axes are squeezed

    Notes
    -----
    The uniform distribution has constant probability density 1/(b-a)
    over the interval [a, b] and zero elsewhere.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)
    """
    a, b = _validate_(a, b)                           # (n, 1)
    x = np.asarray(x, dtype=float).reshape( 1, -1)   # (1, N)

    # f = 1/(b-a) inside [a,b], 0 outside; broadcasts to (n, N)
    f = np.where((x >= a) & (x <= b), 1.0 / (b - a), 0.0)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [a, x].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([a, x]) if v.size == 1)

    return np.squeeze(f, axis=squeeze_axes)


def cdf(x, params):
    """
    uniform.cdf

    Computes the CDF of the uniform distribution on [a, b].

    INPUTS
        x      : float or array_like, shape (N,)   evaluation points
        params : tuple (a, b)
            a : float or array_like, shape (n,)   lower bound(s)
            b : float or array_like, shape (n,)   upper bound(s), must be > a

    OUTPUTS
        F : ndarray, shape (n, N)   CDF values; singleton axes are squeezed

    Notes
    -----
    F(x) = 0 for x < a, (x-a)/(b-a) for a <= x <= b, and 1 for x > b.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)
    """
    a, b = params
    a, b = _validate_(a, b)                           # (n, 1)
    x = np.asarray(x, dtype=float).reshape( 1, -1)   # (1, N)

    # clip to [0,1] handles x < a and x > b exactly; broadcasts to (n, N)
    F = np.clip((x - a) / (b - a), 0.0, 1.0)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [a, x].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([a, x]) if v.size == 1)

    return np.squeeze(F, axis=squeeze_axes)


def inv(F, a, b):
    """
    uniform.inv

    Computes the inverse CDF (quantile function) of the uniform distribution.

    INPUTS
        F : float or array_like, shape (N,)   probability values in [0, 1]
        a : float or array_like, shape (n,)   lower bound(s)
        b : float or array_like, shape (n,)   upper bound(s), must be > a

    OUTPUTS
        x : ndarray, shape (n, N)   quantile values; singleton axes are squeezed

    Notes
    -----
    x = a + F*(b - a) for F in [0, 1].

    Reference
    ---------
    https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)
    """
    a, b = _validate_(a, b)                           # (n, 1)
    F = np.asarray(F, dtype=float).reshape( 1, -1)   # (1, N)
    F = np.clip(F, np.finfo(float).eps, 1.0 - np.finfo(float).eps)

    x = a + F * (b - a)                              # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [a, F].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([a, F]) if v.size == 1)

    return np.squeeze(x, axis=squeeze_axes)


def rnd(a, b, N, R=None, seed=None):
    """
    uniform.rnd

    Generate random samples from the uniform distribution on [a, b].

    INPUTS
        a    : float or array_like, shape (n,)   lower bound(s)
        b    : float or array_like, shape (n,)   upper bound(s), must be > a
        N    : int                                number of samples per variable
        R    : ndarray, shape (n, n), optional    correlation matrix;
               if None, generates uncorrelated samples
        seed : int or None                        random seed for reproducibility

    OUTPUTS
        X : ndarray, shape (n, N)   uniform random samples;
            each row is one random variable, each column one sample;
            singleton axes are squeezed

    Notes
    -----
    Uses the inverse transform method with correlated uniform variates.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)
    """
    if N is None or N < 1:
        raise ValueError("uniform.rnd: N must be greater than zero")

    a, b = _validate_(a, b)                 # (n, 1)
    n = a.size

    # Generate n correlated uniform [0, 1] variates, shape (n, N)
    _, _, U = correlated_rvs(R, n, N, seed)

    # Inverse transform: x = a + U*(b - a); broadcasts (n,1) over (n,N)
    X = a + U * (b - a)                     # (n, N)

    # Squeeze singleton axes from the (n, N) output
    squeeze_axes = tuple(np.where(np.asarray([n, N]) == 1)[0])

    return np.squeeze(X, axis=squeeze_axes)
