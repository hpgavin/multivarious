#! /usr/bin/env -S python3 -i
## beta distribution
# github.com/hpgavin/multivarious ... rvs/beta

import numpy as np
from scipy.special import beta     as beta_func
from scipy.special import betainc
from scipy.special import betaincinv

from multivarious.utl.correlated_rvs import correlated_rvs


def _validate_(a, b, q, p):
    """
    Validate and preprocess beta distribution parameters.

    Converts a, b, q, p to (n, 1) column arrays for broadcasting against
    a (1, N) row array of evaluation points, producing (n, N) output.

    INPUTS
        a : float or array_like   lower bound(s)
        b : float or array_like   upper bound(s), must be > a element-wise
        q : float or array_like   first shape parameter(s),  must be > 0
        p : float or array_like   second shape parameter(s), must be > 0

    OUTPUTS
        a : ndarray, shape (n, 1)
        b : ndarray, shape (n, 1)
        q : ndarray, shape (n, 1)
        p : ndarray, shape (n, 1)
    """
    a = np.asarray(a, dtype=float).reshape(-1, 1)  # (n, 1)
    b = np.asarray(b, dtype=float).reshape(-1, 1)  # (n, 1)
    q = np.asarray(q, dtype=float).reshape(-1, 1)  # (n, 1)
    p = np.asarray(p, dtype=float).reshape(-1, 1)  # (n, 1)

    if not (a.shape == b.shape == q.shape == p.shape):
        raise ValueError(f"beta: a, b, q, p must have the same length. "
                         f"Got a:{a.size}, b:{b.size}, q:{q.size}, p:{p.size}")
    if np.any(b <= a):
        raise ValueError("beta: all b values must be greater than corresponding a values")
    if np.any(q <= 0):
        raise ValueError("beta: q must be positive")
    if np.any(p <= 0):
        raise ValueError("beta: p must be positive")

    return a, b, q, p


def pdf(x, a, b, q, p):
    """
    beta.pdf

    Computes the PDF of the beta distribution on [a, b] with shape parameters q, p.

    INPUTS
        x : float or array_like, shape (N,)   evaluation points
        a : float or array_like, shape (n,)   lower bound(s)
        b : float or array_like, shape (n,)   upper bound(s), must be > a
        q : float or array_like, shape (n,)   first shape parameter(s),  > 0
        p : float or array_like, shape (n,)   second shape parameter(s), > 0

    OUTPUTS
        f : ndarray, shape (n, N)   PDF values; singleton axes are squeezed

    Notes
    -----
    f(x) = (x-a)^(q-1) * (b-x)^(p-1) / (B(q,p) * (b-a)^(q+p-1))
    for x in [a, b], where B(q, p) is the beta function.

    x is clipped to (a, b) before evaluating the formula to avoid NaN from
    raising negative numbers to fractional powers outside [a, b]; the mask
    then zeros out those regions in the result.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Beta_distribution
    """
    a, b, q, p = _validate_(a, b, q, p)                      # (n, 1)
    x = np.asarray(x, dtype=float).reshape( 1, -1)          # (1, N)

    inside = (a <= x) & (x <= b)                             # (n, N) boolean mask

    # Clip x to (a, b) before computing the formula to suppress NaN/inf
    # warnings outside [a, b]; the mask zeros out those values in the result.
    xc = np.clip(x, a + np.finfo(float).eps, b - np.finfo(float).eps)

    numerator   = (xc - a)**(q - 1) * (b - xc)**(p - 1)    # (n, N)
    denominator = beta_func(q, p) * (b - a)**(q + p - 1)    # (n, 1)

    f = np.where(inside, numerator / denominator, 0.0)       # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [a, x].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([a, x]) if v.size == 1)

    return np.squeeze(f, axis=squeeze_axes)


def cdf(x, params):
    """
    beta.cdf

    Computes the CDF of the beta distribution on [a, b] with shape parameters q, p.

    INPUTS
        x      : float or array_like, shape (N,)   evaluation points
        params : tuple (a, b, q, p)
            a : float or array_like, shape (n,)   lower bound(s)
            b : float or array_like, shape (n,)   upper bound(s), must be > a
            q : float or array_like, shape (n,)   first shape parameter(s),  > 0
            p : float or array_like, shape (n,)   second shape parameter(s), > 0

    OUTPUTS
        F : ndarray, shape (n, N)   CDF values; singleton axes are squeezed

    Notes
    -----
    F(x) = I_{(x-a)/(b-a)}(q, p)
    where I is the regularized incomplete beta function (scipy.special.betainc).
    betainc supports NumPy broadcasting, so no loop over n is needed.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Beta_distribution
    """
    a, b, q, p = params
    a, b, q, p = _validate_(a, b, q, p)                      # (n, 1)
    x = np.asarray(x, dtype=float).reshape( 1, -1)          # (1, N)

    # Normalize x to [0, 1]; clip handles x < a (→ 0) and x > b (→ 1)
    z = np.clip((x - a) / (b - a), 0.0, 1.0)               # (n, N)

    # betainc broadcasts (n,1) q,p against (n,N) z → (n,N)
    F = betainc(q, p, z)                                     # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [a, x].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([a, x]) if v.size == 1)

    return np.squeeze(F, axis=squeeze_axes)


def inv(F, a, b, q, p):
    """
    beta.inv

    Computes the inverse CDF (quantile function) of the beta distribution.

    INPUTS
        F : float or array_like, shape (N,)   probability values in [0, 1]
        a : float or array_like, shape (n,)   lower bound(s)
        b : float or array_like, shape (n,)   upper bound(s), must be > a
        q : float or array_like, shape (n,)   first shape parameter(s),  > 0
        p : float or array_like, shape (n,)   second shape parameter(s), > 0

    OUTPUTS
        x : ndarray, shape (n, N)   quantile values; singleton axes are squeezed

    Notes
    -----
    x = a + betaincinv(q, p, F) * (b - a)
    betaincinv supports NumPy broadcasting, so no loop over n is needed.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Beta_distribution
    """
    a, b, q, p = _validate_(a, b, q, p)                      # (n, 1)
    F = np.asarray(F, dtype=float).reshape( 1, -1)          # (1, N)
    F = np.clip(F, np.finfo(float).eps, 1.0 - np.finfo(float).eps)

    # betaincinv broadcasts (n,1) q,p against (1,N) F → (n,N)
    z = betaincinv(q, p, F)                                  # (n, N)
    x = a + z * (b - a)                                      # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [a, F].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([a, F]) if v.size == 1)

    return np.squeeze(x, axis=squeeze_axes)


def rnd(a, b, q, p, N, R=None, seed=None):
    """
    beta.rnd

    Generate random samples from the beta distribution on [a, b].

    INPUTS
        a    : float or array_like, shape (n,)   lower bound(s)
        b    : float or array_like, shape (n,)   upper bound(s), must be > a
        q    : float or array_like, shape (n,)   first shape parameter(s),  > 0
        p    : float or array_like, shape (n,)   second shape parameter(s), > 0
        N    : int                                number of samples per variable
        R    : ndarray, shape (n, n), optional    correlation matrix;
               if None, generates uncorrelated samples
        seed : int or None                        random seed for reproducibility

    OUTPUTS
        X : ndarray, shape (n, N)   beta random samples;
            each row is one random variable, each column one sample;
            singleton axes are squeezed

    Notes
    -----
    Uses the Gaussian copula method via inverse transform:
        X = a + betaincinv(q, p, U) * (b - a)
    where U ~ Uniform(0,1) with correlation structure from correlated_rvs.

    The inverse transform is inlined rather than calling inv(), because U is
    already (n, N) from correlated_rvs, while inv() expects F as (1, N).

    Reference
    ---------
    https://en.wikipedia.org/wiki/Beta_distribution
    """
    if N is None or N < 1:
        raise ValueError("beta.rnd: N must be greater than zero")

    a, b, q, p = _validate_(a, b, q, p)                      # (n, 1)
    n = a.size

    # Generate n correlated uniform [0, 1] variates, shape (n, N)
    _, _, U = correlated_rvs(R, n, N, seed)

    # Inline the inverse transform rather than calling inv(), because U is
    # already (n, N) from correlated_rvs, while inv() expects F as (1, N).
    # betaincinv broadcasts (n,1) q,p against (n,N) U → (n,N)
    z = betaincinv(q, p, U)                                   # (n, N)
    X = a + z * (b - a)                                       # (n, N)

    # Squeeze singleton axes from the (n, N) output
    squeeze_axes = tuple(np.where(np.asarray([n, N]) == 1)[0])

    return np.squeeze(X, axis=squeeze_axes)
