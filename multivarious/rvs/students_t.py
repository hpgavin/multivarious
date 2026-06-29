#! /usr/bin/env -S python3 -i
## Student's t distribution
# github.com/hpgavin/multivarious ... rvs/students_t

import numpy as np
from scipy.special import betainc, betaincinv, gamma 

from multivarious.utl.correlated_rvs import correlated_rvs


def _validate_(k):
    """
    Validate and preprocess Student's t distribution parameters.

    Converts k to an (n, 1) column array for broadcasting against
    a (1, N) row array of evaluation points, producing (n, N) output.

    INPUTS
        k : int or float or array_like   degrees of freedom, must be > 0

    OUTPUTS
        k : ndarray, shape (n, 1), dtype int
    """
    k = np.asarray(k, dtype=int).reshape(-1, 1)  # (n, 1)

    if np.any(k <= 0):
        raise ValueError("students_t: k must be > 0")

    return k


def pdf(t, k):
    """
    students_t.pdf

    Computes the PDF of the Student's t-distribution with k degrees of freedom.

    INPUTS
        t : float or array_like, shape (N,)   evaluation points
        k : int or array_like,   shape (n,)   degrees of freedom, must be > 0

    OUTPUTS
        f : ndarray, shape (n, N)   PDF values; singleton axes are squeezed

    Notes
    -----
    f(t) = Gamma((k+1)/2) / (sqrt(pi*k) * Gamma(k/2)) * (1 + t^2/k)^(-(k+1)/2)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Student%27s_t-distribution
    """
    k = _validate_(k)                                       # (n, 1)
    t = np.asarray(t, dtype=float).reshape( 1, -1)          # (1, N)

    numerator   = gamma((k + 1) / 2.0)                      # (n, 1)
    denominator = gamma(k / 2.0) * np.sqrt(k * np.pi)       # (n, 1)

    f = (numerator / denominator) * (1.0 + t**2 / k) ** (-(k + 1) / 2.0)  # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [k, t].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([k, t]) if v.size == 1)

    return np.squeeze(f, axis=squeeze_axes)


def cdf(t, k):
    """
    students_t.cdf

    Computes the CDF of the Student's t-distribution with k degrees of freedom.

    INPUTS
        t : float or array_like, shape (N,)   evaluation points
        k : int or array_like,   shape (n,)   degrees of freedom, must be > 0

    OUTPUTS
        F : ndarray, shape (n, N)   CDF values; singleton axes are squeezed

    Notes
    -----
    Uses the regularized incomplete beta function:
        x = k / (k + t^2)
        F(t) = 1 - betainc(k/2, 1/2, x) / 2   for t > 0
        F(t) =     betainc(k/2, 1/2, x) / 2   for t < 0
        F(0) = 0.5

    betainc supports NumPy broadcasting, so no loop over n is needed.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Student%27s_t-distribution
    """
    k = _validate_(k)                                     # (n, 1)
    t = np.asarray(t, dtype=float).reshape( 1, -1)        # (1, N)

    x = k / (k + t**2)                                    # (n, N)

    # betainc broadcasts (n,1) k/2 against (n,N) x → (n,N)
    Ibx = betainc(k / 2.0, 0.5, x)                        # (n, N)

    F = np.where(t == 0, 0.5,
        np.where(t  > 0, 1.0 - 0.5 * Ibx,
                         0.5 * Ibx))                      # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [k, t].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([k, t]) if v.size == 1)

    return np.squeeze(F, axis=squeeze_axes)


def inv(F, k):
    """
    students_t.inv

    Computes the inverse CDF (quantile function) of the Student's t-distribution.

    INPUTS
        F : float or array_like, shape (N,)   probability values in [0, 1]
        k : int or array_like,   shape (n,)   degrees of freedom, must be > 0

    OUTPUTS
        x : ndarray, shape (n, N)   quantile values; singleton axes are squeezed

    Notes
    -----
    Uses the relationship with the inverse incomplete beta function:
        z = betaincinv(k/2, 1/2, 2*min(F, 1-F))
        x = sign(F - 0.5) * sqrt(k * (1/z - 1))

    betaincinv supports NumPy broadcasting, so no loop over n is needed.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Student%27s_t-distribution
    """
    k = _validate_(k)                                           # (n, 1)
    if F.ndim <= 1:
        F = F.reshape(1, -1)   # (1, N) - shared F grid for all n variables
    F = np.clip(F, np.finfo(float).eps, 1.0 - np.finfo(float).eps)

    # betaincinv broadcasts (n,1) k/2 against (1,N) F → (n,N)
    z = betaincinv(k / 2.0, 0.5, 2.0 * np.minimum(F, 1.0 - F))  # (n, N)
    x = np.sign(F - 0.5) * np.sqrt(k * (1.0 / z - 1.0))         # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [k, F].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([k, F]) if v.size == 1)

    return np.squeeze(x, axis=squeeze_axes)


def rnd(k, N, R=None, seed=None):
    """
    students_t.rnd

    Generate random samples from the Student's t-distribution.

    INPUTS
        k    : int or array_like, shape (n,)   degrees of freedom, must be > 0
        N    : int                              number of samples per variable
        R    : ndarray, shape (n, n), optional  correlation matrix;
               if None, generates uncorrelated samples
        seed : int or None                      random seed for reproducibility

    OUTPUTS
        X : ndarray, shape (n, N)   Student's t random samples;
            each row is one random variable, each column one sample;
            singleton axes are squeezed

    Notes
    -----
    Uses the inverse transform method with correlated uniform variates.
    The inverse transform is inlined rather than calling inv(), because U is
    already (n, N) from correlated_rvs, while inv() expects F as (1, N).

    Reference
    ---------
    https://en.wikipedia.org/wiki/Student%27s_t-distribution
    """
    if N is None or N < 1:
        raise ValueError("students_t.rnd: N must be greater than zero")

    k = _validate_(k)                                           # (n, 1)
    n = k.size

    # Generate n correlated uniform [0, 1] variates, shape (n, N)
    _, _, U = correlated_rvs(R, n, N, seed)

    # Inline the inverse transform rather than calling inv(), because U is
    # already (n, N) from correlated_rvs, while inv() expects F as (1, N).
    # betaincinv broadcasts (n,1) k/2 against (n,N) U → (n,N)
    z = betaincinv(k / 2.0, 0.5, 2.0 * np.minimum(U, 1.0 - U))  # (n, N)
    X = np.sign(U - 0.5) * np.sqrt(k * (1.0 / z - 1.0))         # (n, N)

    # Squeeze singleton axes from the (n, N) output
    squeeze_axes = tuple(np.where(np.asarray([n, N]) == 1)[0])

    return np.squeeze(X, axis=squeeze_axes)
