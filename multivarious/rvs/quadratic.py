#! /usr/bin/env -S python3 -i
## quadratic distribution
# github.com/hpgavin/multivarious ... rvs/quadratic

import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs


def _validate_(a, b):
    """
    Validate and preprocess quadratic distribution parameters.

    Converts a and b to (n, 1) column arrays for broadcasting against
    a (1, N) row array of evaluation points, producing (n, N) output.

    INPUTS
        a : float or array_like   lower bound(s)
        b : float or array_like   upper bound(s), must be > a element-wise

    OUTPUTS
        a : ndarray, shape (n, 1)
        b : ndarray, shape (n, 1)
    """
    a = np.asarray(a, dtype=float).reshape(-1, 1)  # (n, 1)
    b = np.asarray(b, dtype=float).reshape(-1, 1)  # (n, 1)

    if a.shape != b.shape:
        raise ValueError(f"quadratic: a and b must have the same length. "
                         f"Got a:{a.size}, b:{b.size}")
    if np.any(b <= a):
        raise ValueError("quadratic: all b values must be greater than "
                         "corresponding a values")
    return a, b


def pdf(x, a, b):
    """
    quadratic.pdf

    Computes the PDF of the quadratic distribution on (a, b).

    INPUTS
        x : float or array_like, shape (N,)   evaluation points
        a : float or array_like, shape (n,)   lower bound(s)
        b : float or array_like, shape (n,)   upper bound(s), must be > a

    OUTPUTS
        f : ndarray, shape (n, N)   PDF values; singleton axes are squeezed

    Notes
    -----
    f(x) = 6*(x-a)*(x-b) / (a-b)^3   for a < x < b,   0 otherwise

    Reference
    ---------
    https://en.wikipedia.org/wiki/U-quadratic_distribution
    """
    a, b = _validate_(a, b)                          # (n, 1)
    x = np.asarray(x, dtype=float).reshape( 1, -1)   # (1, N)

    inside  = (a < x) & (x < b)                      # (n, N) boolean mask
    f_inside = 6.0 * (x - a) * (x - b) / (a - b)**3  # (n, N) formula values

    f = np.where(inside, f_inside, 0.0)              # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [a, x].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([a, x]) if v.size == 1)

    return np.squeeze(f, axis=squeeze_axes)


def cdf(x, params):
    """
    quadratic.cdf

    Computes the CDF of the quadratic distribution on (a, b).

    INPUTS
        x      : float or array_like, shape (N,)   evaluation points
        params : tuple (a, b)
            a : float or array_like, shape (n,)   lower bound(s)
            b : float or array_like, shape (n,)   upper bound(s), must be > a

    OUTPUTS
        F : ndarray, shape (n, N)   CDF values; singleton axes are squeezed

    Notes
    -----
    F(x) = (a-x)^2 * (a - 3b + 2x) / (a-b)^3   for a <= x <= b
    F(x) = 0 for x <= a,   1 for x >= b

    Reference
    ---------
    https://en.wikipedia.org/wiki/U-quadratic_distribution
    """
    a, b = params
    a, b = _validate_(a, b)                           # (n, 1)
    x = np.asarray(x, dtype=float).reshape( 1, -1)  # (1, N)

    inside   = (a <  x) & (x <  b)                  # (n, N) boolean masks
    above    = (x >= b)

    F_inside = (a - x)**2 * (a - 3*b + 2*x) / (a - b)**3  # (n, N) formula values

    F = np.where(above, 1.0, np.where(inside, F_inside, 0.0))  # (n, N)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [a, x].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([a, x]) if v.size == 1)

    return np.squeeze(F, axis=squeeze_axes)


def inv(F, a, b):
    """
    quadratic.inv

    Computes the inverse CDF (quantile function) by solving cubic equations.

    INPUTS
        F : float or array_like, shape (N,) or (n, N)
            Probability values in [0, 1].
            Shape (1, N) or (N,): same probabilities applied to all n r.v.'s.
            Shape (n, N): row i gives probabilities for r.v. i (used by rnd).
        a : float or array_like, shape (n,)   lower bound(s)
        b : float or array_like, shape (n,)   upper bound(s), must be > a

    OUTPUTS
        x : ndarray, shape (n, N)   quantile values; singleton axes are squeezed

    Notes
    -----
    The cubic equation 2x^3 - 3(a+b)x^2 + 6ab*x
        + (a^3 - 3a^2*b - F*(a-b)^3) = 0
    has exactly one real root in (a, b) for each F in (0, 1).

    The cubic solve via np.roots is inherently scalar, so a double loop
    over n r.v.'s and N probability values is unavoidable here.
    """
    a, b = _validate_(a, b)                                    # (n, 1)
    n = a.size

    # Accept F as 1D (N,) or 2D (n, N); atleast_2d broadcasts (1, N) over n
    F = np.atleast_2d(np.asarray(F, dtype=float))
    F = np.clip(F, np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    N = F.shape[1]

    x = np.zeros((n, N))

    for i in range(n):
        ai = float(a[i])                                       # scalar for np.roots
        bi = float(b[i])
        for j in range(N):
            Fij = float(F[min(i, F.shape[0]-1), j])           # support (1,N) or (n,N)

            if Fij < np.sqrt(np.finfo(float).eps):
                x[i, j] = ai
            elif Fij > 1.0 - np.sqrt(np.finfo(float).eps):
                x[i, j] = bi
            else:
                # Coefficients of 2x^3 - 3(a+b)x^2 + 6ab*x + (a^3 - 3a^2*b - F*(a-b)^3)
                coeffs = [
                    2,
                    -3 * (ai + bi),
                    6 * ai * bi,
                    ai**3 - 3 * ai**2 * bi - Fij * (ai - bi)**3
                ]
                roots = np.roots(coeffs)

                # Retain only real roots inside (a, b)
                real_roots  = roots[np.abs(roots.imag) < 1e-10].real
                valid_roots = real_roots[(real_roots > ai) & (real_roots < bi)]

                if len(valid_roots) != 1:
                    raise ValueError(
                        f"quadratic.inv: expected 1 root in ({ai}, {bi}), "
                        f"found {len(valid_roots)} for F={Fij:.6g}")

                x[i, j] = valid_roots[0]

    # Squeeze singleton axes from the (n, N) output
    squeeze_axes = tuple(np.where(np.asarray([n, N]) == 1)[0])

    return np.squeeze(x, axis=squeeze_axes)


def rnd(a, b, N, R=None, seed=None):
    """
    quadratic.rnd

    Generate random samples from the quadratic distribution on (a, b).

    INPUTS
        a    : float or array_like, shape (n,)   lower bound(s)
        b    : float or array_like, shape (n,)   upper bound(s), must be > a
        N    : int                                number of samples per variable
        R    : ndarray, shape (n, n), optional    correlation matrix;
               if None, generates uncorrelated samples
        seed : int or None                        random seed for reproducibility

    OUTPUTS
        X : ndarray, shape (n, N)   quadratic random samples;
            each row is one random variable, each column one sample;
            singleton axes are squeezed

    Notes
    -----
    Uses the inverse transform method.  The quadratic CDF has no closed-form
    inverse, so inv() is called with U of shape (n, N) — one row per r.v.
    """
    if N is None or N < 1:
        raise ValueError("quadratic.rnd: N must be greater than zero")

    a, b = _validate_(a, b)                           # (n, 1)
    n = a.size

    # Generate n correlated uniform [0, 1] variates, shape (n, N)
    _, _, U = correlated_rvs(R, n, N, seed)

    # inv() accepts F of shape (n, N): row i holds the probabilities for r.v. i
    X = inv(U, a, b)                                  # (n, N) or squeezed

    return X
