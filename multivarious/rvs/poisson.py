#! /usr/bin/env -S python3 -i
## Poisson distribution
# github.com/hpgavin/multivarious ... rvs/poisson

import numpy as np
from scipy.special import gammaln, gammaincc

from multivarious.utl.correlated_rvs import correlated_rvs


def _validate_(t, T):
    """
    Validate and preprocess Poisson distribution parameters.

    Converts t and T to (n, 1) column arrays for broadcasting against
    a (1, N_k) row array of event counts k, producing (n, N_k) output.

    INPUTS
        t : float or array_like   observation duration(s),    must be > 0
        T : float or array_like   mean return period(s),      must be > 0

    OUTPUTS
        t : ndarray, shape (n, 1)
        T : ndarray, shape (n, 1)
    """
    t = np.asarray(t, dtype=float).reshape(-1, 1)  # (n, 1)
    T = np.asarray(T, dtype=float).reshape(-1, 1)  # (n, 1)

    if t.shape != T.shape:
        raise ValueError(f"poisson: t and T must have the same length. "
                         f"Got t:{t.size}, T:{T.size}")
    if np.any(t <= 0):
        raise ValueError("poisson: all t values must be > 0")
    if np.any(T <= 0):
        raise ValueError("poisson: all T values must be > 0")

    return t, T


def pmf(k, t, T):
    """
    poisson.pmf

    Computes the PMF of the Poisson distribution.

    INPUTS
        k : int or array_like, shape (N_k,)   number of observed events (>= 0)
        t : float or array_like, shape (n,)   observation duration(s),  > 0
        T : float or array_like, shape (n,)   mean return period(s),    > 0

    OUTPUTS
        p : ndarray, shape (n, N_k)   PMF values; singleton axes are squeezed

    Notes
    -----
    The Poisson rate is r = t/T.
    p(k; r) = r^k * exp(-r) / k!
    Computed in log space for numerical stability:
    log p = k*log(r) - r - log(k!)   where log(k!) = gammaln(k+1)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Poisson_distribution
    """
    t, T = _validate_(t, T)                                 # (n, 1)
    k = np.asarray(np.round(k), dtype=int).reshape( 1, -1)  # (1, N_k)
    k = np.where(k < 0, 0, k                         # clip negative counts to 0

    r = t / T                                        # (n, 1) Poisson rate

    # Log-space computation for numerical stability, then exponentiate
    p = np.exp(k * np.log(r) - r - gammaln(k + 1))           # (n, N_k)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [t, k].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([t, k]) if v.size == 1)

    return np.squeeze(p, axis=squeeze_axes)


def cdf(k, params):
    """
    poisson.cdf

    Computes the CDF of the Poisson distribution.

    INPUTS
        k      : int or array_like, shape (N_k,)   number of events (>= 0)
        params : tuple (t, T)
            t : float or array_like, shape (n,)   observation duration(s),  > 0
            T : float or array_like, shape (n,)   mean return period(s),    > 0

    OUTPUTS
        F : ndarray, shape (n, N_k)   CDF values; singleton axes are squeezed

    Notes
    -----
    Uses the regularized upper incomplete gamma function:
        F(k; r) = P(X <= k) = gammaincc(k+1, r)   where r = t/T

    gammaincc supports NumPy broadcasting, so no loop is needed.

    Reference
    ---------
    https://en.wikipedia.org/wiki/Incomplete_gamma_function#Regularized_gamma_functions_and_Poisson_random_variables
    """
    t, T = params
    t, T = _validate_(t, T)                                  # (n, 1)
    k = np.asarray(np.round(k), dtype=int).reshape( 1, -1)   # (1, N_k)
    k = np.where(k < 0, 0, k)                                # clip negative counts to 0

    r = t / T                                                # (n, 1) Poisson rate

    # gammaincc broadcasts (1,N_k) k+1 against (n,1) r → (n, N_k)
    F = gammaincc(k + 1, r)                                   # (n, N_k)

    # Find singleton axes corresponding to scalar or length-1 inputs.
    # enumerate() creates pairs of (index, value)
    # for each element in the list [t, k].
    # The index i is included in squeeze_axes if v.size == 1 for that element.
    squeeze_axes = tuple(i for i, v in enumerate([t, k]) if v.size == 1)

    return np.squeeze(F, axis=squeeze_axes)


def rnd(t, T, N, R=None, seed=None):
    """
    poisson.rnd

    Generate random samples from the Poisson distribution using the
    inversion by sequential search algorithm.

    INPUTS
        t    : float or array_like, shape (n,)   observation duration(s),  > 0
        T    : float or array_like, shape (n,)   mean return period(s),    > 0
        N    : int                                number of samples per process
        R    : ndarray, shape (n, n), optional    correlation matrix;
               if None, generates uncorrelated samples
        seed : int or None                        random seed for reproducibility

    OUTPUTS
        X : ndarray, shape (n, N)   Poisson counts; singleton axes are squeezed

    Notes
    -----
    Uses correlated uniform variates from correlated_rvs, then applies the
    Poisson inversion by sequential search (Devroye 1986) vectorized over
    all n processes and N samples simultaneously via (n, N) boolean masks.

    Algorithm (per element [i, j]):
        x = 0, p = exp(-r), s = p
        while U[i,j] > s:
            x += 1;   p *= r / x;   s += p
        return x

    Reference
    ---------
    Devroye, L. (1986). Non-Uniform Random Variate Generation, pp. 485-553.
    https://en.wikipedia.org/wiki/Poisson_distribution#Computational_methods
    """
    if N is None or N < 1:
        raise ValueError("poisson.rnd: N must be greater than zero")

    t, T = _validate_(t, T)                         # (n, 1)
    n = t.size

    r = t / T                                       # (n, 1) Poisson rate

    # Generate n correlated uniform [0, 1] variates, shape (n, N)
    _, _, U = correlated_rvs(R, n, N, seed)

    # Inversion by sequential search, vectorized over all (n, N) simultaneously.
    # p and s carry the PMF and CDF running values; active marks unsettled elements.
    X = np.zeros((n, N), dtype=int)
    p = np.broadcast_to(np.exp(-r), (n, N)).copy()  # (n, N) initial PMF = exp(-r)
    s = p.copy()                                    # (n, N) initial CDF = exp(-r)
    active = U > s                                  # (n, N) elements still searching

    max_iter = 1000
    for iteration in range(max_iter):
        if not np.any(active):
            break
        X  = np.where(active, X + 1, X)             # increment active counts
        # r / X: (n,1) / (n,N) broadcasts correctly; guard X==0 with clip
        p  = np.where(active, p * r / np.where(X > 0, X, 1), p)  # update PMF term
        s  = np.where(active, s + p, s)             # update running CDF
        active = U > s                              # recheck which are unsettled
    else:
        print(f"poisson.rnd: sequential search did not converge within {max_iter} iterations")

    # Squeeze singleton axes from the (n, N) output
    squeeze_axes = tuple(np.where(np.asarray([n, N]) == 1)[0])

    return np.squeeze(X, axis=squeeze_axes)
