import numpy as np
from scipy.special import comb  # For binomial coefficient

from multivarious.utl.correlated_rvs import correlated_rvs

def _ppp_(n, m, p):
    '''
    Validate and preprocess input parameters for consistency and correctness.

    INPUTS:
        n : number of successful outcomes (N,)
        m : int scalar or array_like (n,)
            number of attempts 
        p : float scalar or array_like (n,)
            event probabilities of a single attempt for each variable
    '''

    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    p = np.atleast_1d(p).reshape(-1,1).astype(float)
    n = np.atleast_1d(np.round(n)).reshape(-1,1).astype(int)  # n is int
    m = np.atleast_1d(np.round(m)).reshape(-1,1).astype(int)  # m is int
    nv = len(p)  # number of random variables 
    Nn = len(n)  # number of successes for each random variable 
    
    # Check parameter validity
    if np.any(p <= 0) or np.any(np.isinf(p)) or np.any( p >= 1):
        raise ValueError(" binomial: p must be between zero and one") 

    if not ( len(m) == nv ):
        raise ValueError(f" m array must have the same length as p. "
                         f"Got m:{len(m)}, p:{len(p)}")

    # n, m, p must be non-negative
    n[ n < 0 ] = 0   
    m[ m < 0 ] = 0
    p[ p < 0 ] = 0.0

    return n, m, p, nv, Nn


def pmf(n, m, p):
    """
    Computes the Probability Mass Function (PMF) of the Binomial distribution.

    Parameters:
    -----------
    n : int or array_like
        Number of successful outcomes (can be scalar or array).
    m : int or array_like
        Number of attempts (must be >= n for each element).
    p : float or array_like
        Probability of success in a single attempt (must be in [0, 1]).

    Returns:
    --------
    P : ndarray
        Probability of observing `n` successes in `m` attempts.
        Shape matches the broadcasted shape of `n`, `m`, and `p`.

    Reference:
    ----------
    https://en.wikipedia.org/wiki/Binomial_distribution

    Notes:
    ------
    The Binomial distribution models the number of successes in `m` independent
    trials, each with success probability `p`.
    """


    # Validate inputs
    assert all(np.asarray(a).ndim <= 1 for a in [p, n, m]), \
        "n, m, p must be scalars or 1D arrays"

    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("Probability `p` must be in the range [0, 1].")
    if np.any(m < n):
        raise ValueError("Number of attempts `m` must be >= number of successes `n`.")

    # Convert inputs to numpy arrays for vectorized operations
    p = np.asarray(p, dtype=float).reshape(-1, 1, 1)  # (P, 1, 1)
    n = np.asarray(n, dtype=int  ).reshape( 1,-1, 1)  # (1, N, 1)
    m = np.asarray(m, dtype=int  ).reshape( 1, 1,-1)  # (M, 1, 1)

    # After reshaping, each of p, n, m occupies one axis of the computed 
    # array of Binomial probabilites - as a (P, N, M) 3-dimensional output

    # Compute the binomial coefficient: C(m, n) = m! / (n! * (m-n)!)
    binomial_coeff = np.vectorize(comb)(m, n, exact=True)

    # Compute the PMF: P(n; m, p) = C(m, n) * p^n * (1-p)^(m-n)
    P = binomial_coeff * (p ** n) * ((1 - p) ** (m - n))

    # Find singleton axes 
    # for each element in the list [ p, n, m ]  
    # enumerate() creates pairs of (index, value)
    # The index idx is included in the tuple if val.size == 1 for that index

    squeeze_axes = tuple(idx for idx,val in enumerate([p, n, m]) if val.size==1)

    return np.squeeze(P, axis=squeeze_axes)


def cdf(n, params):
    '''
    binomial.cdf

    Computes the Cumulative Distribution Function (CDF) of the Binomial distribution.

    INPUTS:
        n : number of successful outcomes (Nn,)
        m : int scalar or array_like (nv,)
            number of attempts 
        p : float scalar or array_like (nv,)
            event probabilities of a single attempt for each variable
    Output:
        F : ndarray or float (nv,Nn)
            Cumulative probability P(N ≤ n)

    Reference:
    https://en.wikipedia.org/wiki/Binomial_distribution
    '''
    m, p = params 

    m = np.asarray(m, dtype=int)
    n = np.asarray(n, dtype=int)
    p = np.asarray(p, dtype=float)

    P = pmf( n, m, p )

    # Find singleton axes ...
    # for each element in the list [ p, n, m ]  
    # enumerate() creates pairs of (index, value)
    # The index idx is included in the tuple if val.size == 1 for that index

    squeeze_axes = tuple(idx for idx,val in enumerate([p, n, m]) if val.size==1)

    # After squeezing, n's axis index shifts left by the number of
    # axes before it (i.e. with index < 1) that were squeezed away.
    # n occupies axis 1 in the unsqueezed (P, N, M) array, so:
    #   n_axis = 1 - (number of squeeze_axes that are < 1)

    n_axis = 1 - sum(s < 1 for s in squeeze_axes)

    F = np.cumsum(P, axis=n_axis)  # cumulative sum of the PMF 

    return F 


def rnd(m, p, N, R=None, seed=None):
    '''
    Generate N samples of n correlated Binomial(m, p) variables with correlation matrix R.
    
    INPUTS:
        m : int scalar or array_like (n,)
            number of attempts 
        p : float scalar or array_like (n,)
            event probabilities of a single attempt for each variable
        N : int
            Number of samples to generate
        R : ndarray shape (n, n)
            Correlation matrix among the n Bernoulli variables at each trial
        seed : int or None
            Random seed
    
    OUTPUTS:
        samples : ndarray shape (n, N)
            Binomial counts for each variable and sample

    Notes:
    Uses the waiting time method (multiplicative) similar to Knuth's algorithm.
    Reference:
    https://en.wikipedia.org/wiki/Binomial_distribution#Generating_Binomial-distributed_random_variables
     '''

    if N == None or N < 1:
        raise ValueError(" binomial.rnd(p,N): N must be greater than zero")
    
    rng = np.random.default_rng(seed)

    m = np.asarray(m, dtype=int ).reshape(-1, 1)  # (nv, 1) broadcasts over N
    p = np.asarray(p, dtype=float).reshape(-1, 1) # (nv, 1) broadcasts over N

    X = np.zeros((nv, N), dtype=int)

    # Accumulate successes over attempts
    for attempt in range(np.max(m)):

        _, _, U = correlated_rvs(R, nv, N, seed)

        successes = (U < p)  # (nv, N) - Bernoulli success if U < p

        mask = (attempt < m) # (nv, 1) - True if this r.v. still has attempts remaining

        # add up successes for all r.v's 
        X += (successes * mask).astype(int)

    # Squeeze singleton axes from the (nv, N) output
    squeeze_axes = tuple(np.where(np.asarray([nv, N]) == 1)[0])
    X = np.squeeze(X, axis=squeeze_axes)

    return X
