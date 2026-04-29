import numpy as np
from scipy.special import factorial

from multivarious.utl.correlated_rvs import correlated_rvs
from scipy.special import comb  # For binomial coefficient


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
    nb = len(p)   
    Nb = len(n)
    
    n = np.where(n < 0, 0, n)  # clip negative values to 0

    # Check parameter validity
    if np.any(p <= 0) or np.any(np.isinf(p)) or np.any( p >= 1):
        raise ValueError(" binomial: p must be between zero and one") 

    if not ( len(m) == nb ):
        raise ValueError(f" m array must have the same length as p. "
                         f"Got m:{len(m)}, p:{len(p)}")

    # n, m, p must be non-negative
    n[ n < 0 ] = 0   
    m[ m < 0 ] = 0
    p[ p < 0 ] = 0.0

    return n, m, p, nb, Nb



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
    # Convert inputs to numpy arrays for vectorized operations
    n = np.asarray(n)
    m = np.asarray(m)
    p = np.asarray(p)

    # Validate inputs
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError("Probability `p` must be in the range [0, 1].")
    if np.any(m < n):
        raise ValueError("Number of attempts `m` must be >= number of successes `n`.")

    # Compute the binomial coefficient: C(m, n) = m! / (n! * (m-n)!)
    binomial_coeff = np.vectorize(comb)(m, n, exact=True)

    # Compute the PMF: P(n; m, p) = C(m, n) * p^n * (1-p)^(m-n)
    P = binomial_coeff * (p ** n) * ((1 - p) ** (m - n))

    return P


def cdf(n, params):
    '''
    binomial.cdf

    Computes the Cumulative Distribution Function (CDF) of the Binomial distribution.

    INPUTS:
        n : number of successful outcomes (Nb,)
        m : int scalar or array_like (nb,)
            number of attempts 
        p : float scalar or array_like (nb,)
            event probabilities of a single attempt for each variable
    Output:
        F : ndarray or float (nb,Nb)
            Cumulative probability P(N ≤ n)

    Reference:
    https://en.wikipedia.org/wiki/Binomial_distribution
    '''
    m, p = params 

    n, m, p, nb, Nb =  _ppp_( n, m, p )

    F = np.zeros((nb,Nb))

    P = pmf( n, m, p )

    for i in range(nb): 
        F[i,:] = np.cumsum( P[i,:] )

#   if F.size > 1 else F[0]        # Return scalar if input was scalar
    if nb == 1 and F.shape[0] == 1:
         F = F.flatten();

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

    _, m, p, nb, _ =  _ppp_( 0, m, p )

    if N == None or N < 1:
        raise ValueError(" binomial.rnd(p,N): N must be greater than zero")
    
    rng = np.random.default_rng(seed)

    X = np.zeros((nb, N), dtype=int)
    
    # Accumulate successes over attempts
    for attempt in range(np.max(m)):

        _, _, U = correlated_rvs(R, nb, N, seed)

        successes = (U < p) # Bernoulli success if U < p, shape (nb, N)
    
        # add up successes for the i-th r.v. only up to m[i] attempts 
        for i in range(nb):
            if attempt <= m[i]:
                X[i,:] += successes[i,:].astype(int)

    if n == 1 and X.shape[0] == 1:
        X = X.flatten()

    if N == 1:
        X = X.flatten()

    return X
