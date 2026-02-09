import numpy as np
from scipy.special import factorial

from multivarious.utl.correlated_rvs import correlated_rvs



def _ppp_(p, n, m ):
    '''
    Validate and preprocess input parameters for consistency and correctness.

    Parameters:
        x : array_like
            Evaluation points
        a : float
            Minimum of the distribution
        b : float
            Maximum of the distribution (must be > a)
        q : float
            First shape parameter
        p : float
            Second shape parameter
    '''

    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    p = np.atleast_1d(p).astype(float)
    n = np.atleast_1d(np.round(n)).astype(int)  # Ensure n is integer-valued
    m = np.atleast_1d(np.round(m)).astype(int)  # Ensure m is integer-valued
    lp = len(p)   
    
    n = np.where(n < 0, 0, n)  # clip negative values to 0

    # Check parameter validity
    if np.any(p <= 0) or np.any(np.isinf(p)) or np.any( p >= 1):
        raise ValueError(" binomial.rnd(p,N): p must be between zero and one") 

    if not ( (len(n) == lp or len(n) == 1) and (len(m) == lp or len(m) == 1 ):
        raise ValueError(f"n and m arrays must have the same length as p. "
                         f"Got p:{len(p)}, n:{len(n)}, m:{len(m)}")
   
    if np.any(p <= 0):
        raise ValueError("binomial: p must be positive")
    if np.any(m < 0):
        raise ValueError("binomial: m must be positive")
    if np.any(n < 0):
        raise ValueError("binomial: n must be positive")

    return p, n, m, lp


def pmf(n, m, p):
    '''
    binomial.pmf

    Computes the Probability Mass Function (PMF) of the Binomial distribution.

    Parameters:
        n : array_like or scalar int
            Number of observed events (integer ≥ 0)
        m : scalar int
            the number of attempts (integer ≥ 0)
        p : scalar float positive valued
            probability of the occurance of an event in one attempt

    Output:
        p : ndarray
            Probability of observing n events out of m attempts

    Reference:
    https://en.wikipedia.org/wiki/Binomial_distribution

    Notes:
    The Binomial distribution models the number of times an event occurs in a 
    m attempts, with the probability of an occurance in one attempt = p
        * Events occur independently
        * The average rate of occurrence is constant (λ events per interval)
        * Two events can't happen at the exact same instant (events are discrete)
        * p = expected number of events in one attempt.
    '''

    p, n, m, _ =  _ppp_( p, n, m ):

    p = factorial(m) / (factorial(n) * factorial(m-n)) * p**n * (1-p)**(m-n)

    return p


def cdf(n, m, p):
    '''
    binomial.cdf

    Computes the Cumulative Distribution Function (CDF) of the Binomial distribution.

    Parameters:
        n : array_like or scalar
            Number of observed events (integer ≥ 0)
        m : float
            Number of attempts for the occurance of the event (integer ≥ 0)
        p : scalar float positive valued
            probability of the occurance of an event in one attempt

    Output:
        F : ndarray or float
            Cumulative probability P(N ≤ n)

    Reference:
    https://en.wikipedia.org/wiki/Binomial_distribution
    '''

    p, n, m, _ =  _ppp_(p, n, m ):

    F = np.empty_like(n, dtype=float)      # Initialize result array

    for i, ni in enumerate(n):
        ns = np.arange(0, ni + 1)          # k = 0 to n
        F[i] = np.sum( factorial(m) / (factorial(ns) * factorial(m-ns) ) * p**ns  * (1-p)**(m-ns) )

    return F if F.size > 1 else F[0]        # Return scalar if input was scalar


def rnd(m, p, N, R=None, seed=None):
    '''
    Generate N samples of n correlated Binomial(m, p) variables with correlation matrix R.
    
    Parameters:
        m : int
            Number of Bernoulli trials per Binomial variable
        p : float or array-like shape (n,)
            Probability of success per trial for each variable
        N : int
            Number of samples to generate
        R : ndarray shape (n, n)
            Correlation matrix among the n Bernoulli variables at each trial
        seed : int or None
            Random seed
    
    Returns:
        samples : ndarray shape (n, N)
            Binomial counts for each variable and sample

    Notes:
    Uses the waiting time method (multiplicative) similar to Knuth's algorithm.
    Reference:
    https://en.wikipedia.org/wiki/Binomial_distribution#Generating_Binomial-distributed_random_variables
     '''

    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    p = np.atleast_2d(p).reshape(-1,1).astype(float)

    p, _, m, n =  _ppp_(p, 0, m ):

    if N == None or N < 1:
        raise ValueError(" binomial.rnd(p,N): N must be greater than zero")
    
    rng = np.random.default_rng(seed)

    X = np.zeros((n, N), dtype=int)
    
    for trial in range(m):
        # Generate correlated uniforms for this sample
        _, _, U = correlated_rvs(n, N, R)

        # Bernoulli success if U < p, shape (n, N)
        successes = (U < p[:, np.newaxis])
        
        # Accumulate successes over trials
        X += successes.astype(int)
    
    return X
