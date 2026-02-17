import numpy as np
from scipy.special import factorial

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
    '''
    binomial.pmf

    Computes the Probability Mass Function (PMF) of the Binomial distribution.

    INPUTS:
        n : number of successful outcomes (N,)
        m : int scalar or array_like (n,)
            number of attempts 
        p : float scalar or array_like (n,)
            event probabilities of a single attempt for each variable

    Output:
        p : ndarray (n,N)
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

    n, m, p, nb, Nb =  _ppp_( n, m, p )

    P = np.zeros((nb,Nb))

    for i in range(nb): 
        ni = np.arange(m[i]) # can not have more successes than attempts!
        facts = factorial(m[i]) / (factorial(ni) * factorial(m[i]-ni))
        facts = facts.flatten()
        P[i,ni] = facts * ( p[i]**ni * (1-p[i])**(m[i]-ni) ).flatten()
   
    if nb == 1:
         P = P.flatten();

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

    return X
