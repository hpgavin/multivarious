# generic pre processing of parameters (ppp) 

def _ppp_(x, a, b, q, p ):
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
    x = np.atleast_1d(x).astype(float)

    a = np.atleast_1d(a).reshape(-1,1).astype(float)
    b = np.atleast_1d(b).reshape(-1,1).astype(float)
    q = np.atleast_1d(q).reshape(-1,1).astype(float)
    p = np.atleast_1d(p).reshape(-1,1).astype(float)
    n = len(a)   
        
    # Validate parameter dimensions 
    if not (len(b) == n and len(q) == n and len(p) == n):
        raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got a:{len(a)}, b:{len(b)}, q:{len(q)}, p:{len(p)}")

   # Validate parameter values 
    if np.any(meanX <= 0):
        raise ValueError("extreme_value_I: meanX must be > 0")
    if np.any(covnX <= 0):
        raise ValueError("extreme_value_I: covnX must be > 0")

    if np.any(b <= a):
        raise ValueError("beta.rnd: all b values must be greater than corresponding a values")
    if np.any(q <= 0):
        raise ValueError("beta.rnd: q must be positive")
    if np.any(p <= 0):
        raise ValueError("beta.rnd: p must be positive")

    return x, a, b, q, p, n

