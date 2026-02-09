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
    a = np.atleast_1d(a).astype(float) 
    b = np.atleast_1d(b).astype(float)
    q = np.atleast_1d(q).astype(float)
    p = np.atleast_1d(p).astype(float)
    n = len(a)   
    
    # Check parameter validity
    if b <= a:
        raise ValueError(f"beta_pdf: a = {a}, b = {b} — a must be less than b")
 

    if not ( (len(t) == n or len(t) == 1) and len(T) == n ):
        raise ValueError(f"T and t arrays must have the same length. "
                         f"Got t:{len(t)}, T:{len(T)}")

    # Validate that all parameter arrays have the same length
    if not (len(b) == n and len(q) == n and len(p) == n):
        raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got a:{len(a)}, b:{len(b)}, q:{len(q)}, p:{len(p)}")
    
    if np.any(b <= a):
        raise ValueError("beta.rnd: all b values must be greater than corresponding a values")
    if np.any(q <= 0):
        raise ValueError("beta.rnd: q must be positive")
    if np.any(p <= 0):
        raise ValueError("beta.rnd: p must be positive")

    return x, a, b, q, p, n

