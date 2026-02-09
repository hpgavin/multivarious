# generic pre processing of parameters (ppp) 

def _ppp_(x, a, b, q, p ):
    '''
    Validate and preprocess input parameters for consistency and correctness.

    Parameters:
        x : array_like
            Evaluation points
        a : float
            Lower bound
        b : float
            Upper bound (must be > a)
        c : float
            Mode (must satisfy a < c < b)
    '''

    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    x = np.atleast_1d(x).astype(float)
    a = np.atleast_1d(a).astype(float)
    b = np.atleast_1d(b).astype(float)
    c = np.atleast_1d(c).astype(float)
    n = len(a)


    # Validate parameter dimensions 
    if not ( len(a) == n and len(b) == n and len(c) == n ):
        raise ValueError(f"a, b, ,c arrays must have the same length. "
                         f"Got a:{len(a)}, b:{len(b)}, c:{len(c)}")
    
    # Validate parameter values
    if not np.any(a <= b):
        raise ValueError(f"triangular: c must be less than b"
                         f"Got: len(c) = {len(c)}, len(b) = {len(b)}")
    if not np.any(c <= b):
        raise ValueError(f"triangular: c must be less than b"
                         f"Got: len(c) = {len(c)}, len(b) = {len(b)}")
    if not np.any(b <= c):
        raise ValueError(f"triangular: b must be less than c"
                         f"Got: len(c) = {len(c)}, len(b) = {len(b)}")
    
    return x, a, b, c, n
    

