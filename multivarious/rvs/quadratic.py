import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs


def _ppp_(x, a, b):
    '''
    Validate and preprocess input parameters for consistency and correctness.

    INPUTS:
        x : array_like
            Evaluation points
        a : float
            Minimum of the distribution
        b : float
            Maximum of the distribution (must be > a)
    ''' 

    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    x = np.atleast_1d(x).astype(float)

    a = np.atleast_1d(a).astype(float)
    b = np.atleast_1d(b).astype(float)
    n = len(a)   
    N = len(x)   
        
    # Validate parameter dimensions 
    if not (len(a) == n and len(b) == n):
        raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got a:{len(a)}, b:{len(b)}")

    if np.any(b <= a):
        raise ValueError("quadratic: all b values must be greater than corresponding a values")

    return x, a, b, n, N


def pdf(x, a, b):
    """
    quadratic.pdf
    
    Computes the PDF of the quadratic distribution on interval (a, b).
    
    INPUTS:
        x : array_like
            Evaluation points
        a : float
            Lower bound (a < b)
        b : float
            Upper bound (b > a)
    
    Output:
        f : ndarray
            PDF values at each point in x
    
    Reference:
    Quadratic distribution with PDF: f(x) = 6(x-a)(x-b)/(a-b)^3 for a < x < b
    """

    x, a, b, n, N = _ppp_(x, a, b)

    f = np.zeros((n,N))

    for i in range(n):
        f[i,:] = 6 * (x - a[i]) * (x - b[i]) / (a[i] - b[i])**3
    
        f[i, x >= b[i]] = 0.0 # PDF = 0 for x >= b
        f[i, x <= a[i]] = 0.0 # PDF = 0 for x <= b
    
    if n == 1 and f.shape[0] == 1:
        f = f.flatten()

    return f


def cdf(x, params):
    """
    quadratic.cdf
    
    Computes the CDF of the quadratic distribution on interval (a, b).
    
    INPUTS:
        x : array_like
            Evaluation points
        params: array_like [ a , b ]
        a : float
            Lower bound (a < b)
        b : float
            Upper bound (b > a)
    
    Output:
        F : ndarray
            CDF values at each point in x
    
    Reference:
    CDF formula: F(x) = (a-x)^2 * (a - 3b + 2x) / (a-b)^3 for a <= x <= b
    """
    a, b = params 

    x, a, b, n, N = _ppp_(x, a, b)

    F = np.zeros((n,N))
    
    # CDF formula for a <= x <= b
    for i in range(n):
        F[i,:] = ((a[i] - x)**2 * (a[i]- 3*b[i] + 2*x)) / (a[i] - b[i])**3

        F[i, x >= b[i]] = 1.0 # CDF = 1 for x >= b
        F[i, x <= a[i]] = 0.0 # CDF = 0 for x <= b
    
    if n == 1 and F.shape[0] == 1:
        F = F.flatten()

    return F


def inv(F, a, b):
    """
    quadratic.inv
    
    Computes the inverse CDF (quantile function) by solving cubic equations
    for given probability values.
    
    INPUTS:
        F : array_like or float
            Probability values (0 <= F <= 1)
        a : scalar float
            Lower bound (a < b)
        b : scalar float
            Upper bound (b > a)
    
    Output:
        x : ndarray or float
            Quantile values corresponding to probabilities F
    """

    _, a, b, n, _ = _ppp_(0, a, b)

    F = np.atleast_2d(F).astype(float)
    F = np.clip(F, np.finfo(float).eps, 1 - np.finfo(float).eps)
    N = F.shape[1]    

    x = np.zeros((n,N))
    
    for i in range(n):
        ai = a[i].item()
        bi = b[i].item()
        for j in range(N):
            # Coefficients of cubic equation
            Fij = F[i,j].item()

            if Fij < np.sqrt( np.finfo(float).eps ):
                  x[i,j] = ai

            elif Fij > 1 - np.sqrt( np.finfo(float).eps ):
                  x[i,j] = bi

            else: 
                coeffs = [
                    2,
                    -3 * (ai + bi),
                    6 * ai * bi,
                    ai**3 - 3 * ai**2 * bi - Fij * (ai - bi)**3
                ]

                # Find roots
                roots = np.roots(coeffs)
        
                # Filter for real roots in valid domain
                real_roots = roots[np.abs(roots.imag) < 1e-10].real
                valid_roots = real_roots[(real_roots > ai) & (real_roots < bi)]
        
                if len(valid_roots) != 1:
                    raise ValueError(f"quadratic.inv() Expected 1 root in ({ai}, {bi}), found {len(valid_roots)}")
        
                x[i,j] = valid_roots[0]
    
    if n == 1 and x.shape[0] == 1:
        x = x.flatten()

    return x
    # Return scalar if input was scalar
    #return x[0] if np.isscalar(u) or len(x) == 1 else x


def rnd(a, b, N, R=None, seed=None):
    """
    quadratic.rnd
    
    Generates random samples from the quadratic distribution on interval (a, b).
    
    INPUTS:
        a : float (n,)
            Lower bound (a < b)
        b : float (n,)
            Upper bound (b > a)
        N : int 
            number of observations of n quadratic random variables 
        R : float, (n,n) optional
            correlation matrixx
    
    Output:
        X : ndarray
            Random samples from the quadratic distribution
    """

    _, a, b, n, _ = _ppp_(0, a, b)

    _, _, U = correlated_rvs( R, n, N, seed )

    X = inv(U, a, b)

    return X
