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
        mask = (a[i] < x) & (x < b[i]) # PDF is nonzero only in (a, b)
        f[i,mask] = 6 * (x[mask] - a[i]) * (x[mask] - b[i]) / (a[i] - b[i])**3
    
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
    
    # CDF = 1 for x >= b
    F[x >= b] = 1.0
    
    # CDF formula for a <= x <= b
    mask = (a <= x) & (x <= b)
    F[mask] = ((a - x[mask])**2 * (a - 3*b + 2*x[mask])) / (a - b)**3
    
    return F


def inv(u, a, b):
    """
    quadratic.inv
    
    Computes the inverse CDF (quantile function) by solving cubic equations
    for given probability values.
    
    INPUTS:
        u : array_like or float
            Probability values (0 <= u <= 1)
        a : scalar float
            Lower bound (a < b)
        b : scalar float
            Upper bound (b > a)
    
    Output:
        x : ndarray or float
            Quantile values corresponding to probabilities u
    """

    _, a, b, _, _ = _ppp_(0, a, b)

    a = a[0] # scalar
    b = b[0] # scalar

    u = np.atleast_1d(u)

    x = np.zeros_like(u, dtype=float)
    
    for j in range(len(u)):
        # Coefficients of cubic equation
        coeffs = [
            2,
            -3 * (a + b),
            6 * a * b,
            a**3 - 3 * a**2 * b - u[j] * (a - b)**3
        ]
        
        # Find roots
        roots = np.roots(coeffs)
        
        # Filter for real roots in valid domain
        real_roots = roots[np.abs(roots.imag) < 1e-10].real
        valid_roots = real_roots[(real_roots > a) & (real_roots < b)]
        
        if len(valid_roots) != 1:
            raise ValueError(f"Expected 1 root in ({a}, {b}), found {len(valid_roots)}")
        
        x[j] = valid_roots[0]
    
    # Return scalar if input was scalar
    return x[0] if np.isscalar(u) or len(x) == 1 else x


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

    X = np.zeros((n, N))
    for i in range(n):
        X[i, :] = inv(U[i,:], a[i], b[i])
    
    if n == 1: 
        X = X.flatten()

    return X
