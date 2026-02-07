import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs

def pdf(x, a, b):
    """
    quadratic.pdf
    
    Computes the PDF of the quadratic distribution on interval (a, b).
    
    Parameters:
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
    x = np.asarray(x, dtype=float)
    f = np.zeros_like(x, dtype=float)
    
    # PDF is nonzero only in (a, b)
    mask = (a < x) & (x < b)
    f[mask] = 6 * (x[mask] - a) * (x[mask] - b) / (a - b)**3
    
    return f


def cdf(x, params):
    """
    quadratic.cdf
    
    Computes the CDF of the quadratic distribution on interval (a, b).
    
    Parameters:
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
    x = np.asarray(x, dtype=float)

    a, b = params 

    F = np.zeros_like(x, dtype=float)
    
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
    
    Parameters:
        u : array_like or float
            Probability values (0 <= u <= 1)
        a : float
            Lower bound (a < b)
        b : float
            Upper bound (b > a)
    
    Output:
        x : ndarray or float
            Quantile values corresponding to probabilities u
    """
    u = np.atleast_1d(u)
    x = np.zeros_like(u, dtype=float)
    
    for i in range(len(u)):
        # Coefficients of cubic equation
        coeffs = [
            2,
            -3 * (a + b),
            6 * a * b,
            a**3 - 3 * a**2 * b - u[i] * (a - b)**3
        ]
        
        # Find roots
        roots = np.roots(coeffs)
        
        # Filter for real roots in valid domain
        real_roots = roots[np.abs(roots.imag) < 1e-10].real
        valid_roots = real_roots[(real_roots > a) & (real_roots < b)]
        
        if len(valid_roots) != 1:
            raise ValueError(f"Expected 1 root in ({a}, {b}), found {len(valid_roots)}")
        
        x[i] = valid_roots[0]
    
    # Return scalar if input was scalar
    return x[0] if np.isscalar(u) or len(x) == 1 else x


def rnd(a, b, N, R=None, seed=None):
    """
    quadratic.rnd
    
    Generates random samples from the quadratic distribution on interval (a, b).
    
    Parameters:
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

    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    a = np.atleast_1d(a).reshape(-1,1).astype(float)#[:,0]
    b = np.atleast_1d(b).reshape(-1,1).astype(float)#[:,0]

    # Determine number of random variables
    n = len(a)

    # Validate that all parameter arrays have the same length
    if not (len(a) == n and len(b) == n):
        raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got a:{len(a)}, b:{len(b)}")

    if np.any(b <= a):
        raise ValueError(" quadratic.rnd: all b values must be greater than corresponding a values")

    _, _, U = correlated_rvs( R, n, N, seed )

    X = np.zeros((n, N))
    for i in range(N):
        X[i, :] = inv(U[i,:], a[i], b[i])
    
    if n == 1: 
        X = X.flatten()

    return X
