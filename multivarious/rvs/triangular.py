import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs


def _ppp_(x, a, b, c):
    '''
    Validate and preprocess input parameters for consistency and correctness.

    INPUTS:
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
    N = len(x)

    # Validate parameter dimensions 
    if not ( len(a) == n and len(b) == n and len(c) == n ):
        raise ValueError(f"a, b, ,c arrays must have the same length. "
                         f"Got a:{len(a)}, b:{len(b)}, c:{len(c)}")

    # Validate parameter values
    if not np.all(a <= b):
        raise ValueError("triangular: all a values must be less than or equal to b")
    if not np.all(a <= c):
        raise ValueError("triangular: all a values must be less than or equal to c")
    if not np.all(c <= b):
        raise ValueError("triangular: all c values must be less than or equal to b") 

    return x, a, b, c, n, N


def pdf(x, a, b, c):
    '''
    triangular.pdf

    Computes the PDF of the triangular distribution on [a, b] with mode c.

    INPUTS:
        x : array_like
            Evaluation points
        a : float
            Lower bound
        b : float
            Upper bound (must be > a)
        c : float
            Mode (must satisfy a < c < b)

    Output:
        f : ndarray
            PDF values at each point in x
            
    Reference:   http://en.wikipedia.org/wiki/Triangular_distribution
    '''

    x, a, b, c, n, N = _ppp_(x, a, b, c)

    f = np.zeros((n,N))

    for i in range(n):
        # Left side of peak (pdf piecewise formula)
        left = (a[i] <= x) & (x < c[i])
        f[i,left] = 2 * (x[left] - a[i]) / ((b[i] - a[i]) * (c[i] - a[i]))

        # Right side of peak (pdf piecewise formula)
        right = (c[i] <= x) & (x <= b[i])
        f[i,right] = 2 * (b[i] - x[right]) / ((b[i] - a[i]) * (b[i] - c[i]))

    if n == 1 and f.shape[0] == 1:
        f = f.flatten()

    return f  # Outside [a, b] => already zero


def cdf(x, params):
    '''
    triangular.cdf

    Computes the CDF of the triangular distribution on [a, b] with mode c.

    INPUTS:
        x : array_like
            Evaluation points
        params : array_like [ a, b, c ]
        a : float
            Lower bound
        b : float
            Upper bound (must be > a)
        c : float
            Mode (must satisfy a < c < b)

    Output:
        F : ndarray
            CDF values at each point in x

    Reference
    ----------
    http://en.wikipedia.org/wiki/Triangular_distribution
    '''

    a, b, c = params

    x, a, b, c, n, N = _ppp_(x, a, b, c)

    F = np.zeros((n,N))

    for i in range(n): 
        left = (x <= a[i])
        mid1 = (a[i] < x) & (x < c[i])
        mid2 = (c[i] <= x) & (x < b[i])
        right = (x >= b[i])

        F[i,mid1] = ((x[mid1] - a[i]) ** 2) / ((b[i] - a[i]) * (c[i] - a[i]))
        F[i,mid2] = 1 - ((b[i] - x[mid2]) ** 2) / ((b[i] - a[i]) * (b[i] - c[i]))
        F[i,right] = 1.0

    if n == 1 and F.shape[0] == 1:
        F = F.flatten()

    return F


def inv(F, a, b, c):
    '''
    triangular.inv

    Computes the inverse CDF (quantile function) of the triangular distribution
    on [a, b] with mode c.

    INPUTS:
        F : array_like
            Probability values (must be in [0, 1])
        a : float
            Lower bound
        b : float
            Upper bound (must be > a)
        c : float
            Mode (must satisfy a < c < b)

    Output:
        x : ndarray
            Quantile values corresponding to probabilities F

    Reference:
    http://en.wikipedia.org/wiki/Triangular_distribution
    '''

    _, a, b, c, n, _ = _ppp_(F, a, b, c)

    F = np.atleast_2d(F).astype(float)
    F = np.clip(F, np.finfo(float).eps, 1 - np.finfo(float).eps)
    N = F.shape[1]    
    
    Fc = (c - a) / (b - a)

    x = np.zeros((n,N))

    for i in range(n):
        # Piecewise conditions
        below = F[i,:] < Fc[i]
        above = ~below

        # Inverse CDF piecewise formula
        x[i,below] = a[i] + np.sqrt(F[i,below] * (b[i] - a[i]) * (c[i] - a[i]))
        x[i,above] = b[i] - np.sqrt((1 - F[i,above]) * (b[i] - a[i]) * (b[i] - c[i]))

    if n == 1 and x.shape[0] == 1:
        x = x.flatten()

    return x


def rnd(a, b, c, N, R=None, seed=None ):
    '''
    triangular.rnd

    Generate random samples from the triangular distribution on [a, b] with mode c.

    INPUTS:
        a : float
            Lower bound
        b : float
            Upper bound (must be > a)
        c : float
            Mode of X (must satisfy a < c < b)
     size : tuple, optional
            Output shape (e.g., (r, c)); default is (1,)
     seed : int or np.random.Generator, optional
            Random seed or existing Generator for reproducibility

    Output:
        X : ndarray
            Array of shape `size` containing triangular random samples

    Reference:
    http://en.wikipedia.org/wiki/Triangular_distribution
    '''

    _, a, b, c, n, _ = _ppp_(0, a, b, c)

    _, _, U = correlated_rvs( R, n, N, seed )

    # Transform each variable to its triangular distribution via inverse CDF
    X = inv( U, a, b, c )

    return X
