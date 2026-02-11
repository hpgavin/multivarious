import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs


def _ppp_(x, a, b, c):
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
    if not np.any(a <= c):
        raise ValueError(f"triangular: a must be less than c"
                         f"Got: len(a) = {len(a)}, len(c) = {len(c)}") 

    return x, a, b, c, n


def pdf(x, a, b, c):
    '''
    triangular.pdf

    Computes the PDF of the triangular distribution on [a, b] with mode c.

    Parameters:
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

    x, a, b, c, n = _ppp_(x, a, b, c)

    pdf = np.zeros_like(x, dtype=float)

    # Left side of peak (pdf piecewise formula)
    left = (a <= x) & (x < c)
    pdf[left] = 2 * (x[left] - a) / ((b - a) * (c - a))

    # Right side of peak (pdf piecewise formula)
    right = (c <= x) & (x <= b)
    pdf[right] = 2 * (b - x[right]) / ((b - a) * (b - c))

    # Outside [a, b] => already zero
    return pdf


def cdf(x, params):
    '''
    triangular.cdf

    Computes the CDF of the triangular distribution on [a, b] with mode c.

    Parameters:
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

    x, a, b, c, n = _ppp_(x, a, b, c)

    cdf = np.zeros_like(x, dtype=float)

    left = (x <= a)
    mid1 = (a < x) & (x < c)
    mid2 = (c <= x) & (x < b)
    right = (x >= b)

    cdf[mid1] = ((x[mid1] - a) ** 2) / ((b - a) * (c - a))
    cdf[mid2] = 1 - ((b - x[mid2]) ** 2) / ((b - a) * (b - c))
    cdf[right] = 1.0

    return cdf


def inv(p, a, b, c):
    '''
    triangular.inv

    Computes the inverse CDF (quantile function) of the triangular distribution
    on [a, b] with mode c.

    Parameters:
        p : array_like
            Probability values (must be in [0, 1])
        a : float
            Lower bound
        b : float
            Upper bound (must be > a)
        c : float
            Mode (must satisfy a < c < b)

    Output:
        x : ndarray
            Quantile values corresponding to probabilities p

    Reference:
    http://en.wikipedia.org/wiki/Triangular_distribution
    '''

    _, a, b, c, n = _ppp_(0, a, b, c)

    p = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
    
    Fc = (c - a) / (b - a)
    x = np.empty_like(p, dtype=float)
    
    # Piecewise conditions
    below = p < Fc
    above = ~below

    # Inverse CDF piecewise formula
    x[below] = a + np.sqrt(p[below] * (b - a) * (c - a))
    x[above] = b - np.sqrt((1 - p[above]) * (b - a) * (b - c))

    return x


def rnd(a, b, c, N, R=None, seed=None ):
    '''
    triangular.rnd

    Generate random samples from the triangular distribution on [a, b] with mode c.

    Parameters:
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

    _, a, b, c, n = _ppp_(0, a, b, c)

    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 

    # Determine number of random variables
    n = len(a)

    # Validate that all parameter arrays have the same length
    if not (len(a) == n and len(b) == n and len(c) == n):
        raise ValueError(f"All parameter arrays must have the same length. "
                        f"Got a:{len(a)}, b:{len(b)}, c:{len(c)}")

    if np.any(b <= a):
        raise ValueError(" triangular.rnd: all b values must be greater than corresponding a values")
    if np.any(c <= a):
        raise ValueError(" triangular.rnd: all c values must be greater than corresponding a values")
    if np.any(b <= c):
        raise ValueError(" triangular.rnd: all b values must be greater than corresponding c values")

    _, _, U = correlated_rvs( R, n, N, seed )

    # Transform each variable to its triangular distribution via inverse CDF
    X = np.zeros((n, N))
    for i in range(n):
        X[i, :] = inv(U[i, :], a[i], b[i], c[i])

    return X
