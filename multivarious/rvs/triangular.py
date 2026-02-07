import numpy as np

from multivarious.utl.correlated_rvs import correlated_rvs

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
    x = np.asarray(x)
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

    x = np.asarray(x)

    a, b, c = params

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
    p = np.asarray(p)
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

    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    a = np.atleast_2d(a).reshape(-1,1).astype(float)
    b = np.atleast_2d(b).reshape(-1,1).astype(float)
    c = np.atleast_2d(c).reshape(-1,1).astype(float)

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
