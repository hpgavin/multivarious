import numpy as np

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


def cdf(x, a, b, c):
    '''
    triangular.cdf

    Computes the CDF of the triangular distribution on [a, b] with mode c.

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
        F : ndarray
            CDF values at each point in x

    Reference
    ----------
    http://en.wikipedia.org/wiki/Triangular_distribution
    '''

    x = np.asarray(x)
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

def rnd(a, b, c, size=(1,), seed=None):
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
        x : ndarray
            Array of shape `size` containing triangular random samples

    Reference:
    http://en.wikipedia.org/wiki/Triangular_distribution
    '''
    if isinstance(seed, (int, type(None))):
        rng = np.random.default_rng(seed)
    else:
        rng = seed  # assume user passed Generator

    u = rng.random(size)
    return inv(u, a, b, c)
