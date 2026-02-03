import numpy as np

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


def cdf(x, a, b):
    """
    quadratic.cdf
    
    Computes the CDF of the quadratic distribution on interval (a, b).
    
    Parameters:
        x : array_like
            Evaluation points
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


def rnd(a, b, r, c=None, seed=None):
    """
    quadratic.rnd
    
    Generates random samples from the quadratic distribution on interval (a, b).
    
    Parameters:
        a : float
            Lower bound (a < b)
        b : float
            Upper bound (b > a)
        r : int or ndarray
            If int: number of rows; if ndarray: matrix of uniform(0,1) values
        c : int, optional
            Number of columns (used only if r is int)
        seed : int or np.random.Generator, optional
            Random seed for reproducibility
    
    Output:
        X : ndarray
            Random samples from the quadratic distribution
    """
    # Setup random number generator
    if isinstance(seed, (int, type(None))):
        rng = np.random.default_rng(seed)
    else:
        rng = seed
    
    # Case 1: r is a matrix of uniform random values
    if c is None and isinstance(r, np.ndarray):
        u = r
        r_dim, c_dim = u.shape
    # Case 2: Generate uniform samples with shape (r, c)
    elif c is not None and isinstance(r, int):
        u = rng.random((r, c))
        r_dim, c_dim = r, c
    else:
        raise ValueError("quadratic_rnd: Either provide a matrix (r) or integers (r, c)")
    
    # Solve cubic for each u value
    X = np.zeros((r_dim, c_dim))
    for i in range(r_dim):
        for j in range(c_dim):
            coeffs = [
                2,
                -3 * (a + b),
                6 * a * b,
                a**3 - 3 * a**2 * b - u[i, j] * (a - b)**3
            ]
            roots = np.roots(coeffs)
            real_roots = roots[np.abs(roots.imag) < 1e-10].real
            valid_roots = real_roots[(real_roots > a) & (real_roots < b)]
            
            if len(valid_roots) != 1:
                raise ValueError(f"Expected 1 root in ({a}, {b}), found {len(valid_roots)}")
            
            X[i, j] = valid_roots[0]
    
    if r_dim == 1: 
        X = X.flatten()

    return X
