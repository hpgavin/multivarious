import numpy as np
from scipy.special import beta as beta_func  # scipy's beta function
from scipy.special import betainc
from scipy.special import betaincinv  # Inverse of the regularized incomplete beta function


def pdf(x, a, b, q, p):
    '''
    beta.pdf

    Computes the Probability Density Function (PDF) of the Beta distribution
    with lower bound a, upper bound b, and shape parameters q and p.

    INPUTS:
      x = evaluation points
      a   = minimum of the distribution
      b = maximum of the distribution (must be > a)
      q = first shape parameter
      p = second shape parameter
 
    OUTPUT:
      f = PDF evaluated at x
    '''
    x = np.asarray(x, dtype=float)
    
    # Initialize PDF output as zeros
    f = np.zeros_like(x)

    # Only compute for values within [a, b]
    valid = (x >= a) & (x <= b)
    
    # Beta PDF formula (with change of variable from [0,1] to [a,b])
    numerator = (x[valid] - a) ** (q - 1) * (b - x[valid]) ** (p - 1)
    denominator = beta_func(q, p) * (b - a) ** (q + p - 1)
    f[valid] = numerator / denominator

    return f


def cdf(x, abqp):
    '''
    beta.cdf

    Computes the Cumulative Distribution Function (CDF) of the beta distribution
    with parameters passed as a single vector: abqp = [a, b, q, p]

    Formula:
      F(x) = I_{(x - a) / (b - a)} (q, p)
    where I is the regularized incomplete beta function.
    '''
    x = np.asarray(x, dtype=float)

    # Unpack the parameters
    a, b, q, p = abqp

    # Check parameter validity
    if b < a:
        raise ValueError(f"beta_cdf: a = {a}, b = {b} — a must be less than b")

    # Compute z = (x - a) / (b - a), clipped to [0, 1]
    z = (x - a) / (b - a)
    z[z < 0] = 0
    z[z > 1] = 1

    # Evaluate the regularized incomplete beta function
    F = betainc(q, p, z)

    return F


    '''------------------------------
# CDF: beta_inv
    '''------------------------------


def inv(F, a, b, q, p):
    '''
    beta.inv

    Compute the inverse CDF (quantile function) of the beta distribution,
    given parameters a (min), b (max), q and p (shape parameters).

    Parameters:
        F : array-like
            Non-exceedance probability values (between 0 and 1).
        a : float
            Lower bound of the distribution (min).
        b : float
            Upper bound of the distribution (max).
        q : float
            First shape parameter (analogous to α).
        p : float
            Second shape parameter (analogous to β).

    Returns:
        x : array-like
            Quantile values corresponding to input probabilities F.
    '''

    # Check that a < b (valid interval)
    if b < a:
        raise ValueError(f'beta_inv: a = {a}, b = {b} → a must be less than b')

    # Compute inverse of regularized incomplete beta function
    z = betaincinv(q, p, F)  # Returns value z such that betainc(q, p, z) = F

    # Rescale from [0, 1] to [a, b]
    x = a + z * (b - a)

    return x


    '''------------------------------
# CDF: beta_rnd
    '''------------------------------


def rnd(a, b, q, p, M, N):
    '''
    beta.rnd

    Generate a sample matrix from a Beta distribution defined by:
    min = a, max = b, shape parameters q and p.

    Parameters:
    a, b : float
        Minimum and maximum values of the support (a < b).
    q, p : float
        Shape parameters of the Beta distribution.
    M, N : int
        Number of rows and columns of the output sample matrix.

    Returns:
    x : ndarray
        An MxN matrix of samples from the Beta(a, b, q, p) distribution.
        
        
    Method:
    1. Generate samples X from Gamma(q, 1) and Y from Gamma(p, 1).
    2. Compute Z = X / (X + Y) which follows a Beta(q, p) distribution on [0, 1].
    3. Rescale Z to the interval [a, b] using x = a + (b - a) *
    
    statistically, this makes sense becasue if X ~ Gamma(q, 1) and Y ~ Gamma(p, 1),
    then Z = X / (X + Y) follows a Beta(q, p) distribution on [0, 1].
    '''

    if b < a:
        raise ValueError(f"beta_rnd: a = {a}, b = {b}; a must be less than b.")

    # Step 1: Generate samples from Gamma(q, 1) and Gamma(p, 1)
    X = np.random.gamma(shape=q, scale=1.0, size=(M, N))
    Y = np.random.gamma(shape=p, scale=1.0, size=(M, N))

    # Step 2: Construct Beta samples from X / (X + Y)
    z = X / (X + Y)

    # Step 3: Rescale Beta samples to interval [a, b]
    x = a + (b - a) * z

    return x

