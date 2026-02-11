import numpy as np
from scipy.special import beta as beta_func, betaincinv

from multivarious.utl.correlated_rvs import correlated_rvs


# generic pre processing of parameters (ppp) 

def _ppp_(t, k):
    '''
    Validate and preprocess input parameters for consistency and correctness.

    Parameters:
        x : array_like
            Evaluation points
        a : float
            Minimum of the distribution
        b : float
            Maximum of the distribution (must be > a)
        q : float
            First shape parameter
        p : float
            Second shape parameter
    ''' 

    # Convert inputs to arrays
    # Python does not implicitly handle scalars as arrays. 
    t = np.atleast_1d(t).astype(float)
    k = np.atleast_2d(k).reshape(-1,1).astype(int)
    n = len(k)   
        
   # Validate parameter values 
    if np.any(k <= 0):
        raise ValueError("students_t: k must be > 0")

    return t, k, n


def pdf(t, k):
    '''
    students_t.pdf

    Computes the PDF of the Student's t-distribution with k degrees of freedom.

    Parameters:
        t : array_like
            Evaluation points
        k : int or float
            Degrees of freedom (must be > 0)
    Output:
        f : ndarray
            PDF values at each point in t

    Reference:
    https://en.wikipedia.org/wiki/Student%27s_t-distribution
    '''
    
    t, k, n = _ppp_(t, k)

    # Compute the PDF using the known closed-form
    f = (np.exp(-(k + 1) * np.log(1 + (t ** 2) / k) / 2)) / (np.sqrt(k) * beta_func(k / 2, 0.5))

    return f


def cdf(t, k):
    '''
    students_t.cdf

    Computes the CDF of the Student's t-distribution with k degrees of freedom.
    Handles k = 1 and k = 2 analytically, and uses recurrence relations
    for integer k > 2 to match the MATLAB implementation.

    Parameters:
        t : array_like
            Evaluation points
        k : int or float
            Degrees of freedom (must be > 0)

    Output:
        F : ndarray
            CDF values at each point in t

    Reference:
    https://en.wikipedia.org/wiki/Student%27s_t-distribution
    '''
    t, k, n = _ppp_(t, k)

    if k == 1:
        # Cauchy distribution
        return 0.5 + np.arctan(t) / np.pi

    elif k == 2:
        return 0.5 + t / (2 * np.sqrt(2 + t**2))

    else:
        ts = t / np.sqrt(k)
        ttf = 1 / (1 + ts**2)

        u = np.ones_like(ts, dtype=float)
        s = np.ones_like(ts, dtype=float)

        if k % 2 == 1:  # odd degrees of freedom
            m = (k - 1) // 2
            for ii in range(2, m + 1):
                u = u * (1 - 1 / (2 * ii - 1)) * ttf
                s = s + u
            return 0.5 + (ts * ttf * s + np.arctan(ts)) / np.pi

        else:  # even degrees of freedom
            m = k // 2
            for ii in range(1, m):
                u = u * (1 - 1 / (2 * ii)) * ttf
                s = s + u
            return 0.5 + (ts * np.sqrt(ttf) * s) / 2.0


def inv(p, k):
    '''
    students_t.inv

    Computes the inverse CDF (quantile function) of the Student's t-distribution
    with k degrees of freedom, using the inverse incomplete beta function.

    Input:
        p : array_like
            Probability values (must be in [0, 1])
        k : int or float
            Degrees of freedom (must be > 0)

    Output:
        x : ndarray
            Quantile values corresponding to probabilities p

    Reference:
    https://en.wikipedia.org/wiki/Student%27s_t-distribution
    '''
    _, k, _ = _ppp_(0, k)
    
    p = np.asarray(p)

    # Compute the inverse CDF using the relationship with the incomplete beta function
    # betaincinv(a, b, y) finds x such that betainc(a, b, x) = y
    z = betaincinv(k / 2.0, 0.5, 2 * np.minimum(p, 1 - p))
    
    # Convert from beta quantile to t quantile
    x = np.sign(p - 0.5) * np.sqrt(k * (1 / z - 1))

    return x


def rnd(k, N, R=None, seed=None ): 
    '''
    students_t.rnd

    Generate random samples from the Student's t-distribution with k degrees of freedom.

    Parameters:
        k : int or float (n,)
            Degrees of freedom (must be > 0)
        N : number of samples of n student_t random variables 
            Output shape (e.g., (r, c)); default is (1,)
        R : float (n,n)
            correlation matrix

    Output:
        X : ndarray
            Array of shape `size` containing t-distributed random samples

    Reference:
    https://en.wikipedia.org/wiki/Student%27s_t-distribution
    '''

    _, k, n = _ppp_(0, k)

    _, _, U = correlated_rvs( R, n, N, seed )

    X = np.zeros((n,N))
    for i in range(n):
        X[i, :] = inv(U[i, ], k[i]) 

    return X
