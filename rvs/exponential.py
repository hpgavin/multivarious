# exponential distribution
import numpy as np

def pdf(x, muX):
    '''
    exponential.pdf

    Computes the Probability Density Function (PDF) of the exponential distribution.

    INPUTS:
      x    = evaluation points (must be x >= 0)
      muX  = mean of the exponential distribution
 
    OUTPUT:
      f    = PDF evaluated at x
 
    FORMULA:
      f(x) = (1/muX) * exp(-x / muX), for x >= 0
    '''
    x = np.asarray(x, dtype=float)

    # Prevent negative or zero values (log not defined)
    x = np.where(x < 0, 0.01, x)

    f = np.exp(-x / muX) / muX
    return f


def cdf(x, muX):
    '''
    exponential.cdf

    Computes the Cumulative Distribution Function (CDF) of the exponential.

    INPUTS:
      x    = values at which to evaluate the CDF (x >= 0)
      muX  = mean of the exponential distribution
 
    OUTPUT:
      F    = CDF values at each x
 
    FORMULA:
      F(x) = 1 - exp(-x / muX), for x >= 0
    '''
    x = np.asarray(x, dtype=float)

    # Prevent issues with x <= 0
    x = np.where(x <= 0, 0.01, x)

    F = 1.0 - np.exp(-x / muX)
    return F


def inv(P, muX):
    '''
    exponential.inv

    Computes the inverse CDF (quantile function) of the exponential distribution.

    INPUTS:
      P    = probability values (0 <= P <= 1)
      muX  = mean of the exponential distribution

    OUTPUT:
      X    = quantiles corresponding to P

    FORMULA:
      X = -muX * log(1 - P)
    '''
    P = np.asarray(P, dtype=float)

    # Clip invalid P values just like in MATLAB
    P = np.where(P < 0, 0.0, P)
    P = np.where(P > 1, 1.0, P)

    X = -muX * np.log(1 - P)
    return X


    '''
# RND: exp_rnd
#
# Generate random samples from an exponential distribution with mean muX.
#
# INPUTS:
#   muX  = mean of the exponential distribution
#   r    = number of rows in output sample
#   c    = number of columns in output sample
#
# OUTPUT:
#   x    = random samples shaped (r, c)
#
# METHOD:
#   Use inverse CDF method: x = -muX * log(U), where U ~ Uniform(0,1)
    '''


def rnd(muX, r, c):
    """
    Generate random samples from an exponential distribution with mean muX.

    Parameters:
        muX : mean of the exponential distribution
        r, c : dimensions of the output matrix (rows × columns)

    Returns:
        x : (r × c) array of random values drawn from Exp(muX)
    """

    # Step 1: Generate uniform random numbers in [0, 1] — shape (r × c)
    u = np.random.uniform(0, 1, size=(r, c))

    # Step 2: Apply the inverse CDF of the exponential distribution:
    #         X = -muX * log(U), where U ~ Uniform(0,1)
    #         This transforms uniform randomness into exponential randomness
    x = -muX * np.log(u)

    # Return the generated exponential random values
    return x


