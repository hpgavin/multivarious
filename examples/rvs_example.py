#! /usr/bin/env -S python3 -i

import numpy as np
import matplotlib.pyplot as plt

from multivarious.rvs import *  # import all the rvs distributions
from multivarious.utl import format_plot

# --- dimensions ---
n = 3        # number of variables
N = 1612     # number of observations

print('--------------------------------------------------')
print('original mean, std.dev. and correlation of  X')
print('used to simulate a sample of correlated X')
print('--------------------------------------------------')

meanX = np.arange(1, n+1).reshape(-1,1) * 2.5 / n
sdvnX = np.arange(n, 0, -1).reshape(-1,1) * 1 / n

covnX = sdvnX / meanX
mednX = meanX / np.sqrt(1 + covnX**2)

a = meanX + 1.0
b = meanX + 2.0
c =  0.3*a + 0.7*b  
q = np.arange(n+1, 1, -1)
p = np.arange(2, n+2)
k = q 
m = meanX
s = sdvnX
T = meanX
t = 1
mb = [ 10,   5,  12 ]   # Bernoulli attempts (integers)
pb = [  0.3, 0.4, 0.7 ] # Bernoulli probabilities 
nb = np.arange(np.max(mb)+1) # Bernoulli successes (integers) 

print("meanX:\n", meanX)
print("sdvnX:\n", sdvnX)

x = np.linspace(0.1*np.min(meanX), 3*np.max(meanX), 100 )

# Correlation matrix R
R = np.array([[1,   -0.5, -0.8],
              [-0.5, 1,    0.9],
              [-0.8, 0.9,  1]])

print("Correlation matrix R:\n", R)

# ----------------------------------------------------------
# Select one of the available distributions ... 
#  'beta' 'binomial' 'chi2'
#  'exponential' 'extreme_value_I' 'extreme_value_II'
#  'gamma' 'gev' 'laplace' 'lognormal' 'normal' 'poisson'
#  'quadratic' 'rayleigh' 'students_t' 'triangular' 'uniform'

distribution = 'uniform'

# --- PDFs , CDFs and Samples of correlated random variables ---

if distribution == 'beta':
    x = np.linspace( np.min(a)-0.1*np.max(b-a), np.max(b)+0.1*np.max(b-a), N)
    fx = beta.pdf( x, a, b, q, p )
    Fx = beta.cdf( x, [ a, b, q, p ] )
    xi = beta.inv( Fx,  a, b, q, p   )
    X  = beta.rnd( a, b, q, p, N, R )

if distribution == 'binomial':
    fx = binomial.pmf( nb, mb, pb )
    Fx = binomial.cdf( nb, [ mb, pb ] )
    X  = binomial.rnd( mb, pb, N, R )
    x = nb
    xi = x # there is no binomial.inv

if distribution == 'chi2':
    if n > 1:
        k = np.array([ 2, 5, 10 ])
    x = np.linspace( 0.1, 3*np.max(k), 250 )
    fx = chi2.pdf( x, k )
    Fx = chi2.cdf( x, k )
    xi = chi2.inv( Fx, k  )
    X  = chi2.rnd( k, N, R )

if distribution == 'exponential':
    fx = exponential.pdf( x, meanX )
    Fx = exponential.cdf( x, [ meanX ] )
    xi = exponential.inv( Fx, meanX   )
    X = exponential.rnd( meanX, N, R )

if distribution == 'extreme_value_I':
    fx = extreme_value_I.pdf( x, meanX, covnX )
    Fx = extreme_value_I.cdf( x, [ meanX, covnX ] )
    xi = extreme_value_I.inv( Fx, meanX , covnX  )
    X  = extreme_value_I.rnd( meanX, covnX, N, R )

if distribution == 'extreme_value_II':
    fx = extreme_value_II.pdf( x, m, s, k )
    Fx = extreme_value_II.cdf( x, [ m, s, k ] )
    xi = extreme_value_II.inv( Fx,  m, s, k   )
    X  = extreme_value_II.rnd( m, s, k, N, R )

if distribution == 'gamma':
    fx = gamma.pdf( x, meanX, covnX )
    Fx = gamma.cdf( x, [ meanX, covnX ] )
    xi = gamma.inv( Fx,  meanX, convX   )
    X = gamma.rnd( meanX, covnX, N, R )

if distribution == 'gev':
    fx = gev.pdf( x, m, s, k )
    Fx = gev.cdf( x, [ m, s, k ] )  
    xi = gev.inv( Fx,  m, s, k   )
    X  = gev.rnd( m, s, k, N, R )  

if distribution == 'laplace':
    fx = laplace.pdf( x, meanX, sdvnX )
    Fx = laplace.cdf( x, [ meanX, sdvnX ] )
    xi = laplace.inv( Fx,  meanX, sdvnX   )
    X  = laplace.rnd( meanX, sdvnX, N, R )

if distribution == 'lognormal':
    fx = lognormal.pdf( x, mednX, covnX )
    Fx = lognormal.cdf( x, [ mednX, covnX ] )
    xi = lognormal.inv( Fx,  mednX, covnX   )
    X  = lognormal.rnd( mednX, covnX, N, R )

if distribution == 'normal':
    fx = normal.pdf( x, meanX, sdvnX )
    Fx = normal.cdf( x, [ meanX, sdvnX ] )
    xi = normal.inv( Fx,  meanX, sdvnX   )
    X = normal.rnd( meanX, sdvnX, N, R )

if distribution == 'poisson':
    fx = poisson.pmf(nb, t, T ) 
    Fx = poisson.cdf(nb, [ t, T ] ) 
    X  = poisson.rnd(t, T, N, R ) 
    x = nb
    xi = np # no .inv for poisson

if distribution == 'quadratic':
    x  = np.linspace( np.min(a), np.max(b), 250 )
    fx = quadratic.pdf( x, a, b ) 
    Fx = quadratic.cdf( x, [ a, b ] ) 
    xi = quadratic.inv( Fx,  a, b   )
    X  = quadratic.rnd( a, b, N, R ) 

if distribution == 'rayleigh':
    fx = rayleigh.pdf( x, meanX )
    Fx = rayleigh.cdf( x, [ meanX ] )
    xi = rayleigh.inv( Fx,  meanX   )
    X  = rayleigh.rnd( meanX, N, R )

if distribution == 'students_t':
    x  = np.linspace( -10, 10, 250 )
    fx = students_t.pdf( x, k )
    Fx = students_t.cdf( x, [ k ] )
    xi = students_t.inv( Fx,  k )
    X  = students_t.rnd( k, N, R )

if distribution == 'triangular':
    x  = np.linspace( np.min(a), np.max(b), 250 )
    fx = triangular.pdf( x, a, b, c )
    Fx = triangular.cdf( x, [ a, b, c ] )
    xi = triangular.inv( Fx,  a, b, c   )
    X  = triangular.rnd( a, b, c, N, R )

if distribution == 'uniform':
    fx = uniform.pdf( x, a, b )
    Fx = uniform.cdf( x, [ a, b ] )
    xi = uniform.inv(Fx,   a, b   )
    X  = uniform.rnd( a, b, N, R )


# log transform exponential-like distributions
"""
if distribution in [ 'chi2',  'exponential',  'extreme_value_I',  'extreme_value_II',  'gamma',  'lognormal',  'rayleigh']:
    X = np.log10(X)  
"""

print('X', X)

print('distribution: ', distribution)


# Decorrelate the sample 
if n > 1: 
    mean_X_sample = np.mean(X, axis=1, keepdims=True)
    covr_X_sample = np.cov(X)

    # Eigen decomposition
    eVal, eVec = np.linalg.eigh(covr_X_sample)
    invT = np.linalg.inv(np.sqrt(np.diag(eVal))) @ eVec.T
    Z = invT @ (X - mean_X_sample)

    print('------------------------------------------------------------------')
    print(f'estimated mean, std.dev. and correlation of log10 X  ...  sample size = {N}')
    print('estimated from the sample of log-normal correlated X')
    print('------------------------------------------------------------------')

    print("Estimated mean of X:\n", mean_X_sample.flatten())
    print("Estimated sdvn of X:\n", np.sqrt(np.diag(covr_X_sample)))
    print("Estimated correlation of X:\n", np.corrcoef(X))
    print("Estimated correlation of Z:\n", np.corrcoef(Z))
else:
    Z = X


# Plots ===================================================

format_plot( font_size=16, line_width=2, marker_size=4 )
plt.ion() # interactive plot mode: on

if n == 1:
    X  = np.atleast_2d(X)
    fx = np.atleast_2d(fx)
    Fx = np.atleast_2d(Fx)
    xi = np.atleast_2d(xi)

# Figure 1: scatter plot of correlated random variables ------
plt.figure(1, figsize=(10, 10))
kk = 0
for ii in range(n):
    for jj in range(n):
        kk += 1
        plt.subplot(n, n, kk)
        if ii == jj:
            # empirical PDF
            plt.hist(X[ii, :], bins=20, color='royalblue', edgecolor='black',density=True)
            # theoreticap PDF
            plt.plot(x, fx[ii,:].T, '-k')  
        if ii > jj:
            plt.plot(X[jj, :], X[ii, :], 'o', color='royalblue',markersize=3) 
        if ii < jj:
            plt.plot(Z[jj, :], Z[ii, :], 'o', color='black', markersize=3)
        if jj == 0:
            plt.ylabel(rf'$X_{ii+1}$')
        if ii == n-1:
            plt.xlabel(rf'$X_{jj+1}$')
plt.tight_layout()


# Figure 2: CDF F_X(x) and empirical CDF  ---------------------------------
ECDF = (np.arange(N)+1)/(N+1)  
plt.figure(2, figsize=(8, 6))
if distribution in [ 'binomial', 'poisson' ]:
    plt.plot(nb,fx.T) # binomial and poisson
    plt.xlabel(r'$n$') # binomial and poisson
    plt.ylabel(r'PMF $p_N(n)$') # binomial and poisson
else:
    for i in range(n):
        plt.subplot(1, n, i+1)
        #   empirical  CDF ...
        plt.step(np.sort(X[i,:]),ECDF, color='royalblue', linewidth=6)
        # theoretical  CDF ...
        plt.plot(x,Fx[i,:], '-k')                         
        plt.xlabel(rf'$X_{i+1}$')
        if i == 0:
            plt.ylabel(r'CDF $F_X(x)$') 
plt.tight_layout()

# Figure3: CDF and inverse CDF
plt.figure(3)
plt.plot(x,Fx.T,'-', linewidth=5)
for i in range(n): 
    plt.plot(xi[i,:],Fx[i,:],'--', color='yellow')
plt.xlabel(r'$x$')
plt.ylabel(r'$F_X(x)$')

"""
plt.figure(4)
plt.plot(x,xi.T,'o', markersize=5)
plt.xlabel(r'$x$')
plt.ylabel(r' inv $(F_X(x))$')


plt.figure(5)
plt.loglog(Fx,np.abs(xi-x),'o', markersize=5)
plt.xlabel(r'$F_X(x)$')
plt.ylabel(r'| inv $(F_X(x)) - x$ |')
"""

