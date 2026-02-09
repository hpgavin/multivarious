#! /usr/bin/env -S python3 -i

import numpy as np
import matplotlib.pyplot as plt

from multivarious.rvs import *           # import all the distributions

from multivarious.utl import format_plot

# --- Parameters ---
n = 3       # number of variables
N = 1050    # number of observations

print('--------------------------------------------------')
print('original mean, std.dev. and correlation of  X')
print('used to simulate a sample of correlated X')
print('--------------------------------------------------')

meanX = np.arange(1, n+1).reshape(-1,1) * 2.5 / n
stdvX = np.arange(n, 0, -1).reshape(-1,1) * 1 / n

covnX = stdvX / meanX
mednX = meanX / np.sqrt(1 + covnX**2)

a = meanX - 0.3*meanX
b = meanX + 0.3*meanX
c =  0.3*a + 0.7*b  
q = np.arange(n+1, 1, -1)
p = np.arange(2, n+2)
k = q 
m = meanX
s = stdvX
T = meanX

print("meanX:\n", meanX)
print("stdvX:\n", stdvX)

# Correlation matrix R
R = np.array([[1,   -0.5, -0.8],
              [-0.5, 1,    0.9],
              [-0.8, 0.9,  1]])

print("Correlation matrix R:\n", R)

# --- Generate sample of correlated random variables ---

#X = beta.rnd( a, b, q, p, N, R )
#X = chi2.rnd( k, N, R )
#X = exponential.rnd( meanX, N, R )
#X = extreme_value_I.rnd( meanX, covnX, N, R )
#X = extreme_value_II.rnd( m, s, k, N, R )
#X = gamma.rnd( meanX, covnX, N, R )
#X = gev.rnd( m, s, k, N, R ) ## error 
#X = laplace.rnd( meanX, stdvX, N, R )
X = lognormal.rnd( mednX, covnX, N, R )
#X = normal.rnd( meanX, stdvX, N, R )
#X = poisson.rnd( T, N, R )
#X = quadratic.rnd( a, b, N, R ) ## trouble
#X = rayleigh.rnd( meanX, N, R )
#X = students_t.rnd( k, N, R )
#X = triangular.rnd( a, b, c, N, R )
#X = uniform.rnd( a, b, N, R )

#X = np.log10(X)  # for lognormal, chi2 rv's 

print('X', X)

# Decorrelate the sample 
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

#print("Estimated mean log10(X):\n", mean_log10X_sample.flatten())
#print("Estimated std.dev log10(X):\n", np.sqrt(np.diag(covr_log10X_sample)))
#print("Estimated correlation log10(X):\n", np.corrcoef(log10X))

# --- Plots ---

format_plot( font_size=16, line_width=2, marker_size=4 )
plt.ion() # interactive plot mode: on

# Figure 1: histogram of correlated random variables
plt.figure(1, figsize=(8, 6))
for jj in range(n):
    plt.subplot(n, 1, jj+1)
    plt.hist(X[jj, :], bins=20, color='skyblue', edgecolor='black')
    plt.xlabel(rf'$X_{jj+1}$')
    if jj == 0:
        plt.title('Histogram of X')
plt.tight_layout()

# Figure 2: scatter plot of correlated random variables
plt.figure(2, figsize=(10, 10))
kk = 0
for ii in range(n):
    for jj in range(n):
        kk += 1
        plt.subplot(n, n, kk)
        plt.plot(X[jj, :], X[ii, :], 'o', markersize=3, alpha=0.6)
        if jj == 0:
            plt.ylabel(rf'$X_{ii+1}$')
        if ii == n-1:
            plt.xlabel(rf'$X_{jj+1}$')
plt.tight_layout()

# Figure 3: scatter plot of decorrelated variables Z
plt.figure(3, figsize=(10, 10))
kk = 0
for ii in range(n):
    for jj in range(n):
        kk += 1
        plt.subplot(n, n, kk)
        plt.plot(Z[jj, :], Z[ii, :], '.k', markersize=10)
        plt.axis('square')
        if jj == 0:
            plt.ylabel(rf'$Z_{ii+1}$')
        if ii == n-1:
            plt.xlabel(rf'$Z_{jj+1}$')
plt.tight_layout()

"""
# Figure 4: histogram of log-transformed log-normal correlated random variables
plt.figure(4, figsize=(8, 6))
for jj in range(n):
    plt.subplot(n, 1, jj+1)
    plt.hist(log10(X[jj, :]), bins=20, color='lightgreen', edgecolor='black')
    plt.xlabel(rf'log$_{{10}} X_{jj+1}$')
    if jj == 0:
        plt.title(rf'Histogram of log$_{{10}}(X)$')
plt.tight_layout()

# Figure 5: scatter plot of log-transformed correlated random variables
plt.figure(5, figsize=(10, 10))
kk = 0
for ii in range(n):
    for jj in range(n):
        kk += 1
        plt.subplot(n, n, kk)
        plt.plot(log10(X[ii, :]), log10(X[jj, :]), 'o', markersize=3, alpha=0.6)
        if jj == 0:
            plt.ylabel(rf'log$_{{10}} X_{ii+1}$')
        if ii == n-1:
            plt.xlabel(rf'log$_{{10}} X_{jj+1}$')
plt.tight_layout()
"""

plt.show()

