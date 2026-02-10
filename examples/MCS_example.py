#! /usr/bin/env -S python3 -i
"""
MCS_intro.py
Monte Carlo Simulation ... an introductory example

Y = g(X1,X2,X3) = sin(X1) + sqrt(X2) - exp(-X3) - 2

H.P. Gavin, Dept. Civil and Environmental Engineering, Duke Univ, Jan. 2012
Translated to Python: 2025
"""

import numpy as np
import matplotlib.pyplot as plt

from multivarious.rvs import lognormal
from multivarious.rvs import normal
from multivarious.rvs import rayleigh
from multivarious.utl import format_plot

# Distribution parameters
#  X1 normal
mean1 = 6       # mean of normal
sdvn1 = 2       # std dev of normal
#  X2 lognormal  
medn2 = 2        # median of lognormal
covn2 = 0.3      # coefficient of variation of lognormal
#  X3 Rayleigh
mean3 = 1       # mean of Rayleigh

N = 500  # number of random values in the sample

# (1) Generate a large sample for each random variable in the problem
X1 = normal.rnd(mean1, sdvn1, N)
X2 = lognormal.rnd(medn2, covn2, N)
X3 = rayleigh.rnd(mean3, N)

# (2) Evaluate the function for each random variable to compute a new sample
Y = np.sin(X1) + np.sqrt(X2) - np.exp(-X3) - 2

# Suppose "probability of failure" is Prob[g(X1,X2,X3) > 0]
Probability_of_failure = np.sum(Y > 0) / N
print(f"Probability of failure: {Probability_of_failure:.4f}")

# (3) Plot histograms of the random variables
sort_X1 = np.sort(X1)
sort_X2 = np.sort(X2)
sort_X3 = np.sort(X3)
CDF = np.arange(1, N + 1) / (N + 1)  # empirical CDF (i/(N+1)) of ordered x

nBins = int(np.floor(N / 20))

plt.ion()

format_plot(line_width = 1, font_size = 14, marker_size = 5)
ppl = np.array([60, 30,  90]) / 256  # Purple color to match histogram bars
blu = np.array([20, 135, 235]) / 256  # Blue   color to match histogram bars
fig1 = plt.figure(1, figsize=(8, 5))
fig1.clear()

# X1: Normal distribution
plt.subplot(2, 3, 1)
counts, bins, _ = plt.hist(X1.T, bins=nBins, density=True, color=ppl, edgecolor='none')
plt.plot(sort_X1, normal.pdf(sort_X1, mean1, sdvn1), '-', linewidth=2.5)
plt.ylabel('P.D.F.')
plt.tight_layout()

plt.subplot(2, 3, 4)
plt.step(sort_X1.T, CDF, where='post', color=ppl, linewidth=5, label='Empirical')
plt.plot(sort_X1, normal.cdf(sort_X1, [mean1, sdvn1]), color=blu, linewidth=2, label='Theoretical')
plt.ylabel('C.D.F.')
plt.xlabel(r'$X_1$ : normal')
plt.tight_layout()

# X2: Lognormal distribution
plt.subplot(2, 3, 2)
counts, bins, _ = plt.hist(X2.T, bins=nBins, density=True, color=ppl, edgecolor='none')
plt.plot(sort_X2, lognormal.pdf(sort_X2, medn2, covn2), color=blu, linewidth=2.5)
plt.tight_layout()

plt.subplot(2, 3, 5)
plt.step(sort_X2, CDF, where='post', color=ppl, linewidth=5)
plt.plot(sort_X2, lognormal.cdf(sort_X2, [medn2, covn2]), color=blu, linewidth=2)
plt.xlabel(r'$X_2$ : log-normal')
plt.tight_layout()

# X3: Rayleigh distribution
plt.subplot(2, 3, 3)
counts, bins, _ = plt.hist(X3.T, bins=nBins, density=True, color=ppl, edgecolor='none')
plt.plot(sort_X3, rayleigh.pdf(sort_X3, mean3), color=blu, linewidth=2.5)
plt.tight_layout()

plt.subplot(2, 3, 6)
plt.step(sort_X3, CDF, where='post', color=ppl, linewidth=5)
plt.plot(sort_X3, rayleigh.cdf(sort_X3, mean3), color=blu, linewidth=2)
plt.xlabel(r'$X_3$ : Rayleigh')
plt.tight_layout()

plt.savefig('MCS-example-1.pdf', bbox_inches='tight')

# (4) Plot histogram of the random function Y = g(X1,X2,X3)
nBins = int(np.floor(N / 10))

fig2 = plt.figure(2, figsize=(8, 6))
fig2.clear()

plt.subplot(2, 1, 1)
counts, bins, _ = plt.hist(Y.T, bins=nBins, density=True, color=ppl, edgecolor='none')
plt.axvline(x=0, color='r', linewidth=3, label='Failure threshold')
plt.ylabel('P.D.F.')
plt.tight_layout()

plt.subplot(2, 1, 2)
plt.step(np.sort(Y), CDF.T, where='post', color=ppl, linewidth=2)
plt.axvline(x=0, color='r', linewidth=3)
plt.text(0.5, 0.5, 'Y>0', fontsize=12)
plt.ylabel('C.D.F.')
plt.xlabel(r'$Y = g(X_1,X_2,X_3)$')
plt.tight_layout()

plt.savefig('MCS-example-2.pdf', bbox_inches='tight')
plt.show()
