#! /usr/bin/env -S python3 -i

import numpy as np
from multivarious.rvs import beta.py, binomial.py, chi2.py, exponential.py, extreme_value_I.py, extreme_value_II.py, gamma.py, gev.py, laplace.py, lognormal.py, normal.py, poisson.py, quadratic.py, rayleigh.py, students_t.py, triangular.py, uniform.py 
import matplotlib.pyplot as plt



meanX = 10
covnX =  0.3

x = np.linspace(0.01,meanX*(1+9*covnX), 100)

F  = beta.cdf(x,[meanX,covnX])
xi = beta.inv(F, meanX,covnX)

# =========================================

plt.ion()

plt.figure(1)
plt.clf()
plt.plot(x,F,'-', linewidth=5)
plt.plot(xi,F,'--', color='yellow')
plt.xlabel(r'$x$')
plt.ylabel(r'$F_X(x)$')
plt.show()

plt.figure(2)
plt.clf()
plt.plot(x,xi,'o', markersize=5)
plt.xlabel(r'$x$')
plt.ylabel(r' inv $(F_X(x))$')
plt.show()

plt.figure(3)
plt.clf()
plt.loglog(F,np.abs(xi-x),'o', markersize=5)
plt.xlabel(r'$F_X(x)$')
plt.ylabel(r'| inv $(F_X(x)) - x$ |')
plt.show()

