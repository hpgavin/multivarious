#! /usr/bin/env -S python3 -i
## rvs_example.py
## Test and visualize all seventeen random variable distribution modules.
## github.com/hpgavin/multivarious ... rvs/

import numpy as np
import matplotlib.pyplot as plt
from multivarious.rvs import *          # import all rvs distributions
from multivarious.utl import format_plot
from rich.traceback import install; install()

# ── dimensions ────────────────────────────────────────────────────────────────
n = 3        # number of correlated random variables
N = 1612     # number of observations (samples)

# ── shared distribution parameters ────────────────────────────────────────────
meanX = 5 + 2*np.arange(1, n+1).reshape(-1, 1) * 2.5 / n   # (n,1) means
sdvnX = 2 + np.arange(n+1, 1, -1).reshape(-1, 1) * 1.0 / n # (n,1) std devs
covnX = sdvnX / meanX                                  # (n,1) coefficients of variation
mednX = meanX / np.sqrt(1 + covnX**2)                 # (n,1) medians (lognormal)

a = meanX + 1.0                   # (n,1) lower bounds
b = meanX + 2.0                   # (n,1) upper bounds
c = 0.3*a + 0.7*b                 # (n,1) modes (triangular)
q = np.arange(n+1, 1, -1).reshape(-1, 1).astype(float)  # (n,1) shape param 1
p = np.arange(2, n+2).reshape(-1, 1).astype(float)       # (n,1) shape param 2

m = meanX                         # (n,1) location (gev, extreme_value_I/II)
s = sdvnX                         # (n,1) scale    (gev, extreme_value_I/II)

# Poisson parameters
T = meanX                         # (n,1) mean return periods
t = np.ones((n, 1))              # (n,1) observation durations

# Binomial parameters
mb = np.array([10,  5, 12])      # (n,)  number of Bernoulli trials  (integers)
pb = np.array([ 0.3, 0.4, 0.7]) # (n,)  Bernoulli success probabilities
nb = np.arange(np.max(mb) + 1)  # (N_k,) number of successes (0 … max(mb))

# Default evaluation grid
x = np.linspace(0.1 * np.min(meanX), 3.0 * np.max(meanX), 200)

# Correlation matrix R (must be positive definite)
R = np.array([[1.0, -0.5, -0.8],
              [-0.5,  1.0,  0.9],
              [-0.8,  0.9,  1.0]])

print('─' * 60)
print(f'n = {n} variables,   N = {N} observations')
print('meanX:', meanX.flatten())
print('sdvnX:', sdvnX.flatten())
print('covnX:', covnX.flatten())
print('Correlation matrix R:\n', R)
print('─' * 60)

# ── Select distribution ────────────────────────────────────────────────────────
#  'beta'            'binomial'        'chi2'
#  'exponential'     'extreme_value_I' 'extreme_value_II'
#  'gamma'           'gev'             'laplace'
#  'lognormal'       'normal'          'poisson'
#  'quadratic'       'rayleigh'        'students_t'
#  'triangular'      'uniform'

distribution = 'rayleigh'

# ── PMF/PDF, CDF, inverse CDF, and random samples ─────────────────────────────

if distribution == 'beta':
    x  = np.linspace(np.min(a) - 0.1*np.max(b-a),
                     np.max(b) + 0.1*np.max(b-a), 200)
    fx = beta.pdf( x, a, b, q, p )
    Fx = beta.cdf( x, (a, b, q, p) )
    xi = beta.inv( Fx, a, b, q, p )
    X  = beta.rnd( a, b, q, p, N, R )

if distribution == 'binomial':
    # mb, pb as 1D; nb as evaluation grid — no inv for discrete distributions
    fx = binomial.pmf( nb, mb, pb )    # shape (n, len(nb)) or squeezed
    Fx = binomial.cdf( nb, (mb, pb) )
    X  = binomial.rnd( mb, pb, N, R )
    x  = nb
    xi = None                          # no inverse CDF for discrete distributions

if distribution == 'chi2':
    k_chi2 = np.array([2, 5, 10]).reshape(-1, 1)   # local k for chi2
    x  = np.linspace(0.1, 3.0*np.max(k_chi2), 250)
    fx = chi2.pdf( x, k_chi2 )
    Fx = chi2.cdf( x, k_chi2 )
    xi = chi2.inv( Fx, k_chi2 )
    X  = chi2.rnd( k_chi2, N, R )

if distribution == 'exponential':
    fx = exponential.pdf( x, meanX )
    Fx = exponential.cdf( x, meanX )          # cdf(x, meanX), not (x, params)
    xi = exponential.inv( Fx, meanX )
    X  = exponential.rnd( meanX, N, R )

if distribution == 'extreme_value_I':
    fx = extreme_value_I.pdf( x, meanX, covnX )
    Fx = extreme_value_I.cdf( x, (meanX, covnX) )
    xi = extreme_value_I.inv( Fx, meanX, covnX )
    X  = extreme_value_I.rnd( meanX, covnX, N, R )

if distribution == 'extreme_value_II':
    # m, s, k must all be > 0
    fx = extreme_value_II.pdf( x, m, s, q )
    Fx = extreme_value_II.cdf( x, (m, s, q) )
    xi = extreme_value_II.inv( Fx, m, s, q )
    X  = extreme_value_II.rnd( m, s, q, N, R )

if distribution == 'gamma':
    fx = gamma.pdf( x, meanX, covnX )
    Fx = gamma.cdf( x, (meanX, covnX) )
    xi = gamma.inv( Fx, meanX, covnX )         # covnX, not convX
    X  = gamma.rnd( meanX, covnX, N, R )

if distribution == 'gev':
    # k is the shape parameter; small values give well-behaved distributions
    k_gev = np.array([0.2, 0.3, 0.4]).reshape(-1, 1)
    fx = gev.pdf( x, m, s, k_gev )
    Fx = gev.cdf( x, (m, s, k_gev) )
    xi = gev.inv( Fx, m, s, k_gev )
    X  = gev.rnd( m, s, k_gev, N, R )

if distribution == 'laplace':
    fx = laplace.pdf( x, meanX, sdvnX )
    Fx = laplace.cdf( x, (meanX, sdvnX) )
    xi = laplace.inv( Fx, meanX, sdvnX )
    X  = laplace.rnd( meanX, sdvnX, N, R )

if distribution == 'lognormal':
    fx = lognormal.pdf( x, mednX, covnX )
    Fx = lognormal.cdf( x, (mednX, covnX) )
    xi = lognormal.inv( Fx, mednX, covnX )
    X  = lognormal.rnd( mednX, covnX, N, R )

if distribution == 'normal':
    fx = normal.pdf( x, meanX, sdvnX )
    Fx = normal.cdf( x, (meanX, sdvnX) )
    xi = normal.inv( Fx, meanX, sdvnX )
    X  = normal.rnd( meanX, sdvnX, N, R )

if distribution == 'poisson':
    # t and T as (n,1); nb as evaluation grid — no inv for discrete distributions
    fx = poisson.pmf( nb, t, T )
    Fx = poisson.cdf( nb, (t, T) )
    X  = poisson.rnd( t, T, N, R )
    x  = nb
    xi = None                          # no inverse CDF for discrete distributions

if distribution == 'quadratic':
    x  = np.linspace(np.min(a), np.max(b), 250)
    fx = quadratic.pdf( x, a, b )
    Fx = quadratic.cdf( x, (a, b) )
    xi = quadratic.inv( Fx, a, b )
    X  = quadratic.rnd( a, b, N, R )

if distribution == 'rayleigh':
    fx = rayleigh.pdf( x, meanX )
    Fx = rayleigh.cdf( x, meanX )              # cdf(x, meanX), not (x, params)
    xi = rayleigh.inv( Fx, meanX )
    X  = rayleigh.rnd( meanX, N, R )

if distribution == 'students_t':
    k_t = np.array([2, 5, 10]).reshape(-1, 1)  # local k: degrees of freedom
    x   = np.linspace(-10, 10, 250)
    fx  = students_t.pdf( x, k_t )
    Fx  = students_t.cdf( x, k_t )             # cdf(t, k), not (t, [k])
    xi  = students_t.inv( Fx, k_t )
    X   = students_t.rnd( k_t, N, R )

if distribution == 'triangular':
    x  = np.linspace(np.min(a), np.max(b), 250)
    fx = triangular.pdf( x, a, b, c )
    Fx = triangular.cdf( x, (a, b, c) )
    xi = triangular.inv( Fx, a, b, c )
    X  = triangular.rnd( a, b, c, N, R )

if distribution == 'uniform':
    x  = np.linspace(np.min(a), np.max(b), 250)
    fx = uniform.pdf( x, a, b )
    Fx = uniform.cdf( x, (a, b) )
    xi = uniform.inv( Fx, a, b )
    X  = uniform.rnd( a, b, N, R )

# ── Ensure 2D shapes for plotting ─────────────────────────────────────────────
X  = np.atleast_2d(X)
fx = np.atleast_2d(fx)
Fx = np.atleast_2d(Fx)
if xi is not None:
    xi = np.atleast_2d(xi)

print(f'distribution : {distribution}')
print(f'X   shape    : {X.shape}')
print(f'fx  shape    : {fx.shape}')
print(f'Fx  shape    : {Fx.shape}')

from scipy.stats import kstest

is_discrete = distribution in ['binomial', 'poisson']

# ── Check 1: PDF/PMF integrates (sums) to 1 ───────────────────────────────────
integrals = np.full(n, np.nan)
for i in range(n):
    if is_discrete:
        integrals[i] = np.sum(fx[i, :])           # PMF sums to 1
    else:
        integrals[i] = np.trapz(fx[i, :], x)      # PDF integrates to 1

# ── Check 2: CDF is monotone non-decreasing and spans [0, 1] ──────────────────
cdf_monotone = np.full(n, True)
cdf_lb       = np.full(n, np.nan)   # CDF at left edge of x grid
cdf_ub       = np.full(n, np.nan)   # CDF at right edge of x grid
for i in range(n):
    cdf_monotone[i] = np.all(np.diff(Fx[i, :]) >= -1e-10)
    cdf_lb[i]       = Fx[i,  0]
    cdf_ub[i]       = Fx[i, -1]

# ── Check 3: KS test — sample consistent with theoretical CDF ─────────────────
# Build a scalar CDF callable for variable i from the pre-computed (x, Fx) grid.
# np.interp extrapolates as 0 / 1 outside the grid, which is correct for KS.
ks_stat = np.full(n, np.nan)
ks_pval = np.full(n, np.nan)
if not is_discrete:
    for i in range(n):
        cdf_i = lambda v, i=i: np.interp(v, x, Fx[i, :], left=0.0, right=1.0)
        ks_stat[i], ks_pval[i] = kstest(X[i, :], cdf_i)

# ── Check 4: Sample moments vs theoretical moments ────────────────────────────
mean_sample = np.mean(X, axis=1)          # (n,)
sdvn_sample = np.std( X, axis=1)          # (n,)

# ── Check 5: Sample correlation vs R ──────────────────────────────────────────
corr_sample    = np.corrcoef(X)           # (n, n)
corr_max_error = np.max(np.abs(corr_sample - R))
corr_2sigma    = 2.0 / np.sqrt(N)        # approximate 2-sigma tolerance

# ── Check 6: Round-trip error  max|inv(cdf(x)) - x| ──────────────────────────
rt_max_err = np.full(n, np.nan)
if xi is not None and not is_discrete:
    for i in range(n):
        rt_max_err[i] = np.max(np.abs(xi[i, :] - x))

# ── Decorrelated sample Z (for scatter-plot matrix upper triangle) ─────────────
if n > 1 and not is_discrete:
    mean_X_col = mean_sample.reshape(-1, 1)
    covr_X     = np.cov(X)
    eVal, eVec = np.linalg.eigh(covr_X)
    invT       = np.linalg.inv(np.sqrt(np.diag(eVal))) @ eVec.T
    Z          = invT @ (X - mean_X_col)
else:
    Z = X

# ── Summary table ─────────────────────────────────────────────────────────────
W = 72    # total table width
print()
print('─' * W)
print(f'  Quantitative checks   distribution = {distribution}   N = {N}')
print('─' * W)

# Check 1 — PDF/PMF normalisation
label1 = 'PMF Σ p(k)' if is_discrete else 'PDF ∫ f(x)dx'
print(f'\n  Check 1 — {label1}   (should be 1.0)')
print(f'  {"var":>4}   {"integral":>10}   {"error":>10}   {"pass/fail":>9}')
print(f'  {"─"*4}   {"─"*10}   {"─"*10}   {"─"*9}')
for i in range(n):
    err  = abs(integrals[i] - 1.0)
    flag = 'PASS' if err < 1e-2 else 'FAIL'
    print(f'  X{i+1:>3}   {integrals[i]:>10.6f}   {err:>10.2e}   {flag:>9}')

# Check 2 — CDF bounds and monotonicity
print(f'\n  Check 2 — CDF bounds and monotonicity')
print(f'  {"var":>4}   {"F(x_min)":>10}   {"F(x_max)":>10}   {"monotone":>9}   {"pass/fail":>9}')
print(f'  {"─"*4}   {"─"*10}   {"─"*10}   {"─"*9}   {"─"*9}')
for i in range(n):
    lb_ok = cdf_lb[i] < 0.10
    ub_ok = cdf_ub[i] > 0.90
    flag  = 'PASS' if (cdf_monotone[i] and lb_ok and ub_ok) else 'FAIL'
    print(f'  X{i+1:>3}   {cdf_lb[i]:>10.4f}   {cdf_ub[i]:>10.4f}'
          f'   {"yes" if cdf_monotone[i] else "NO":>9}   {flag:>9}')

# Check 3 — KS test
if not is_discrete:
    print(f'\n  Check 3 — Kolmogorov-Smirnov test   (p > 0.01 → PASS)')
    print(f'  {"var":>4}   {"KS stat":>10}   {"p-value":>10}   {"pass/fail":>9}')
    print(f'  {"─"*4}   {"─"*10}   {"─"*10}   {"─"*9}')
    for i in range(n):
        flag = 'PASS' if ks_pval[i] > 0.01 else 'FAIL'
        print(f'  X{i+1:>3}   {ks_stat[i]:>10.4f}   {ks_pval[i]:>10.4f}   {flag:>9}')

# Check 4 — Sample moments
print(f'\n  Check 4 — Sample moments vs theoretical')
print(f'  {"var":>4}   {"mean theory":>12}   {"mean sample":>12}'
      f'   {"sdvn theory":>12}   {"sdvn sample":>12}')
print(f'  {"─"*4}   {"─"*12}   {"─"*12}   {"─"*12}   {"─"*12}')
for i in range(n):
    try:
        mu_th = meanX[i].item()   # .item() extracts a plain Python scalar
        sd_th = sdvnX[i].item()   # from a single-element array of any shape
    except Exception:
        mu_th = np.nan
        sd_th = np.nan
    print(f'  X{i+1:>3}   {float(mu_th):>12.4f}   {float(mean_sample[i]):>12.4f}'
          f'   {float(sd_th):>12.4f}   {float(sdvn_sample[i]):>12.4f}')

# Check 5 — Sample correlation vs R
print(f'\n  Check 5 — Sample correlation vs R'
      f'   (2-sigma tolerance = {corr_2sigma:.4f})')
print(f'  max |corr(X) - R| = {corr_max_error:.4f}   '
      f'{"PASS" if corr_max_error < corr_2sigma else "FAIL"}')
print(f'\n  Estimated correlation of X:')
for row in corr_sample:
    print('  ' + '  '.join(f'{v:7.4f}' for v in row))
print(f'\n  Target correlation R:')
for row in R:
    print('  ' + '  '.join(f'{v:7.4f}' for v in row))

# Check 6 — Round-trip error
if xi is not None and not is_discrete:
    print(f'\n  Check 6 — Round-trip error   max|inv(cdf(x)) - x|')
    print(f'  {"var":>4}   {"max error":>12}   {"pass/fail":>9}')
    print(f'  {"─"*4}   {"─"*12}   {"─"*9}')
    for i in range(n):
        flag = 'PASS' if rt_max_err[i] < 1e-6 else 'WARN'
        print(f'  X{i+1:>3}   {rt_max_err[i]:>12.2e}   {flag:>9}')

print()
print('─' * W)

# ── Plots ──────────────────────────────────────────────────────────────────────
format_plot(font_size=16, line_width=2, marker_size=4)
plt.ion()

# Figure 1: scatter-plot matrix  ───────────────────────────────────────────────
# diagonal: histogram + theoretical PDF
# lower triangle: scatter plot of correlated X_i vs X_j
# upper triangle: scatter plot of decorrelated Z_i vs Z_j

if distribution in ['binomial', 'poisson']:
    # For discrete distributions plot PMF only
    plt.figure(1, figsize=(8, 5))
    plt.plot(x, fx.T, '-o', markersize=4)
    plt.xlabel(r'$n$')
    plt.ylabel(r'PMF $p_N(n)$')
    plt.title(distribution)
    plt.tight_layout()

else:
    plt.figure(1, figsize=(10, 10))
    kk = 0
    for ii in range(n):
        for jj in range(n):
            kk += 1
            plt.subplot(n, n, kk)
            if ii == jj:
                plt.hist(X[ii, :], bins=25, color='royalblue',
                         edgecolor='black', density=True)
                plt.plot(x, fx[ii, :], '-k')
            elif ii > jj:
                plt.plot(X[jj, :], X[ii, :], 'o',
                         color='royalblue', markersize=2)
            else:
                plt.plot(Z[jj, :], Z[ii, :], 'o',
                         color='black', markersize=2)
            if jj == 0:
                plt.ylabel(rf'$X_{ii+1}$')
            if ii == n-1:
                plt.xlabel(rf'$X_{jj+1}$')
    plt.suptitle(distribution, y=1.01)
    plt.tight_layout()

# Figure 2: empirical vs theoretical CDF  ─────────────────────────────────────
ECDF = (np.arange(N) + 1) / (N + 1)

if distribution in ['binomial', 'poisson']:
    plt.figure(2, figsize=(8, 5))
    plt.plot(x, Fx.T, '-o', markersize=4)
    plt.xlabel(r'$n$')
    plt.ylabel(r'CDF $F_N(n)$')
    plt.title(distribution)
    plt.tight_layout()

else:
    plt.figure(2, figsize=(4*n, 4))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.step(np.sort(X[i, :]), ECDF, color='royalblue', linewidth=4,
                 label='empirical')
        plt.plot(x, Fx[i, :], '-k', label='theoretical')
        plt.xlabel(rf'$X_{i+1}$')
        if i == 0:
            plt.ylabel(r'CDF $F_X(x)$')
    plt.suptitle(distribution)
    plt.tight_layout()

# Figure 3: CDF and round-trip inverse CDF  ───────────────────────────────────
if xi is not None and distribution not in ['binomial', 'poisson']:
    plt.figure(3, figsize=(6, 5))
    for i in range(n):
        color = f'C{i}'
        plt.plot(x,      Fx[i, :], '-',  color=color, linewidth=4,
                 label=rf'$X_{i+1}$ theoretical')
        plt.plot(xi[i,:], Fx[i, :], '--', color='k',   linewidth=1)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$F_X(x)$')
    plt.title(rf'{distribution}   CDF and inv(CDF)')
    plt.tight_layout()

# Figure 4: round-trip error  |inv(F_X(x)) - x|  ─────────────────────────────
if xi is not None and distribution not in ['binomial', 'poisson']:
    plt.figure(4, figsize=(6, 5))
    for i in range(n):
        err = np.abs(xi[i, :] - x)
        plt.semilogy(x, err + 1e-16, label=rf'$X_{i+1}$')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$|$inv$(F_X(x)) - x|$')
    plt.title(rf'{distribution}   round-trip error')
    plt.legend()
    plt.tight_layout()
