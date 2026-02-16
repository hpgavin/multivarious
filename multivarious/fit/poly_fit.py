"""
poly_fit.py - General Purpose Polynomial Curve Fitting with Error Analysis

Fit a power-polynomial y_fit(x;a) to data pairs (x,y) where:
    y_fit(x;a) = SUM_i a_i * x^p_i

Minimizes the Chi-square error criterion with optional regularization.
Provides comprehensive error analysis including parameter uncertainties,
confidence intervals, correlation matrix, and information criteria.

Reference:
H.P. Gavin, "Fitting Models to Data: Generalized Linear Least Squares 
and Error Analysis"
https://people.duke.edu/~hpgavin/SystemID/CouresNotes/linear-least-sqaures.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as scipy_normal
from multivarious.utl.plot_ECDF_ci import plot_ECDF_ci


def poly_fit(x, y, p, fig_no=0, Sy=None, rof=None, b=0.0):
    """
    Fit a power-polynomial to data with comprehensive error analysis.
    
    Fits: y_fit(x;c) = SUM_i c_i * x^p_i
    
    Minimizes Chi-square criterion:
        X2 = SUM_k [(y_fit(x_k;c) - y_k)^2 / Sy_k^2]
    
    Parameters
    ----------
    x : array_like, shape (N,)
        Known vector of independent variables
    y : array_like, shape (N,)
        Measured vector of dependent variables
    p : array_like, shape (n,)
        Vector of real powers (x^p) for each polynomial term
    fig_no : int, optional
        Figure number for plotting. Use 0 to suppress plotting (default: 0)
    Sy : float or array_like, optional
        Measurement errors for each value of y. 
        Scalar or shape (N,) (default: 1.0)
    rof : array_like, shape (2,), optional
        Range of fit [x_min, x_max] (default: [min(x), max(x)])
    b : float, optional
        Regularization constant (default: 0.0)
    
    Returns
    -------
    c : ndarray, shape (n,)
        Values of polynomial coefficients
    x_fit : ndarray, shape (100,)
        Values of x within range of fit
    y_fit : ndarray, shape (100,)
        Polynomial evaluated at x_fit
    Sc : ndarray, shape (n,)
        Standard errors of polynomial coefficients
    Sy_fit : ndarray, shape (100,)
        Standard errors of the curve fit
    Rc : ndarray, shape (n, n)
        Parameter correlation matrix
    R2 : float
        R-squared error criterion
    Sr : float
        Standard error of unweighted residuals
    AIC : float
        Akaike Information Criterion
    BIC : float
        Bayesian Information Criterion
    condNo : float
        Condition number of regularized system matrix
    
    Unlike numpy.polyfit, this function allows:
    . Any real-valued powers (not just integer exponents)
    . Weighted least squares with measurement errors
    . Regularization parameter
    . Comprehensive error analysis and visualization
    """
    
    # Convert inputs to numpy arrays
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    p = np.asarray(p).flatten()
    
    # Error checking
    if len(x) != len(y):
        raise ValueError('Length of x must equal length of y')
    
    Nd = len(x)  # Number of data points
    Np = len(p)  # Number of parameters
    Nf = 100     # Number of values in the fit
    
    # Default range of fit (with 5% extrapolation)
    ee = 0.05
    rfd = np.array([min(x), max(x)]) @ np.array([[1+ee, -ee], [-ee, 1+ee]])
    
    # Handle optional arguments
    if Sy is None:
        Sy = 1.0
        compute_Vr = True  # Will compute measurement errors from residuals
    else:
        compute_Vr = False
        Sy = np.asarray(Sy)
        if np.any(Sy <= 0):
            Sy = np.where(Sy <= 0, 1.0, Sy)
    
    if rof is None:
        rof = rfd
    
    # Ensure Sy is proper shape
    if np.isscalar(Sy):
        Sy = Sy * np.ones(Nd)
    
    # x values for the fit
    x_fit = np.linspace(rof[0], rof[1], Nf)
    # Inverse measurement error covariance matrix
    ISy = np.diag(1.0 / (Sy**2))
    
    # Build basis matrices
    B = np.zeros((Nd, Np))
    B_fit = np.zeros((Nf, Np))
    
    for i in range(Np):
        B[:, i] = x ** p[i]          # Model basis vectors for data
        B_fit[:, i] = x_fit ** p[i]  # Model basis vectors for fit
    
    # Condition number
    condNo = np.linalg.cond(B.T @ ISy @ B + b * np.eye(Np))
    
    # Least squares parameters
    c = np.linalg.solve(B.T @ ISy @ B + b * np.eye(Np), B.T @ ISy @ y)
    
    # Least squares fit
    y_fit = B_fit @ c
    
    # Unbiased variance of unweighted residuals
    Vr = np.sum((y - B @ c)**2) / (Nd-Np)
    
    # Measurement error covariance
    if compute_Vr:
        invVy = np.eye(Nd) / Vr  # Computed from residuals
    else:
        invVy = ISy              # Provided by user
    # Parameter covariance matrix
    Vc = np.linalg.inv(B.T @ invVy @ B + b * np.eye(Np))
    # Regularized least squares adjustment
    if b != 0:
        Vy = np.linalg.inv(ISy)
        Vc = Vc @ (B.T @ ISy @ Vy @ ISy @ B) @ Vc
    # Standard errors of parameters
    Sc = np.sqrt(np.diag(Vc))
    # Parameter cross-correlation matrix 
    Rc = Vc / np.outer(Sc, Sc) 
    # Standard error of the fit
    Sy_fit = np.sqrt(np.diag(B_fit @ Vc @ B_fit.T))
    # R-squared (coefficient of determination)
    R2 = 1 - np.sum((y - B @ c)**2) / np.sum((y - np.mean(y))**2)
    # Akaike and Bayesian Information Criteria
    AIC = np.log(2 * np.pi * Nd * Vr) + (B @ c - y).T @ invVy @ (B @ c - y) + 2 * Np
    BIC = np.log(2 * np.pi * Nd * Vr) + (B @ c - y).T @ invVy @ (B @ c - y) + Np* np.log(Nd)
    
    # Print results
    print('\n' + '='*79)
    print('Polynomial Fit Results')
    print('='*79)
    print('     p         c            +/-   dc           (percent)    correlation')
    print('-'*79)
    for i in range(Np):
        pct = 100 * Sc[i] / abs(c[i]) if c[i] != 0 else np.inf
        rr_str = ' '.join(f'{x:5.2f}' for x in Rc[i])
        if p[i] == int(p[i]):
            print(f'   c[{int(p[i]):2d}] =  {c[i]:11.3e}    +/- {Sc[i]:10.3e}     ({pct:7.2f} %)  {rr_str}')
        else:
            print(f' {p[i]:8.2f} :  {c[i]:11.3e}     +/- {Sc[i]:10.3e}      ({pct:7.2f} %)   {rr:6.2f}'   )
    print('='*79 + '\n')
    
    # Plotting
    if fig_no > 0:
        _plot_results(x, y, x_fit, y_fit, B, c, Sy_fit, Vr, 
                     condNo, R2, AIC, BIC, Nd, Np, fig_no)
    
    return c, x_fit, y_fit, Sc, Sy_fit, Rc, R2, Vr, AIC, BIC, condNo


def _plot_results(x, y, x_fit, y_fit, B, c, Sy_fit, Vr, 
                  condNo, R2, AIC, BIC, Nd, Np, fig_no):
    """
    Create visualization of polynomial fit results.
    
    Internal function called by poly_fit when fig_no > 0.
    """
    
    # Confidence intervals
    CI = np.array([0.90, 0.99])
    z = scipy_normal.ppf(1 - (1 - CI) / 2)

    # Confidence bands for the model
    yps95 = y_fit + z[0] * Sy_fit
    yms95 = y_fit - z[0] * Sy_fit
    yps99 = y_fit + z[1] * Sy_fit
    yms99 = y_fit - z[1] * Sy_fit
    
    # Coordinates for patch plots
    xp = np.concatenate([x_fit, x_fit[::-1], [x_fit[0]]])
    yp95 = np.concatenate([yps95, yms95[::-1], [yps95[0]]])
    yp99 = np.concatenate([yps99, yms99[::-1], [yps99[0]]])
    
    # Colors for confidence intervals
    patchColor95 = [0.95, 0.95, 0.1]
    patchColor99 = [0.2, 0.95, 0.2]
    
    # Set plotting style
    plt.rcParams.update({
        'font.size': 12,
        'lines.linewidth': 2,
        'lines.markersize': 5
    })
    
    # Figure 1: Data, fit, and confidence intervals
    fig_1 = plt.figure(fig_no, figsize=(12, 5))
    fig_1.clf()
    
    # Left subplot: Data and model with confidence intervals
    ax1 = plt.subplot(1, 2, 1)
    ax1.fill(xp, yp99, color=patchColor99, edgecolor=patchColor99, 
             alpha=0.3, label=f'{int(CI[1]*100)}% c.i.')
    ax1.fill(xp, yp95, color=patchColor95, edgecolor=patchColor95, 
             alpha=0.5, label=f'{int(CI[0]*100)}% c.i.')
    ax1.plot(x, y, 'ob', linewidth=3, markersize=6, label=r'data $y$')
    ax1.plot(x_fit, y_fit, '-k', linewidth=2, label=r'model $\hat y(x)$')
    ax1.set_xlabel(r'$x$', fontsize=15)
    ax1.set_ylabel(r'data $y$   and   model $\hat y(x; c^*)$', fontsize=15)
    ax1.legend(loc='best', fontsize=11)
    ax1.set_xlim([min(xp), max(xp)])
    y_range = max(y) - min(y)
    ax1.set_ylim([min(y) - 0.1*y_range, max(y) + 0.1*y_range])
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Polynomial Fit with Confidence Intervals', fontsize=14)
    
    # Right subplot: Data vs model (correlation plot)
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(y, y, '-k', linewidth=0.5, alpha=0.5)  # 1-to-1 line
    ax2.plot(B @ c, y, 'ob', linewidth=3, markersize=6)
    
    # Add statistics text
    tx = min(y)
    ty_range = max(y) - min(y)
    positions = [0.99, 0.91, 0.83, 0.75, 0.67, 0.59, 0.51]
    
    ax2.text(tx, min(y) + positions[0]*ty_range, 
             rf'$N$ = {Nd} data points', fontsize=12) 
    ax2.text(tx, min(y) + positions[1]*ty_range, 
             rf'$n$ = {Np} coefficients', fontsize=12)
    ax2.text(tx, min(y) + positions[2]*ty_range, 
             f'cond # = {condNo:.1f}', fontsize=12)
    ax2.text(tx, min(y) + positions[3]*ty_range, 
             rf'$\sigma_{{r}}$ = {np.sqrt(Vr):.3f}', fontsize=12)
    ax2.text(tx, min(y) + positions[4]*ty_range, 
             rf'$R^2$ = {R2:.3f}', fontsize=12)
    ax2.text(tx, min(y) + positions[5]*ty_range, 
             f'AIC = {AIC:.2f}', fontsize=12)
    ax2.text(tx, min(y) + positions[6]*ty_range, 
             f'BIC = {BIC:.2f}', fontsize=12)
    
    ax2.set_xlabel(r'model   $\hat y(x; c^*)$', fontsize=15)
    ax2.set_ylabel(r'data   $y$', fontsize=15)
    ax2.axis('tight')
    ax2.grid(True, alpha=0.3)
    ax2.set_title(r'Data $y$ vs Model $\hat y(x; c^*)$ (Correlation)', fontsize=14)

    filename = f'poly_fit-{fig_no:04d}.pdf'
    fig_1.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"    Saved: {filename}")
    
    plt.tight_layout()
    
    # Figure 2: Histogram of residuals
    residuals = y - B @ c
    nBars = max(10, round(Nd / 10))
    
    fig_2 = plt.figure(fig_no + 1, figsize=(8, 5))
    fig_2.clf()
    
    counts, bins, _ = plt.hist(residuals, bins=nBars, 
            color='royalblue', edgecolor='black', alpha=0.7)
    plt.xlabel(r'Residuals, $r = y - \hat y(x; c^*)$', fontsize=15)
    plt.ylabel(r'Empirical PDF, $f_R(r)$', fontsize=15)
    plt.title('Distribution of Residuals', fontsize=15)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add normal distribution overlay
    meanR = np.sum(residuals)/Np
    xmin, xmax = plt.xlim()
    x_normal = np.linspace(xmin, xmax, 100)
    p_normal = scipy_normal.pdf(x_normal, meanR, np.sqrt(Vr))
    # Scale to match histogram
    p_normal_scaled = p_normal * len(residuals) * (bins[1] - bins[0])
    plt.plot(x_normal, p_normal_scaled, '-', color='darkblue',linewidth=4) 
    plt.text( np.sqrt(Vr), np.max(p_normal_scaled), rf'$\sigma_{{r}} = {np.sqrt(Vr):.3f}$', fontsize=18)
    
    plt.tight_layout()
    
    filename = f'poly_fit-{fig_no+1:04d}.pdf'
    fig_2.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"    Saved: {filename}")
    
    # Figure 3: ECDF of residuals with confidence intervals
    plot_ECDF_ci(residuals, 95, fig_no + 2, x_label= r'Residuals, $r = y - \hat y(x; c^*)$')

