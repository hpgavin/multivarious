"""
poly_fit.py - General Purpose Polynomial Curve Fitting with Error Analysis

Fit a power-polynomial y_fit(x;a) to data pairs (x,y) where:
    y_fit(x;a) = SUM_i a_i * x^p_i

Minimizes the Chi-square error criterion with optional regularization.
Provides comprehensive error analysis including parameter uncertainties,
confidence intervals, correlation matrix, and information criteria.

Translation from MATLAB by Claude, 2025-10-24
Original by H.P. Gavin

Reference:
H.P. Gavin, "Fitting Models to Data: Generalized Linear Least Squares 
and Error Analysis"
https://people.duke.edu/~hpgavin/SystemID/linear-least-sqaures.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from multivarious.utl.plot_ECDF_ci import plot_ECDF_ci


def poly_fit(x, y, p, figNo=0, Sy=None, rof=None, b=0.0):
    """
    Fit a power-polynomial to data with comprehensive error analysis.
    
    Fits: y_fit(x;c) = SUM_i c_i * x^p_i
    
    Minimizes Chi-square criterion:
        X2 = SUM_k [(y_fit(x_k;c) - y_k)^2 / Sy_k^2]
    
    Parameters
    ----------
    x : array_like, shape (m,)
        Known vector of independent variables
    y : array_like, shape (m,)
        Measured vector of dependent variables
    p : array_like, shape (n,)
        Vector of real powers (x^p) for each polynomial term
    figNo : int, optional
        Figure number for plotting. Use 0 to suppress plotting (default: 0)
    Sy : float or array_like, optional
        Measurement errors for each value of y. 
        Scalar or shape (m,) (default: 1.0)
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
    Vr : float
        Unbiased variance of unweighted residuals
    AIC : float
        Akaike Information Criterion
    condNo : float
        Condition number of regularized system matrix
    
    Notes
    -----
    Unlike numpy.polyfit, this function allows:
    - Any real-valued powers (not just integer exponents)
    - Weighted least squares with measurement errors
    - Regularization parameter
    - Comprehensive error analysis and visualization
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
    Vr = np.sum((y - B @ c)**2) / (Nd - Np - 1)
    
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
    
    # Akaike Information Criterion
    AIC = np.log(2 * np.pi * Np * Vr) + (B @ c - y).T @ invVy @ (B @ c - y) + 2 * Np
    
    # Print results
    print('\n' + '='*70)
    print('Polynomial Fit Results')
    print('='*70)
    print('     p         c            +/-   dc           (percent)')
    print('-'*65)
    for i in range(Np):
        pct = 100 * Sc[i] / abs(c[i]) if c[i] != 0 else np.inf
        if p[i] == int(p[i]):
            print(f'   c[{int(p[i]):2d}] =  {c[i]:11.3e}    +/- {Sc[i]:10.3e}    '
                  f'({pct:7.2f} %)')
        else:
            print(f' {p[i]:8.2f} :  {c[i]:11.3e}     +/- {Sc[i]:10.3e}    '
                  f'({pct:7.2f} %)')
    print('='*70 + '\n')
    
    # Plotting
    if figNo > 0:
        _plot_results(x, y, x_fit, y_fit, B, c, Sy_fit, Vr, 
                     condNo, R2, AIC, Nd, Np, figNo)
    
    return c, x_fit, y_fit, Sc, Sy_fit, Rc, R2, Vr, AIC, condNo


def _plot_results(x, y, x_fit, y_fit, B, c, Sy_fit, Vr, 
                  condNo, R2, AIC, Nd, Np, figNo):
    """
    Create visualization of polynomial fit results.
    
    Internal function called by poly_fit when figNo > 0.
    """
    
    # Confidence intervals
    CI = np.array([0.90, 0.99])
    z = norm.ppf(1 - (1 - CI) / 2)
    
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
    fig = plt.figure(figNo, figsize=(14, 6))
    fig.clf()
    
    # Left subplot: Data and model with confidence intervals
    ax1 = plt.subplot(1, 2, 1)
    ax1.fill(xp, yp99, color=patchColor99, edgecolor=patchColor99, 
             alpha=0.3, label=f'{int(CI[1]*100)}% c.i.')
    ax1.fill(xp, yp95, color=patchColor95, edgecolor=patchColor95, 
             alpha=0.5, label=f'{int(CI[0]*100)}% c.i.')
    ax1.plot(x, y, 'ob', linewidth=3, markersize=6, label='data')
    ax1.plot(x_fit, y_fit, '-k', linewidth=2, label='y_fit')
    ax1.set_xlabel('x', fontsize=13)
    ax1.set_ylabel('y', fontsize=13)
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
    tx = 0.90 * min(y) + 0.10 * max(y)
    ty_range = max(y) - min(y)
    positions = [0.98, 0.90, 0.82, 0.74, 0.66]
    
    ax2.text(tx, min(y) + positions[0]*ty_range, 
             f'cond # = {condNo:.1f}', fontsize=11)
    ax2.text(tx, min(y) + positions[1]*ty_range, 
             f'σ_r = {np.sqrt(Vr):.3f}', fontsize=11)
    ax2.text(tx, min(y) + positions[2]*ty_range, 
             f'AIC = {AIC:.1f}', fontsize=11)
    ax2.text(tx, min(y) + positions[3]*ty_range, 
             f'R² = {R2:.3f}', fontsize=11)
    ax2.text(tx, min(y) + positions[4]*ty_range, 
             f'n = {Np}', fontsize=11)
    
    ax2.set_xlabel('y_fit', fontsize=13)
    ax2.set_ylabel('y', fontsize=13)
    ax2.axis('tight')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Model vs Data (Correlation)', fontsize=14)
    
    plt.tight_layout()
    
    # Figure 2: Histogram of residuals
    residuals = y - B @ c
    nBars = max(10, round(Nd / 5))
    
    fig2 = plt.figure(figNo + 1, figsize=(10, 6))
    fig2.clf()
    
    counts, bins, _ = plt.hist(residuals, bins=nBars, 
                                color=[0.3, 0.7, 0.9], 
                                edgecolor='black', alpha=0.7)
    plt.xlabel('Residuals, r = y - y_fit', fontsize=13)
    plt.ylabel('Empirical PDF, f_R(r)', fontsize=13)
    plt.title('Distribution of Residuals', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add normal distribution overlay
    mu, std = np.mean(residuals), np.std(residuals)
    xmin, xmax = plt.xlim()
    x_normal = np.linspace(xmin, xmax, 100)
    p_normal = norm.pdf(x_normal, mu, std)
    # Scale to match histogram
    p_normal_scaled = p_normal * len(residuals) * (bins[1] - bins[0])
    plt.plot(x_normal, p_normal_scaled, 'r-', linewidth=2, 
             label=f'Normal(μ={mu:.3f}, σ={std:.3f})')
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    
    # Figure 3: ECDF of residuals with confidence intervals
    plot_ECDF_ci(residuals, 95, figNo + 2)

