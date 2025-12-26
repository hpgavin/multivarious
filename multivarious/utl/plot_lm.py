"""
Utility for plotting results of a Levenberg-Marquardt curve-fitting process
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


def plot_lm(t: np.ndarray,
             y_data: np.ndarray,
             y_fit: np.ndarray,
             sigma_y: np.ndarray,
             chi_sq: float, 
             aic: float, 
             bic: float, 
             cvg_history: np.ndarray,
             title_prefix: str = "LM_fit") -> None:
    """
    Plot convergence history and fit results for Levenberg-Marquardt.
    
    Creates three figures:
    1. Convergence history (coefficients and chi-squared vs iterations)
    2. Data, fit, and confidence intervals
    3. Histogram of residuals
    
    Parameters
    ----------
    t : ndarray
        Independent variable
    y_data : ndarray
        Measured data
    y_fit : ndarray
        Fitted model
    sigma_y : ndarray
        Standard errors of fit
    chi_sq : float
        Reduced chi-squared least-squares criterion
    aic : float
        Akaike Information Criterion
    bic : float
        Bayes Information Criterion
    cvg_history : ndarray
        Convergence history from LM algorithm
    title_prefix : str
        Prefix for plot titles and filenames
    """
    plt.ion()
    
    max_iter, n_cols = cvg_history.shape
    n_coeffs = n_cols - 3

#   fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    #fig = plt.figure(layout='constrained', figsize=(12, 12))
    fig = plt.figure(figsize=(16, 13))

    subfigs = fig.subfigures(2, 2, wspace=0.07)    # 2 x 2 grid of subfigures

    ax0 = subfigs[0,0].add_subplot()               # data and fit evolution
    ax1 = subfigs[0,1].add_subplot()               # model vs data scatter plot
    ax2 = subfigs[1,0].subplots(2, 1, sharex=True) # coeff. and cvgnce evolution
    ax3 = subfigs[1,1].add_subplot()               # histogram of resid subplot 

    # ========================================================================
    # subfig 0: Data, fit, and confidence intervals
    # ========================================================================
    # fig0, ax0 = plt.subplots(figsize=(10, 6))
    if t.ndim == 1: 
    
        # Confidence interval patches
        color_95 = [0.95, 0.95, 0.1]
        color_99 = [0.2, 0.95, 0.2]
    
        # 95% confidence interval
        y_upper_95 = y_fit + 1.96 * sigma_y
        y_lower_95 = y_fit - 1.96 * sigma_y
    
        # 99% confidence interval
        y_upper_99 = y_fit + 2.58 * sigma_y
        y_lower_99 = y_fit - 2.58 * sigma_y
    
        # Plot confidence intervals as filled regions
        ax0.fill_between(t, y_lower_99, y_upper_99, 
                        color=color_99, alpha=0.6, label='99% C.I.')
        ax0.fill_between(t, y_lower_95, y_upper_95, 
                        color=color_95, alpha=0.8, label='95% C.I.')
    
        # Plot data and fit
        ax0.plot(t, y_data, 'ob', markersize=4, label='$y_{data}$')
        ax0.plot(t, y_fit, '-k', linewidth=2, label='$y_{fit}$')
        
        ax0.set_xlabel('$t$', fontsize=12)
        ax0.set_ylabel('$y(t)$', fontsize=12)
        ax0.set_title(f'{title_prefix}: Data and Fit with Confidence Intervals', 
                    fontsize=12, fontweight='bold')
        ax0.legend(loc='best', fontsize=10)
        ax0.grid(True, alpha=0.3)
    
       
    #plt.tight_layout()

    # ========================================================================
    # subfig 1: Predicted vs Observed
    # ========================================================================
    # fig1, ax1 = plt.subplots(figsize=(10, 6))

    # Scatter plot of fit vs data
    ax1.plot(np.array([ np.min(y_data), np.max(y_data) ]), np.array([ np.min(y_fit), np.max(y_fit) ]), '-k', linewidth=0.5 )
    ax1.plot(y_data, y_fit, 'ob', markersize=4)
    
    ax1.set_xlabel('observed $y_{data}$', fontsize=12)
    ax1.set_ylabel('predicted $y_{fit}$', fontsize=12)

    ax1.set_title(f'{title_prefix}: Predicted vs. Observed', fontsize=12, fontweight='bold')
    
    ax1.grid(True, alpha=0.3, axis='y')

    # Add fit criteria : Chi-squared and AIC and BIC 
    ax1.text(1.35, 0.95, f'$\\chi^2_\\nu$={chi_sq:5.3f}   AIC={aic:5.3f}   BIC={bic:5.3f}',
            transform=ax0.transAxes,
            verticalalignment='top', horizontalalignment='left', fontsize=14)

    #plt.tight_layout()

    # ========================================================================
    # subfig 2: Convergence history
    # ========================================================================
    # fig2, ax2 = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot coefficient evolution
    for i in range(n_coeffs):
        ax2[0].plot(cvg_history[:, 0], cvg_history[:, i+1], '-o', 
                linewidth=2, markersize=4, label=f'$a_{i+1}$')
        # Label final values
        ax2[0].text(cvg_history[-1, 0] * 1.02, cvg_history[-1, i+1], 
                f'{i+1}', fontsize=10)
    
    ax2[0].set_ylabel('Coefficient Values', fontsize=12)
    ax2[0].legend(loc='best', fontsize=10)
    ax2[0].grid(True, alpha=0.3)
    ax2[0].set_title(f'{title_prefix}: Convergence History', fontsize=12, fontweight='bold')
    
    # Plot chi-squared and lambda
    ax2[1].semilogy(cvg_history[:, 0], cvg_history[:, n_coeffs+1], 
                '-o', linewidth=2, markersize=4, label='$\\chi^2_\\nu$')
    ax2[1].semilogy(cvg_history[:, 0], cvg_history[:, n_coeffs+2], 
                '-s', linewidth=2, markersize=4, label='$\\lambda$')
    
    # Label start and end points
    ax2[1].text(cvg_history[0, 0], cvg_history[0, n_coeffs+1], 
            '$\\chi^2_\\nu$', fontsize=12, ha='right')
    ax2[1].text(cvg_history[0, 0], cvg_history[0, n_coeffs+2], 
            '$\\lambda$', fontsize=12, ha='right')
    ax2[1].text(cvg_history[-1, 0], cvg_history[-1, n_coeffs+1], 
            '$\\chi^2_\\nu$', fontsize=12)
    ax2[1].text(cvg_history[-1, 0], cvg_history[-1, n_coeffs+2], 
            '$\\lambda$', fontsize=12)
    
    ax2[1].set_xlabel('Number of Function Evaluations', fontsize=12)
    ax2[1].set_ylabel('$\\chi^2_\\nu$ and $\\lambda$', fontsize=12)
    ax2[1].legend(loc='best', fontsize=10)
    ax2[1].grid(True, alpha=0.3)
    
    #plt.tight_layout()
    
       
    # ========================================================================
    # subfig 3: Histogram of residuals
    # ========================================================================
    # fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    residuals = y_data - y_fit
    ax3.hist(residuals, bins=20, color='blue', alpha=0.7, edgecolor='black')
    
    ax3.set_xlabel('$y_{data} - y_{fit}$', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add statistics to plot
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    ax3.axvline(mean_res, color='r', linestyle='--', linewidth=2, 
              label=f'Mean = {mean_res:.3f}')
    ax3.axvline(mean_res + std_res, color='orange', linestyle=':', linewidth=2)
    ax3.axvline(mean_res - std_res, color='orange', linestyle=':', linewidth=2,
              label=f'Std Dev = {std_res:.3f}')
    ax3.legend(loc='best', fontsize=10)
    
    #plt.tight_layout()

    ax3.set_title(f'{title_prefix}: Histogram of Residuals', 
                fontsize=12, fontweight='bold')
    fig.suptitle(f'{title_prefix}') 

    plt.show()

