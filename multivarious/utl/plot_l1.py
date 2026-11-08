#! /usr/bin/env -S python3 -i

"""
plot_l1.py - Visualization of L1 Regularization Results

Plot results from the fit of a linear model via L1 regularization using l1

Translation from MATLAB by Claude, 2025-10-24
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_l1(B, c, y, cvg_hst, alfa, w, fig_num=1, save_plots=False, var_name=None):
    """
    Plot results from L1 regularization fit.
    
    Creates three figures:
    1. Coefficient comparison (OLS vs L1)
    2. Data fit comparison
    3. Convergence history
    
    Input
    ======
    B : ndarray, shape (m, n)
        Basis matrix (design matrix)
    c : ndarray, shape (n,)
        L1-fitted coefficients
    y : ndarray, shape (m,)
        Data vector
    cvg_hst : ndarray, shape (5*n+2, n_iter)
        Convergence history from l1
    alfa : float
        Final L1 regularization parameter
    w : float
        Weighting parameter used
    fig_num : int, optional
        Starting figure number (default: 1)
    save_plots : boolean, optional (default: False)
        Save figures 
    var_name : character string for the output variable
    
    Output
    ======
    None
        Displays matplotlib figures
    """

    plt.ion() # plotting interactive mode: on
    
    # Ensure y is 1D
    y = np.asarray(y).flatten()
    c = np.asarray(c).flatten()
    
    # Extract dimensions
    n_total, max_iter = cvg_hst.shape
    n = (n_total - 2) // 5  # Number of coefficients
    m = len(y)
    
    # OLS model coefficients for comparison
    c0 = np.linalg.lstsq(B, y, rcond=None)[0]
    y0 = B @ c0  # OLS model prediction
    y1 = B @ c   # L1 model prediction
    
    # Compute normalized errors
    err_norm_0 = np.linalg.norm(y0 - y) / (m - n)
    err_norm_1 = np.linalg.norm(y1 - y) / (m - n)
    
    # Print results
    print("\n" + "="*70)
    print("L1 Regularization Results")
    print("="*70)
    print(f"\nOLS error (alpha=0):        {err_norm_0:.6f}")
    print(f"L1 Error  (alpha={alfa:.5f}): {err_norm_1:.6f}")
    print(f"\nOLS Coefficients (alpha=0, w={w:.1f}):")
    print(c0)
    print(f"\nL1 Coefficients  (alpha={alfa:.5f}, w={w:.1f}):")
    print(c)
    print("="*70 + "\n")
    
    vn = var_name or ''
    # Figure 1: Coefficient Comparison
    if fig_num > 9:
        plt.figure(fig_num, figsize=(10, 6))
        plt.clf()
        
        indices = np.arange(1, n + 1)
        plt.plot(indices, c0, '+r', markersize=20, label=f'α = 0 (OLS)')
        plt.plot(indices, c, 'o', color=[0, 0.8, 0], label=f'α = {alfa:.5f}, w = {w:.1f}')
        plt.axhline(0, color='k', linestyle='--', linewidth=1)
        plt.xlabel(r'Coefficient index, $i$')
        plt.ylabel(r'Coefficients, $c_i$')
        plt.title(rf'{vn}  Coefficient Comparison: OLS vs $L_1$')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
    # Figure 2: Data Fit Comparison
    if fig_num > 99:
        plt.figure(fig_num + 1, figsize=(10, 6))
        plt.clf()
        
        t = np.arange(1, m + 1)
        plt.plot(t, y, 'ok', label='Data', markerfacecolor='none')
        plt.plot(t, y0, 'or', label='α = 0 (OLS)')
        plt.plot(t, y1, 'o', color=[0, 0.8, 0], label=f'α = {alfa:.5f}, w = {w:.0f}')
        plt.xlabel(rf'Data index, $i$')
        plt.ylabel(rf'$y_i$')
        plt.title(rf'{vn} $L_1$ Model Fit Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
    # Figure 3: Convergence History
    if fig_num > 999:
        plt.figure(fig_num + 2, figsize=(10, 6))
        plt.clf()
        
        y_labels = ['$c$', '$p$', '$q$', '$\\mu$', '$\\nu$', '$\\alpha$ and $L_2$ error']
        iterations = np.arange(1, max_iter + 1)
        
        for ii in range(5):
            ax = plt.subplot(6, 1, ii + 1)
            start_idx = n * ii
            end_idx = n * (ii + 1)
            plt.plot(iterations, cvg_hst[start_idx:end_idx, :].T)
            plt.ylabel(y_labels[ii])
            plt.grid(True, alpha=0.3)
            ax.set_xticklabels([])

            if ii == 0:
                plt.title(rf'{vn} $L_1$ Convergence History')
        
        # Last subplot: alfa and error
        plt.subplot(6, 1, 6)
        plt.semilogy(iterations, cvg_hst[5*n, :], '-g', label='$\\alpha$')
        plt.semilogy(iterations, cvg_hst[5*n+1, :], '-k', label='$L_2$ error')
        plt.ylabel(y_labels[5])
        plt.xlabel(r'$L_1$ iteration number')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout(pad=0.2, h_pad=0.2, w_pad=0.2 )

    if fig_num > 9:
        plt.show()
     
    # Save figures
    if save_plots and fig_num > 9:
        plt.figure(fig_num)
        plt.savefig(f'L1_coefficients-{fig_num}.pdf', bbox_inches='tight')
        print("    L1_coefficients-{fig_num}.pdf saved.")
    
    if save_plots and fig_num > 99:
        plt.figure(fig_num+1)
        plt.savefig(f'l1_comparison-{fig_num+1}.pdf', bbox_inches='tight')
        print("    L1_coefficients-{fig_num+1}.pdf saved.")
    
    if save_plots and fig_num > 999:
        plt.figure(fig_num+2)
        plt.savefig(f'L1_convergence-{fig_num+2}.pdf', bbox_inches='tight')
        print("    L1_coefficients-{fig_num+2}.pdf saved.")
    

# Test function
if __name__ == '__main__':
    """
    Test plot_l1 with sample data
    """
    from multivarious.fit import l1
    from multivarious.utl import format_plot
    
    print("\nTesting plot_l1.py")
    print("=" * 70)
    
    # Generate test data
    np.random.seed(42)
    x = np.linspace(-1.2, 1.2, 49)
    m = len(x)
    
    # Power polynomial basis
    B = np.column_stack([x**i for i in range(8)])
    
    # Generate noisy data
    noise = 0.15 * np.random.randn(m)
    y = 1 - x**2 + np.sin(np.pi * x) + noise
    
    # Fit with L1
    w = 1.0
    alfa = 0.1
    c, mu, nu, cvg_hst = l1(B, y, alfa, w)
    
    # Get final alfa from convergence history
    alfa_final = cvg_hst[-2, -1]
    
    # Create plots
    format_plot(font_size=14, line_width=3, marker_size=5)
    plot_l1(B, c, y, cvg_hst, alfa_final, w, fig_num=7000)
   
    print("\n" + "=" * 70)
    print("plot_l1 test completed successfully!")
    print("=" * 70)

