"""
plot_cvg_hst.py - Plot Optimization Convergence History

Plot the convergence history for a solution computed by ORSopt, NMAopt, SQPopt
or similar optimization algorithms.

Translation from MATLAB to Python by Claude, 2025-11-18
Original by HP Gavin, Duke Univ., 2013-03-01, 2018-03-08, 2020-01-15
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_cvg_hst(cvg_hst, v_opt, opts=[1,np.nan,np.nan,np.nan], fig_num=1000, clr=None, save_plots=False):
    """
    Plot the convergence history for optimization solutions.
    
    Creates two figures:
    - Figure fig_num+1: F and V convergence criteria
    - Figure fig_num+2: Objective function, design variables, and constraints
    
    Parameters
    ----------
    cvg_hst : ndarray, shape (n+5, max_iter)
        Convergence history matrix where:
        - Rows 0:n contain design variable history
        - Row n contains objective function values
        - Row n+1 contains max constraint values
        - Row n+2 contains function count
        - Row n+3 contains V convergence criterion
        - Row n+4 contains F convergence criterion
    v_opt : ndarray, shape (n,)
        Optimal design variables computed by optimizer
    opts : array-like, optimization options
        opts[1] = tol_v - convergence tolerence on design variables 
        opts[2] = tol_f - convergence tolerence on design objective
        opts[3] = tol_g - convergence tolerence on design constraings
    fig_num : int, optional
        Figure number for plotting (default: 1002)
    clr : ndarray, optional
        Colormap for plotting (default: matplotlib tab10 colormap)
    
    Returns
    -------
    None
        Displays and saves matplotlib figures
    """


    msg   = opts[0]
    tol_v = opts[1]
    tol_f = opts[2]
    tol_g = opts[3]

    #plt.rcParams['text.usetex'] = True # Set to True if LaTeX is installed

    interactive = True        # Enable interactive mode for matplotlib
    
    # Convert inputs to numpy arrays
    cvg_hst = np.asarray(cvg_hst)
    v_opt = np.asarray(v_opt).flatten()
    
    # Extract dimensions
    max_iter = cvg_hst.shape[1]  # Number of iterations
    n = len(v_opt)               # Number of design variables
    max_feval = cvg_hst[n+2,-1]  # largest number of function counts
    
    # Determine plot style based on number of iterations
    if max_iter > 100:
        pltstr = '-'  # Line only
        marker = None
    else:
        pltstr = '-o'  # Line with markers
        marker = 'o'
    
    # Set up colormap
    if clr is None:
        # Use tab10 colormap and repeat if necessary
        base_colors = cm.tab10(np.linspace(0, 1, 10))
        clr = base_colors
        while n > len(clr):
            clr = np.vstack([clr, base_colors])
    
    # Extract key data
    fc = cvg_hst[n+2, :]  # Function count
    
    if fig_num:  # Make plots
        plt.ion() # interactive plot mode: on
        
        # ====================================================================
        # FIGURE fig_num+1: Convergence Criteria
        # ====================================================================
        fig1 = plt.figure(fig_num+1, figsize=(10, 8))
        fig1.clf()
        
        # Subplot 1: F convergence criterion
        plt.subplot(2, 1, 1)
        f_conv = cvg_hst[n+4, :]
        
        # Auto log-scale detection
        if np.max(f_conv) > 1e2 * np.min(f_conv) and np.min(f_conv) > 0:
            if marker:
                plt.semilogy([fc[0],fc[-1]], tol_f*np.array([1,1]), '--g', linewidth=1)
                plt.semilogy(fc, f_conv, pltstr)
            else:
                plt.semilogy([fc[0],fc[-1]], tol_f*np.array([1,1]), '--g', linewidth=1)
                plt.semilogy(fc, f_conv)
        else:
            if marker:
                plt.plot([fc[0],fc[-1]], tol_f*np.array([1,1]), '--g', linewidth=1)
                plt.plot(fc, f_conv, pltstr)
            else:
                plt.plot([fc[0],fc[-1]], tol_f*np.array([1,1]), '--g', linewidth=1)
                plt.plot(fc, f_conv)
        
        plt.ylabel(r'objective convergence')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: V convergence criterion
        plt.subplot(2, 1, 2)
        v_conv = cvg_hst[n+3, :]
        
        # Auto log-scale detection
        if np.max(v_conv) > (1e2 * np.min(v_conv)) and np.min(v_conv) > 0:
            if marker:
                plt.semilogy([fc[0],fc[-1]], tol_v*np.array([1,1]), '--g', linewidth=1)
                plt.semilogy(fc, v_conv, pltstr)
            else:
                plt.semilogy([fc[0],fc[-1]], tol_v*np.array([1,1]), '--g', linewidth=1)
                plt.semilogy(fc, v_conv)
        else:
            if marker:
                plt.plot([fc[0],fc[-1]], tol_v*np.array([1,1]), '--g', linewidth=1)
                plt.plot(fc, v_conv, pltstr)
            else:
                plt.plot([fc[0],fc[-1]], tol_v*np.array([1,1]), '--g', linewidth=1)
                plt.plot(fc, v_conv)
        
        plt.ylabel(r'variable convergence')
        plt.xlabel('function evaluations')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # ====================================================================
        # FIGURE fig_num: Objective, Variables, and Constraints
        # ====================================================================
        fig2 = plt.figure(fig_num+2, figsize=(10, 10))
        fig2.clf()
        
        # Subplot 1: Objective function convergence
        plt.subplot(3, 1, 1)
        obj_vals = cvg_hst[n, :]
        
        fmin = np.min(obj_vals)
        fmax = np.max(obj_vals)
        rnge = fmax - fmin
        
        # Auto log-scale detection
        if (fmax > (1e2 * fmin) and fmin > 0 and fmin > 0):
            if marker:
                plt.semilogy(fc, obj_vals, pltstr)
            else:
                plt.semilogy(fc, obj_vals)
        else:
            if marker:
                plt.plot(fc, obj_vals, pltstr)
            else:
                plt.plot(fc, obj_vals)
        
        plt.ylabel(r'objective   $f_A$')
        plt.grid(True, alpha=0.3)
        
        # Title with final values
        f_opt = obj_vals[-1]
        g_opt = cvg_hst[n+1, -1]
        plt.title(rf"$f_{{opt}}$ = {f_opt:10.3e}         max($g_{{opt}}$) = {g_opt:10.3e}")
        
        # Subplot 2: Design variable convergence
        plt.subplot(3, 1, 2)
        var_vals = cvg_hst[0:n, :]
        
        pmin = np.min(var_vals)
        pmax = np.max(var_vals)
        rnge = pmax - pmin
        
        # Plot all design variables
        if marker:
            for i in range(n):
                plt.plot(fc, var_vals[i, :], pltstr, color=clr[i])
        else:
            for i in range(n):
                plt.plot(fc, var_vals[i, :], color=clr[i])
        
        plt.ylabel('variables')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Max constraint convergence
        plt.subplot(3, 1, 3)
        constr_vals = cvg_hst[n+1, :]
        
        gmin = np.min(constr_vals)
        gmax = np.max(constr_vals)
        rnge = gmax - gmin
        
        if marker:
            plt.plot([fc[0],fc[-1]], tol_g*np.array([1,1]), '--g', linewidth=1)
            plt.plot(fc, constr_vals, pltstr)
        else:
            plt.plot([fc[0],fc[-1]], tol_g*np.array([1,1]), '--g', linewidth=1)
            plt.plot(fc, constr_vals)
        
        plt.ylabel('max(constraints)')
        plt.xlabel('function evaluations')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()

    # Display plots
    plt.show()
    
    # Save plots 
    if save_plots:
        plt.figure(fig_num+1)
        filename = f'plot_cvg_hst-{fig_num+1:04d}.pdf'
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Saved: {filename}")

        plt.figure(fig_num+2)
        filename = f'plot_cvg_hst-{fig_num+2:04d}.pdf'
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Saved: {filename}")

        if msg > 2:
            plt.figure(fig_num+3)
            filename = f'plot_cvg_hst-{fig_num+3:04d}.pdf'
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            print(f"Saved: {filename}")
 

# Example usage / test
if __name__ == "__main__":
    print("\n" + "="*70)
    print("Testing plot_cvg_hst.py")
    print("="*70 + "\n")
    
    # Create synthetic convergence history data
    n = 3  # Number of design variables
    max_iter = 50  # Number of iterations
    
    # Initialize convergence history array
    cvg_hst = np.zeros((n+5, max_iter))
    
    # Generate synthetic data
    fc = np.arange(1, max_iter + 1)  # Function count
    
    # Design variables converging to [1, 2, 3]
    for i in range(n):
        cvg_hst[i, :] = (i + 1) + 2 * np.exp(-0.1 * fc) * np.cos(0.3 * fc)
    
    # Objective function decreasing exponentially
    cvg_hst[n, :] = 100 * np.exp(-0.15 * fc) + 1.0
    
    # Max constraint decreasing (becoming feasible)
    cvg_hst[n+1, :] = 10 * np.exp(-0.2 * fc)
    
    # Function count
    cvg_hst[n+2, :] = fc
    
    # V convergence criterion
    cvg_hst[n+3, :] = 10 * np.exp(-0.12 * fc)
    
    # F convergence criterion
    cvg_hst[n+4, :] = 50 * np.exp(-0.18 * fc)
    
    # Optimal design variables
    v_opt = np.array([1.0, 2.0, 3.0])
    
    # Call the plotting function
    plot_cvg_hst(cvg_hst, v_opt, fig_num=1000)

