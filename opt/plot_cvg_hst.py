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


def plot_cvg_hst(cvg_hst, v_opt, figNo=1002, clr=None):
    """
    Plot the convergence history for optimization solutions.
    
    Creates two figures:
    - Figure figNo+1: F and X convergence criteria
    - Figure figNo: Objective function, design variables, and constraints
    
    Parameters
    ----------
    cvg_hst : ndarray, shape (n+5, maxiter)
        Convergence history matrix where:
        - Rows 0:n contain design variable history
        - Row n contains objective function values
        - Row n+1 contains max constraint values
        - Row n+2 contains function count
        - Row n+3 contains X convergence criterion
        - Row n+4 contains F convergence criterion
    v_opt : ndarray, shape (n,)
        Optimal design variables computed by optimizer
    figNo : int, optional
        Figure number for plotting (default: 1002)
    clr : ndarray, optional
        Colormap for plotting (default: matplotlib tab10 colormap)
    
    Returns
    -------
    None
        Displays and saves matplotlib figures
    """
    
    # Convert inputs to numpy arrays
    cvg_hst = np.asarray(cvg_hst)
    v_opt = np.asarray(v_opt).flatten()
    
    # Extract dimensions
    maxiter = cvg_hst.shape[1]  # Number of iterations
    n = len(v_opt)               # Number of design variables
    
    # Determine plot style based on number of iterations
    if maxiter > 100:
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
    fc = cvg_hst[n + 2, :]  # Function count
    lw = 3                   # Line width
    ms = 6                   # Marker size
    
    if figNo:  # Make plots
        # Set up plot formatting
        plt.ion() # interactive plot mode: on
        plt.rcParams['font.size'] = 14
        plt.rcParams['lines.linewidth'] = 2
        #plt.rcParams['axes.linewidth'] = 1
        
        # ====================================================================
        # FIGURE figNo+1: Convergence Criteria
        # ====================================================================
        fig1 = plt.figure(figNo + 1, figsize=(10, 8))
        fig1.clf()
        
        # Subplot 1: F convergence criterion
        plt.subplot(2, 1, 1)
        f_conv = cvg_hst[n + 4, :]
        
        # Auto log-scale detection
        if np.max(f_conv) > 100 * np.min(f_conv) and np.min(f_conv) > 0:
            if marker:
                plt.semilogy(fc, f_conv, pltstr, linewidth=lw, markersize=ms)
            else:
                plt.semilogy(fc, f_conv, linewidth=lw)
        else:
            if marker:
                plt.plot(fc, f_conv, pltstr, linewidth=lw, markersize=ms)
            else:
                plt.plot(fc, f_conv, linewidth=lw)
        
        plt.ylabel(r'$F$ convergence')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: X convergence criterion
        plt.subplot(2, 1, 2)
        x_conv = cvg_hst[n + 3, :]
        
        # Auto log-scale detection
        if np.max(x_conv) > 100 * np.min(x_conv) and np.min(x_conv) > 0:
            if marker:
                plt.semilogy(fc, x_conv, pltstr, linewidth=lw, markersize=ms)
            else:
                plt.semilogy(fc, x_conv, linewidth=lw)
        else:
            if marker:
                plt.plot(fc, x_conv, pltstr, linewidth=lw, markersize=ms)
            else:
                plt.plot(fc, x_conv, linewidth=lw)
        
        plt.ylabel(r'$X$ convergence')
        plt.xlabel('function evaluations')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename1 = f'plot_cvg_hst-{figNo+1}.png'
        #plt.savefig(f'/mnt/user-data/outputs/{filename1}', dpi=150, bbox_inches='tight')
        #print(f"Saved: {filename1}")
        
        # ====================================================================
        # FIGURE figNo: Objective, Variables, and Constraints
        # ====================================================================
        fig2 = plt.figure(figNo, figsize=(10, 10))
        fig2.clf()
        
        # Subplot 1: Objective function convergence
        plt.subplot(3, 1, 1)
        obj_vals = cvg_hst[n, :]
        
        cmin = np.min(obj_vals)
        cmax = np.max(obj_vals)
        rnge = cmax - cmin
        
        # Auto log-scale detection
        if (cmax > 100 * cmin and cmin + 0.01 * rnge > 0 and cmin > 0):
            if marker:
                plt.semilogy(fc, obj_vals, pltstr, linewidth=lw, markersize=ms)
            else:
                plt.semilogy(fc, obj_vals, linewidth=lw)
        else:
            if marker:
                plt.plot(fc, obj_vals, pltstr, linewidth=lw, markersize=ms)
            else:
                plt.plot(fc, obj_vals, linewidth=lw)
        
        plt.ylabel(r'objective   $f_A$')
        plt.grid(True, alpha=0.3)
        
        # Title with final values
        f_opt = obj_vals[-1]
        g_opt = cvg_hst[n + 1, -1]
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
                plt.plot(fc, var_vals[i, :], pltstr, 
                        color=clr[i], linewidth=lw, markersize=ms)
        else:
            for i in range(n):
                plt.plot(fc, var_vals[i, :], 
                        color=clr[i], linewidth=lw)
        
        plt.ylabel('variables')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Max constraint convergence
        plt.subplot(3, 1, 3)
        constr_vals = cvg_hst[n + 1, :]
        
        gmin = np.min(constr_vals)
        gmax = np.max(constr_vals)
        rnge = gmax - gmin
        
        # Auto log-scale detection
        if (gmax / (gmin + 0.01) > 100 and gmin - 0.1 * rnge > 0):
            if marker:
                plt.plot(fc, constr_vals, pltstr, linewidth=lw, markersize=ms)
            else:
                plt.plot(fc, constr_vals, linewidth=lw)
        else:
            if marker:
                plt.plot(fc, constr_vals, pltstr, linewidth=lw, markersize=ms)
            else:
                plt.plot(fc, constr_vals, linewidth=lw)
        
        plt.ylabel('max(constraints)')
        plt.xlabel('function evaluations')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        #filename2 = f'plot_cvg_hst-{figNo}.png'
        #plt.savefig(f'/mnt/user-data/outputs/{filename2}', dpi=150, bbox_inches='tight')
        #print(f"Saved: {filename2}")


# Example usage / test
if __name__ == "__main__":
    print("\n" + "="*70)
    print("Testing plot_cvg_hst.py")
    print("="*70 + "\n")
    
    # Create synthetic convergence history data
    n = 3  # Number of design variables
    maxiter = 50  # Number of iterations
    
    # Initialize convergence history array
    cvg_hst = np.zeros((n + 5, maxiter))
    
    # Generate synthetic data
    fc = np.arange(1, maxiter + 1)  # Function count
    
    # Design variables converging to [1, 2, 3]
    for i in range(n):
        cvg_hst[i, :] = (i + 1) + 2 * np.exp(-0.1 * fc) * np.cos(0.3 * fc)
    
    # Objective function decreasing exponentially
    cvg_hst[n, :] = 100 * np.exp(-0.15 * fc) + 1.0
    
    # Max constraint decreasing (becoming feasible)
    cvg_hst[n + 1, :] = 10 * np.exp(-0.2 * fc)
    
    # Function count
    cvg_hst[n + 2, :] = fc
    
    # X convergence criterion
    cvg_hst[n + 3, :] = 10 * np.exp(-0.12 * fc)
    
    # F convergence criterion
    cvg_hst[n + 4, :] = 50 * np.exp(-0.18 * fc)
    
    # Optimal design variables
    v_opt = np.array([1.0, 2.0, 3.0])
    
    # Call the plotting function
    plot_cvg_hst(cvg_hst, v_opt, figNo=1002)
    
    plt.show()
    
    #print("\n" + "="*70)
    #print("plot_cvg_hst test completed successfully!")
    #print("Figures saved to /mnt/user-data/outputs/")
    #print("="*70 + "\n")
