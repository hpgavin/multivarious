"""
plot_opt_surface.py - 3D Surface Plot of Objective Function

Draw a surface plot of f(v) vs. v[i], v[j], where all other values in v
are held constant. Useful for visualizing optimization landscapes and 
convergence paths.

Translation from MATLAB to Python by Claude, 2025-11-19
Original by H.P. Gavin, Civil & Environ. Eng'g, Duke Univ.
Updated 2015-09-29, 2016-03-23, 2016-09-12, 2020-01-20, 2021-12-31, 2025-01-26
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D


def plot_opt_surface(func, v, v_lb, v_ub, options, consts=None, fig_num=1):
    """
    Draw a surface plot of objective function f(v) vs. v[i], v[j].

    Creates a 3D mesh surface showing the objective function landscape over
    two design variables while holding all others constant. Marks the initial
    point and the grid minimum on the surface.

    Parameters
    ----------
    func : callable
        Function to be optimized with signature:
        [objective, constraints] = func(v, consts)
        Returns objective value (float) and constraint values (array)
    v : ndarray, shape (n,)
        Vector of initial parameter values (column vector)
    v_lb : ndarray, shape (n,)
        Lower bounds on permissible parameter values
    v_ub : ndarray, shape (n,)
        Upper bounds on permissible parameter values
    options : ndarray or list
        options[0] = 3 (reserved for surface plotting flag)
        options[3] = tol_g - tolerance on constraint functions
        options[5] = penalty - penalty factor on constraint violations
        options[6] = q - exponent on constraint violations
        options[10] = i - 1st index for plotting (0-based in Python!)
        options[11] = j - 2nd index for plotting (0-based in Python!)
        options[12] = Ni - number of points in 1st dimension
        options[13] = Nj - number of points in 2nd dimension
    consts : optional
        Optional vector of constants to be passed to func(v, consts)
    fig_num : int, optional
        Figure number for the plot (default: 1)

    Returns
    -------
    fmin : float
        Minimum value of the meshed surface data
    fmax : float
        Maximum value of the meshed surface data (clipped for visualization)
    ax :
        To allow other functions to update the figure

    Notes
    -----
    - Handles constraint violations by adding penalties to objective
    - Automatically scales z-axis to avoid extreme outliers
    - Marks initial point (green) and grid minimum (red) on surface
    - If penalty <= 0 and constraints violated, point is set to NaN
    """

    #plt.rcParams['text.usetex'] = True # Set to True if LaTeX is installed
    
    interactive = True # Enable interactive mode for matplotlib
        
    if interactive:
        plt.ion() # plot interactive mode: on

    # Convert inputs to numpy arrays
    v = np.asarray(v).flatten()
    v_lb = np.asarray(v_lb).flatten()
    v_ub = np.asarray(v_ub).flatten()
    options = np.asarray(options)

    # Extract options (note: MATLAB is 1-indexed, Python is 0-indexed)
    tol_g = options[3]      # Constraint tolerance
    penalty = options[5]    # Penalty factor
    q = options[6]          # Penalty exponent
    i = int(options[10])    # 1st variable index (0-based)
    j = int(options[11])    # 2nd variable index (0-based)
    Ni = int(options[12])   # Number of points in 1st dimension
    Nj = int(options[13])   # Number of points in 2nd dimension

    # Store initial v values
    v_init = v.copy()

    # Create grid for the two variables
    v_i = np.linspace(v_lb[i], v_ub[i], Ni)
    v_j = np.linspace(v_lb[j], v_ub[j], Nj)

    # Initialize objective function mesh
    f_mesh = np.full((Ni, Nj), np.nan)

    # Evaluate objective function on grid
    for ii in range(Ni):
        for jj in range(Nj):
            # Set the two variables to grid values
            v[i] = v_i[ii]
            v[j] = v_j[jj]

            # Evaluate function
            if consts is not None:
                f, g = func(v, consts)
            else:
                f, g = func(v)

            # Ensure g is an array
            g = np.asarray(g).flatten()

            # Add penalty for constraint violations
            constraint_penalty = np.sum(g * (g > tol_g)) ** q
            f_mesh[ii, jj] = f + penalty * constraint_penalty

            # If no penalty and constraints violated, set to NaN
            if penalty <= 0 and np.max(g) > tol_g:
                f_mesh[ii, jj] = np.nan

    # Compute RMS and set z-axis limits
    frms = np.sqrt(np.nansum(f_mesh**2))
    fmin = 0.9 * np.nanmin(f_mesh)
    fmax = min(np.nanmax(f_mesh), 5 * frms)

    # Handle case where all values are NaN
    if np.isnan(fmin) or np.isnan(fmax):
        fmin = 0.0
        fmax = 1.0

    # Find grid minimum location
    if np.all(np.isnan(f_mesh)):
        ii_min, jj_min = 0, 0
    else:
        min_val = np.nanmin(f_mesh)
        ii_min, jj_min = np.where(f_mesh == min_val)
        ii_min = ii_min[0]  # Take first if multiple minima
        jj_min = jj_min[0]

    # Create the plot
    fig = plt.figure(fig_num, figsize=(12, 9))
    fig.clf()
    ax = fig.add_subplot(111, projection='3d')

    # Create meshgrid for plotting
    V_i, V_j = np.meshgrid(v_i, v_j, indexing='ij')

    cmap = cm.rainbow# set color scheme here
    norm = plt.Normalize(f_mesh.min(), f_mesh.max())

    """
    # Alternative #1: Plot the surface mesh - all lines the same color
    ax.plot_wireframe(V_i, V_j, f_mesh, linewidth=1.5, alpha=0.7)
    """

    # Alternative #2: use plot_surface for filled surface
    ax.plot_surface(V_i, V_j, f_mesh, cmap='rainbow', alpha=0.5, linewidth=0.5, edgecolor='k')

    """
    # Alternative #3: Plot the surface mesh - lines colored separately
    # Plot lines along the rows
    for ii in range(V_i.shape[0]):
        vis = V_i[ii, :]
        vjs = V_j[ii, :]
        fs  = f_mesh[ii, :]
        for jj in range(len(vis)-1):
            z_avg = (fs[jj] + fs[jj+1]) / 2
            ax.plot(vis[jj:jj+2], vjs[jj:jj+2], fs[jj:jj+2], color=cmap(norm(z_avg)))

    # Plot lines along the columns
    for jj in range(V_j.shape[1]):
        vis = V_i[:, jj]
        vjs = V_j[:, jj]
        fs = f_mesh[:, jj]
        for ii in range(len(vis)-1):
            z_avg = (fs[ii] + fs[ii+1]) / 2
            ax.plot(vis[ii:ii+2], vjs[ii:ii+2], fs[ii:ii+2], color=cmap(norm(z_avg)))

    plt.show()
    """

    # Set labels with LaTeX-style formatting
    ax.set_xlabel(f'$v_{{{i+1}}}$', fontsize=14)
    ax.set_ylabel(f'$v_{{{j+1}}}$', fontsize=14)
    ax.set_zlabel(f'objective   $f_A(v_{{{i+1}}}, v_{{{j+1}}})$', fontsize=14)

    # Set axis limits
    ax.set_xlim([np.min(v_i), np.max(v_i)])
    ax.set_ylim([np.min(v_j), np.max(v_j)])
    ax.set_zlim([fmin, fmax])

    # Plot initial point (green circle)
    v_plot = v_init.copy()
    if consts is not None:
        f_init, g_init = func(v_init, consts)
    else:
        f_init, g_init = func(v_init)

    g_init = np.asarray(g_init).flatten()
    constraint_penalty_init = np.sum(g_init * (g_init > tol_g)) ** q
    fa_init = f_init + penalty * constraint_penalty_init

    # Plot optimization end point (red circle)
    ax.plot( [v_init[i]], [v_init[j]], fa_init,
            'o', alpha=1.0, color='green', markersize=22, markeredgewidth=3,
            markerfacecolor='green', markeredgecolor='darkgreen',
            label='Initial point')

    # Plot optimization end point (red circle)
    #ax.plot([v_i[ii_min]], [v_j[jj_min]], [f_mesh[ii_min, jj_min]],
    #        'o', alpha=1.0, color='red', markersize=22, markeredgewidth=3,
    #        markerfacecolor='red', markeredgecolor='darkred',
    #        label='Grid minimum')

    # Add legend
    #ax.legend(fontsize=11)

    # Improve viewing angle
    ax.view_init(elev=40, azim=-130)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Tight layout
    plt.tight_layout()

    return fmin, fmax, ax

