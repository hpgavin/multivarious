import numpy as np
import matplotlib.pyplot as plt


def plot_cvg_hst(cvg_hst, x_opt, figNo=1002, clr=None):
    """
    plot_cvg_hst(cvg_hst, x_opt, figNo, clr)
    Plot the convergence history for a solution computed by ORS, NMS, or SQP
    
    INPUT
    =====
    cvg_hst : a record of p, f, and g as returned by ors, nms, or sqp
    x_opt   : the optimal design variables computed by ors, nms, or sqp
    figNo   : the figure number used for plotting, default figNo = 1002
    clr     : colormap for subplot 2
    
    HP Gavin, Duke Univ., 2013-03-01, 2018-03-08, 2020-01-15
    Translated to Python: 2025-12-01
    """
    
    if clr is None:
        clr = plt.cm.tab10(np.linspace(0, 1, 10))
    
    maxiter = cvg_hst.shape[1]  # number of iterations
    n = len(x_opt)              # number of design variables
    
    if maxiter > 100:
        pltstr = '-'
    else:
        pltstr = '-o'
    
    # Extend colormap if needed
    while n > len(clr):
        clr = np.vstack([clr, clr])
    
    fc = cvg_hst[n+2, :]  # function count
    lw = 3                # line width
    ms = 6                # marker size
    
    # Figure for convergence criteria
    plt.figure(figNo + 1)
    plt.clf()
    
    # Plot objective function convergence criterion
    plt.subplot(211)
    if (np.max(cvg_hst[n+4, :]) > 100 * np.min(cvg_hst[n+4, :]) and 
        np.min(cvg_hst[n+3, :]) > 0):
        plt.semilogy(fc, cvg_hst[n+4, :], pltstr, linewidth=lw, markersize=ms)
    else:
        plt.plot(fc, cvg_hst[n+4, :], pltstr, linewidth=lw, markersize=ms)
    plt.ylabel('F convergence')
    
    # Plot design variable convergence criterion
    plt.subplot(212)
    if (np.max(cvg_hst[n+3, :]) > 100 * np.min(cvg_hst[n+3, :]) and 
        np.min(cvg_hst[n+3, :]) > 0):
        plt.semilogy(fc, cvg_hst[n+3, :], pltstr, linewidth=lw, markersize=ms)
    else:
        plt.plot(fc, cvg_hst[n+3, :], pltstr, linewidth=lw, markersize=ms)
    plt.ylabel('X convergence')
    plt.xlabel('function evaluations')
    
    # Main convergence history figure
    plt.figure(figNo)
    plt.clf()
    
    # Plot objective function convergence
    plt.subplot(311)
    cmin = np.min(cvg_hst[n, :])
    cmax = np.max(cvg_hst[n, :])
    rnge = cmax - cmin
    
    if (cmax > 100 * cmin and cmin + 0.01 * rnge > 0 and cmin > 0):
        plt.semilogy(fc, cvg_hst[n, :], pltstr, linewidth=lw, markersize=ms)
    else:
        plt.plot(fc, cvg_hst[n, :], pltstr, linewidth=lw, markersize=ms)
    plt.ylabel('objective   f_A')
    plt.title(f'f_opt = {cvg_hst[n, maxiter-1]:11.4e}         '
              f'max(g_opt) = {cvg_hst[n+1, maxiter-1]:11.4e}')
    
    # Plot design variable convergence
    plt.subplot(312)
    pmin = np.min(cvg_hst[0:n, :])
    pmax = np.max(cvg_hst[0:n, :])
    rnge = pmax - pmin
    
    if (pmax / pmin > 100 and pmin - 0.1 * rnge > 0):
        plt.plot(fc, cvg_hst[0:n, :].T, pltstr, linewidth=lw, markersize=ms)
    else:
        plt.plot(fc, cvg_hst[0:n, :].T, pltstr, linewidth=lw, markersize=ms)
    plt.ylabel('variables')
    
    # Plot max constraint convergence
    plt.subplot(313)
    gmin = np.min(cvg_hst[n+1, :])
    gmax = np.max(cvg_hst[n+1, :])
    rnge = gmax - gmin
    
    if (gmax / (gmin + 0.01) > 100 and gmin - 0.1 * rnge > 0):
        plt.plot(fc, cvg_hst[n+1, :], pltstr, linewidth=lw, markersize=ms)
    else:
        plt.plot(fc, cvg_hst[n+1, :], pltstr, linewidth=lw, markersize=ms)
    plt.ylabel('max(constraints)')
    plt.xlabel('function evaluations')
    
    plt.tight_layout()

# ------------------------------------------------------------ plot_cvg_hst.py