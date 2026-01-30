#! /usr/bin/env -S python3 -i
"""
Linear Elastic Response Spectrum calculation for 2D ground motion.

H.P. Gavin, Duke University, Civil and Environmental Engineering
"""

import numpy as np
import matplotlib.pyplot as plt
from multivarious.lti import lsym
from multivarious.utl import format_plot


def lers_2d(ax, ay, t, g, Tn, zz, method='SRSS', fig_no=0, save_plot='False'):
    """
    Compute linear elastic response spectrum for 2D ground motion.
    
    Computes response spectrum at a set of natural periods Tn and damping 
    ratio zz for input acceleration records ax and ay at times t.
    
    Parameters
    ----------
    ax : array_like
        Input x-axis acceleration values (row vector)
    ay : array_like
        Input y-axis acceleration values (row vector)
    t : array_like
        Time values for acceleration records (row vector)
    g : float
        Gravitational acceleration (m/s² or in/s²)
    Tn : array_like
        Natural periods at which to compute response (s)
    zz : float
        Damping ratio (dimensionless, e.g., 0.05 for 5%)
    method : str, optional
        Combination method: 'SRSS' (default) or 'GM'
        'SRSS' = Square Root of Sum of Squares of dx and dy
        'GM' = Geometric Mean of max|dx| and max|dy|
    fig_no : int, optional
        Figure number for plotting. Default: 0 (no plotting)
    save_plot : boolean, optional
        save the plot
    
    Returns
    -------
    PSA : ndarray
        Pseudo-spectral acceleration at periods Tn
        PSA = (peak displacement response) × (2π/T)²
    SD : ndarray
        Spectral displacement (peak displacement response)
    
    Notes
    -----
    The function sets up nT parallel LTI state-space models, one for each
    natural period, where each oscillator has the form:
        ẍ + 2ζωₙẋ + ωₙ²x = -aₘ(t)
    
    Examples
    --------
    >>> Tn = np.array([0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    >>> zz = 0.05  # 5% damping
    >>> g = 9.81   # m/s²
    >>> PSA, SD = lers_2d(ax, ay, t, g, Tn, zz, method='SRSS', fig_no=3)
    """
    
    # Convert inputs to numpy arrays
    ax = np.asarray(ax).flatten()
    ay = np.asarray(ay).flatten()
    t  = np.asarray(t).flatten()
    Tn = np.asarray(Tn).flatten()
    
    # Natural frequencies
    wn = 2 * np.pi / Tn
    
    nT = len(Tn)  # number of natural period values
    
    # Set up nT parallel LTI state space models, one for each natural period
    A = np.zeros((2*nT, 2*nT))
    B = np.zeros((2*nT, 1))
    C = np.zeros((nT, 2*nT))
    D = np.zeros((nT, 1))
    
    Bi = np.array([[0], [-1]])       # ground motion accel input matrix
    Ci = np.array([[1, 0]])          # displacement response output matrix
    
    # Assemble the state-space model of parallel systems
    for i in range(nT):
        Ai = np.array([[0, 1],
                       [-wn[i]**2, -2*zz*wn[i]]])
        
        A[2*i:2*i+2, 2*i:2*i+2] = Ai
        B[2*i:2*i+2, :] = Bi
        C[i, 2*i:2*i+2] = Ci
    
    # Simulate responses of nT parallel state space models
    dx = lsym(A, B, C, D, ax, t)
    
    if np.dot(ay, ay) > 0:
        dy = lsym(A, B, C, D, ay, t)
    else:
        dy = np.zeros_like(dx)
    
    # Extract peak displacement responses for each natural period
    method = method.upper()
    
    if method == 'GM':
        # Geometric mean of max|dx| and max|dy|
        dx_max = np.max(np.abs(dx), axis=1)
        dy_max = np.max(np.abs(dy), axis=1)
        SD = np.sqrt(dx_max * dy_max)
        
    elif method == 'SRSS':
        # Square root of sum of squares of dx and dy
        d = np.sqrt(dx**2 + dy**2)
        max_D = np.max(d, axis=1)
        SD = max_D
        
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'SRSS' or 'GM'.")
    
    # Pseudo-spectral acceleration
    PSA = SD * wn**2 / g
    
    # Plots
    if fig_no > 0:
        _plot_response_spectra(Tn, PSA, SD, g, fig_no, save_plot)
    
    return PSA, SD


def _plot_response_spectra(Tn, PSA, SD, g, fig_no, save_plot):
    """
    Create response spectrum plots.
    
    Parameters
    ----------
    Tn : ndarray
        Natural periods
    PSA : ndarray
        Pseudo-spectral acceleration
    SD : ndarray
        Spectral displacement
    g : float
        Gravitational acceleration
    fig_no : int
        Starting figure number
    """
    plt.ion()  # Interactive mode: on
    
    format_plot(font_size=15, line_width=2, marker_size=7)

    # Figure 1: Response spectra SD vs Tn and PSA vs Tn
    plt.figure(fig_no, figsize=(5,5))
    plt.clf()
    plt.plot(Tn, PSA, '-o', label='PSA, g')
    plt.plot(Tn, SD, '-o', label='SD, m')
    plt.legend()
    plt.xlabel(r'natural period, $T_n$, s')
    plt.ylabel('response spectra')
    plt.grid(True, alpha=0.3)
    
    # Figure 2: Response spectra PSA vs SD with constant period lines
    maxSD = np.max(SD)
    maxPSA = np.max(PSA)
    setTn = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
    
    plt.figure(fig_no + 1, figsize=(5,5))
    plt.clf()
    
    # Draw constant period lines
    for Tn_val in setTn:
        omega = 2 * np.pi / Tn_val
        cSD = min(maxSD, maxPSA * g / omega**2)
        if cSD < maxSD:
            cPSA = maxPSA
        else:
            cPSA = omega**2 * cSD / g
        
        plt.plot([0, cSD], [0, cPSA], '-k', linewidth=0.5)
        plt.text(cSD, cPSA, f'{Tn_val:4.1f} s', fontsize=10)
    
    # Plot response spectrum
    plt.plot(SD, PSA, '-o', linewidth=3)
    
    plt.axis([0, 1.05*maxSD, 0, 1.05*maxPSA])
    plt.xlabel(r'spectral displacement, $S_D$, m')
    plt.ylabel(r'pseudo spectral acceleration, $S_A$, g')
    plt.grid(True, alpha=0.3)
    
    plt.show()

    if save_plot: 
        plt.savefig(f'lers-2d-{fig_no}.pdf', dpi=150)

# ------------------------------------------------------------- lers_2d.py
