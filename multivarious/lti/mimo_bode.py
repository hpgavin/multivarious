import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from multivarious.lti import abcd_dim

def mimo_bode(A, B, C, D, w=None, dt=None, figno=100, ax='n', leg=None, tol=1e-18):
    """
    Plot the magnitude and phase of the steady-state harmonic response 
    of a MIMO linear dynamic system.
    
    State-space system:
        dx/dt = Ax + Bu  (continuous time, dt=None)
        x[k+1] = Ax[k] + Bu[k]  (discrete time, dt specified)
        y = Cx + Du
    
    Parameters:
        A     : dynamics matrix (n x n)
        B     : input matrix (n x r)
        C     : output matrix (m x n)
        D     : feedthrough matrix (m x r)
        w     : vector of frequencies in rad/s (default: logspace(-2,2,200)*2*pi)
        dt    : sample period for discrete-time systems (default: None for continuous)
        figno : figure number for plotting (default: 100, 0 or None for no plot)
        ax    : axis scaling - 'x'=semilogx, 'y'=semilogy, 'n'=linear, 'b'=loglog (default: 'n')
        leg   : legend labels (default: None)
        tol   : tolerance for SVD-based matrix inversion (default: 1e-18)
    
    Returns:
        mag : magnitude of frequency response (nw x m x r)
        pha : phase of frequency response in degrees (nw x m x r)
        G   : complex frequency response function (nw x m x r)
        
    Reference:
        Krajnik, Eduard, 'A simple and reliable phase unwrapping algorithm,'
        http://www.mathnet.or.kr/mathnet/paper_file/Czech/Eduard/phase.ps
    
    Author: Henri Gavin, Dept. Civil Engineering, Duke University
    """
    
    # Convert to numpy arrays
    A = np.asarray(A)
    B = np.asarray(B)
    C = np.asarray(C)
    D = np.asarray(D)
    
    # Check for compatible dimensions
    n, r, m = abcd_dim(A, B, C, D)
    
    # Default frequency vector
    if w is None:
        w = np.logspace(-2, 2, 200) * 2 * np.pi
    else:
        w = np.asarray(w)
    
    nw = len(w)
    lw = 3  # line width
    
    In = np.eye(n)
    
    # Continuous time or discrete time
    if dt is None:
        sz = 1j * w  # s = jω for continuous time
    else:
        sz = np.exp(1j * w * dt)  # z = e^(jωΔt) for discrete time
    
    # Allocate memory for the frequency response function
    G = np.zeros((nw, m, r), dtype=complex)
    mag = np.full((nw, m, r), np.nan)
    pha = np.full((nw, m, r), np.nan)
    
    # Compute the frequency response function G(jω) = C(sI-A)^(-1)B + D
    for ii in range(nw):
        # Use SVD for robust inversion of (sI-A)
        u, s, vh = np.linalg.svd(sz[ii] * In - A)
        idx = np.where(s > s[0] * tol)[0]
        if len(idx) > 0:
            s_inv = np.diag(1.0 / s[idx])
            char_eq_inv = vh[:len(idx), :].conj().T @ s_inv @ u[:, idx].conj().T
            G[ii, :, :] = C @ char_eq_inv @ B + D
        else:
            G[ii, :, :] = D  # Fallback if matrix is singular
    
    # Compute magnitude
    mag = np.abs(G)
    
    # Compute phase with unwrapping (Krajnik algorithm)
    pha1 = np.arctan2(np.imag(G[0, :, :]), np.real(G[0, :, :]))
    pha[0, :, :] = pha1
    
    # Phase unwrapping using cumulative trapezoidal integration
    pha1_rep = np.tile(pha1, (nw - 1, 1, 1))
    phase_diff = np.angle(G[1:nw, :, :] / G[0:nw-1, :, :])
    pha[1:nw, :, :] = pha1_rep + cumulative_trapezoid(
        phase_diff, axis=0, initial=0
    )
    
    # Remove out-of-Nyquist range values for discrete-time systems
    if dt is not None:
        w_out = np.where(w > np.pi / dt)[0]
        mag[w_out, :, :] = np.nan
        pha[w_out, :, :] = np.nan
    
    # Convert phase to degrees
    pha_deg = pha * 180 / np.pi
    
    # PLOTS ================================================================
    if figno is not None and figno > 0:
        
        plt.figure(figno)
        plt.clf()
        
        for k in range(r):
            # Magnitude plot
            plt.subplot(2, r, k + 1)
            
            if ax == 'x':
                plt.semilogx(w / (2 * np.pi), mag[:, :, k], linewidth=lw)
            elif ax == 'y':
                plt.semilogy(w / (2 * np.pi), mag[:, :, k], linewidth=lw)
            elif ax == 'n':
                plt.plot(w / (2 * np.pi), mag[:, :, k], linewidth=lw)
            else:  # 'b' or anything else -> loglog
                plt.loglog(w / (2 * np.pi), mag[:, :, k], linewidth=lw)
            
            if leg is not None:
                plt.legend(leg)
            
            mag_min = np.nanmin(mag)
            mag_max = np.nanmax(mag)
            plt.axis([np.min(w) / (2 * np.pi), np.max(w) / (2 * np.pi), 
                     mag_min, 1.2 * mag_max])
            
            if k == 0:
                plt.ylabel('magnitude')
            plt.grid(True)
            
            # Phase plot
            plt.subplot(2, r, k + r + 1)
            
            if ax == 'n' or ax == 'y':
                plt.plot(w / (2 * np.pi), pha_deg[:, :, k], linewidth=lw)
            else:
                plt.semilogx(w / (2 * np.pi), pha_deg[:, :, k], linewidth=lw)
            
            pha_min = np.floor(np.nanmin(pha_deg) / 90) * 90
            pha_max = np.ceil(np.nanmax(pha_deg) / 90) * 90
            plt.yticks(np.arange(pha_min, pha_max + 1, 90))
            plt.axis([np.min(w) / (2 * np.pi), np.max(w) / (2 * np.pi),
                     pha_min, pha_max])
            plt.xlabel('frequency (Hertz)')
            
            if k == 0:
                plt.ylabel('phase (degrees)')
            plt.grid(True)
        
        plt.tight_layout()
    
    return mag, pha_deg, G

# -------------------------------------------------------------------------- 
