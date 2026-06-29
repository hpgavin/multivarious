"""
Spectral Matrix Plotting

This module provides visualization for matrices of frequency response functions
(FRF) and power spectral densities (PSD), useful for multi-input multi-output
system analysis and cross-spectral analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Literal


def plot_spectra(fa: Optional[np.ndarray] = None,
                 Sa: Optional[np.ndarray] = None,
                 fb: Optional[np.ndarray] = None,
                 Sb: Optional[np.ndarray] = None,
                 fc: Optional[np.ndarray] = None,
                 Sc: Optional[np.ndarray] = None,
                 frf_psd: Literal['FRF', 'PSD'] = 'PSD',
                 fig_num: Optional[int] = None) -> plt.Figure:
    """
    Plot a matrix of frequency response functions or power spectral densities.
    
    Creates a grid of subplots showing spectral relationships between channels.
    Up to three different datasets can be overlaid for comparison.
    
    For FRF mode: Plots magnitude on log-log scale in a single row.
    For PSD mode: Creates m×r grid where:
        - Diagonal (i=j): Auto-spectra (real values)
        - Off-diagonal (i≠j): Cross-spectra (real and imaginary parts)
    
    Parameters
    ----------
    fa : ndarray, optional
        Frequency vector for first dataset, shape (nf,)
    Sa : ndarray, optional
        First spectral matrix, shape (nf, m, r)
        - nf: number of frequency points
        - m: number of output channels
        - r: number of input channels (or response channels)
    fb : ndarray, optional
        Frequency vector for second dataset, shape (nf,)
    Sb : ndarray, optional
        Second spectral matrix, shape (nf, m, r)
    fc : ndarray, optional
        Frequency vector for third dataset, shape (nf,)
    Sc : ndarray, optional
        Third spectral matrix, shape (nf, m, r)
    frf_psd : {'FRF', 'PSD'}, optional
        Plot type:
        - 'FRF': Frequency Response Function (magnitude, log-log)
        - 'PSD': Power Spectral Density (semilog, real/imaginary)
        Default: 'PSD'
    fig_num : int, optional
        Figure number. If None, creates new figure
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plots
    
    Notes
    -----
    FRF Mode:
        - Single row of r subplots showing magnitude vs frequency
        - Black solid line for Sa (if provided)
        - Black circles for Sb (if provided)
        - Blue solid line for Sc (if provided)
    
    PSD Mode:
        - m×r grid of subplots
        - Diagonal elements: auto-spectra (one line each)
        - Off-diagonal: cross-spectra (real and imaginary)
        - Black line for Sa, green for Sb, red for Sc
        - Row labels: y_i (output channels)
        - Column labels: y_j (input/response channels)
    
    Examples
    --------
    >>> import numpy as np
    >>> from csd import csd
    >>> 
    >>> # Generate test signals
    >>> Fs = 1000
    >>> t = np.arange(0, 2, 1/Fs)
    >>> x1 = np.sin(2*np.pi*50*t) + 0.5*np.random.randn(len(t))
    >>> x2 = np.sin(2*np.pi*120*t) + 0.5*np.random.randn(len(t))
    >>> 
    >>> # Compute cross-spectra
    >>> P11, f, _ = csd(x1, x1, Fs)
    >>> P12, f, _ = csd(x1, x2, Fs)
    >>> P21, f, _ = csd(x2, x1, Fs)
    >>> P22, f, _ = csd(x2, x2, Fs)
    >>> 
    >>> # Organize into spectral matrix
    >>> S = np.zeros((len(f), 2, 2), dtype=complex)
    >>> S[:, 0, 0] = P11
    >>> S[:, 0, 1] = P12
    >>> S[:, 1, 0] = P21
    >>> S[:, 1, 1] = P22
    >>> 
    >>> # Plot spectral matrix
    >>> fig = plot_spectra(fb=f, Sb=S, frf_psd='PSD')
    """
    # Validate inputs
    if Sb is None and Sa is None and Sc is None:
        raise ValueError("At least one spectral matrix (Sa, Sb, or Sc) must be provided")
    
    # Use Sb as reference if available, otherwise use first non-None
    S_ref = Sb if Sb is not None else (Sa if Sa is not None else Sc)
    
    # Get dimensions from reference spectrum
    nf, m, r = S_ref.shape
    
    # Plot parameters
    lw = 2.0  # line width
    fs = 14   # font size
    
    # Create or get figure
    if fig_num  is not None:
        fig_obj = plt.figure(fig_num )
        plt.clf()
    else:
        if frf_psd == 'FRF':
            fig_obj = plt.figure(figsize=(5*r, 5))
        else:  # PSD
            fig_obj = plt.figure(figsize=(4*r, 4*m))
    
    # ========================================================================
    # FRF Mode: Frequency Response Functions
    # ========================================================================
    if frf_psd == 'FRF':
        for k in range(r):
            ax = fig_obj.add_subplot(1, r, k+1)
            
            # Plot Sa (black solid line)
            if Sa is not None:
                ax.loglog(fa, np.abs(Sa[:, :, k]), '-k', linewidth=lw)
            
            # Plot Sb (black circles)
            if Sb is not None:
                ax.loglog(fb, np.abs(Sb[:, :, k]), 'ok', markersize=4)
            
            # Plot Sc (blue solid line)
            if Sc is not None:
                ax.loglog(fc, np.abs(Sc[:, :, k]), '-b', linewidth=lw)
            
            ax.grid(True, alpha=0.3, which='both')
            ax.set_xlabel('Frequency (Hz)', fontsize=fs)
            
            if k == 0:
                ax.set_ylabel('Magnitude', fontsize=fs)
            
            ax.set_title(f'Input {k+1}', fontsize=fs)
            ax.tick_params(labelsize=fs-2)
    
    # ========================================================================
    # PSD Mode: Power Spectral Densities
    # ========================================================================
    elif frf_psd == 'PSD':
        for ii in range(m):
            for jj in range(r):
                subplot_idx = ii * r + jj + 1
                ax = fig_obj.add_subplot(m, r, subplot_idx)
                
                # Diagonal elements: auto-spectra (real values only)
                if ii == jj:
                    # Plot Sa (black)
                    if Sa is not None:
                        ax.semilogx(fa, np.real(Sa[:, ii, jj]), '-k', 
                                   linewidth=lw, label='Sa')
                    
                    # Plot Sb (green)
                    if Sb is not None:
                        ax.semilogx(fb, np.real(Sb[:, ii, jj]), '-g', 
                                   linewidth=lw, label='Sb')
                    
                    # Plot Sc (red)
                    if Sc is not None:
                        ax.semilogx(fc, np.real(Sc[:, ii, jj]), '-r', 
                                   linewidth=lw, label='Sc')
                
                # Off-diagonal elements: cross-spectra (real and imaginary)
                else:
                    # Plot Sa
                    if Sa is not None:
                        ax.semilogx(fa, np.real(Sa[:, ii, jj]), '-k', 
                                   linewidth=lw, label='Sa (real)')
                        if not np.allclose(np.imag(Sa[:, ii, jj]), 0):
                            ax.semilogx(fa, np.imag(Sa[:, ii, jj]), '--k', 
                                       linewidth=lw, label='Sa (imag)')
                    
                    # Plot Sb
                    if Sb is not None:
                        ax.semilogx(fb, np.real(Sb[:, ii, jj]), '-g', 
                                   linewidth=lw, label='Sb (real)')
                        if not np.allclose(np.imag(Sb[:, ii, jj]), 0):
                            ax.semilogx(fb, np.imag(Sb[:, ii, jj]), '--g', 
                                       linewidth=lw, label='Sb (imag)')
                    
                    # Plot Sc
                    if Sc is not None:
                        ax.semilogx(fc, np.real(Sc[:, ii, jj]), '-r', 
                                   linewidth=lw, label='Sc (real)')
                        if not np.allclose(np.imag(Sc[:, ii, jj]), 0):
                            ax.semilogx(fc, np.imag(Sc[:, ii, jj]), '--r', 
                                       linewidth=lw, label='Sc (imag)')
                
                # Formatting
                ax.grid(True, alpha=0.3, which='both')
                ax.tick_params(labelsize=fs-4)
                
                # Labels
                if ii == 0:
                    ax.set_title(f'$y_{{{jj+1}}}$', fontsize=fs)
                if jj == 0:
                    ax.set_ylabel(f'$y_{{{ii+1}}}$', fontsize=fs)
                if ii == m - 1:
                    ax.set_xlabel('Frequency (Hz)', fontsize=fs-2)
                
                # Add legend only to first subplot if multiple datasets
                if ii == 0 and jj == 0:
                    n_datasets = sum([Sa is not None, Sb is not None, Sc is not None])
                    if n_datasets > 1:
                        ax.legend(fontsize=fs-6, loc='best')
    
    else:
        raise ValueError(f"frf_psd must be 'FRF' or 'PSD', got '{frf_psd}'")
    
    plt.tight_layout()
    
    return fig_obj


# ============================================================================
# Example usage
# ============================================================================
if __name__ == "__main__":
    """
    Demonstration of the plot_spectra function with synthetic spectral data.
    """

    #plt.rcParams['text.usetex'] = True # Set to True if LaTeX is installed

    pdf_plots = True  # Set to True to save PDF files
    interactive = True # Enable interactive mode for matplotlib

    if interactive:
        plt.ion() # plot interactive mode: on

    # ========================================================================
    # Example 1: PSD Mode - Cross-spectral density matrix
    # ========================================================================
    fig_num = 1
    print("\n" + "="*70)
    print("Generating synthetic cross-spectral density data...")
    print("="*70)
    
    # Generate synthetic signals
    Fs = 1000  # Sampling frequency
    T = 4.0    # Duration
    t = np.arange(0, T, 1/Fs)
    
    # Three correlated signals at different frequencies
    f1, f2, f3 = 30, 80, 150  # Hz
    
    x1 = np.sin(2*np.pi*f1*t) + 0.3*np.random.randn(len(t))
    x2 = (0.8*np.sin(2*np.pi*f2*t) + 
          0.3*np.sin(2*np.pi*f1*t) +  # Some correlation with x1
          0.3*np.random.randn(len(t)))
    x3 = (np.sin(2*np.pi*f3*t) + 
          0.2*np.sin(2*np.pi*f2*t) +  # Some correlation with x2
          0.3*np.random.randn(len(t)))
    
    # Import csd function (assumes it's available)
    try:
        from multivarious.dsp import csd
        
        # Compute all cross-spectra
        print("Computing cross-spectral densities...")
        P11, f, _ = csd(x1, x1, Fs, window='hann')
        P12, f, _ = csd(x1, x2, Fs, window='hann')
        P13, f, _ = csd(x1, x3, Fs, window='hann')
        P21, f, _ = csd(x2, x1, Fs, window='hann')
        P22, f, _ = csd(x2, x2, Fs, window='hann')
        P23, f, _ = csd(x2, x3, Fs, window='hann')
        P31, f, _ = csd(x3, x1, Fs, window='hann')
        P32, f, _ = csd(x3, x2, Fs, window='hann')
        P33, f, _ = csd(x3, x3, Fs, window='hann')
        
        # Organize into spectral matrix (nf x m x r)
        S_psd = np.zeros((len(f), 3, 3), dtype=complex)
        S_psd[:, 0, 0] = P11
        S_psd[:, 0, 1] = P12
        S_psd[:, 0, 2] = P13
        S_psd[:, 1, 0] = P21
        S_psd[:, 1, 1] = P22
        S_psd[:, 1, 2] = P23
        S_psd[:, 2, 0] = P31
        S_psd[:, 2, 1] = P32
        S_psd[:, 2, 2] = P33
        
        print(f"Spectral matrix shape: {S_psd.shape}")
        print(f"Frequency range: {f[0]:.2f} - {f[-1]:.2f} Hz")
        
        # Plot PSD matrix
        fig1 = plot_spectra(fb=f, Sb=S_psd, frf_psd='PSD', fig_num=fig_num)
        fig1.suptitle('Cross-Spectral Density Matrix (3×3)', 
                     fontsize=10, fontweight='bold', y=0.995)

        
        print("\nFigure 1: PSD matrix")
        print("  - Diagonal: Auto-spectra (power spectral densities)")
        print("  - Off-diagonal: Cross-spectra (real and imaginary parts)")
        print("  - Green lines show the spectral relationships")
        
    except ImportError:
        print("Note: csd module not found. Creating synthetic spectral data instead...")
        
        # Create synthetic spectral matrix
        f = np.logspace(0, 2.5, 200)  # 1 to ~316 Hz
        S_psd = np.zeros((len(f), 3, 3), dtype=complex)
        
        for ii in range(3):
            for jj in range(3):
                # Peak at different frequencies
                peak_freq = 30 + ii*50 + jj*20
                peak = 100 * np.exp(-((f - peak_freq)/20)**2)
                
                if ii == jj:
                    # Auto-spectra: real and positive
                    S_psd[:, ii, jj] = peak + 1.0
                else:
                    # Cross-spectra: complex
                    phase = np.pi/4 * (ii - jj)
                    S_psd[:, ii, jj] = (peak + 1.0) * np.exp(1j * phase)
        
        fig1 = plot_spectra(fb=f, Sb=S_psd, frf_psd='PSD', fig_num=fig_num)
        fig1.suptitle('Synthetic Cross-Spectral Density Matrix (3×3)', 
                     fontsize=10, fontweight='bold', y=0.995)

    # Display plots
    if not interactive:
        plt.show()
    
    # Save plots to .pdf
    if pdf_plots:
        filename = f'plot_spectra-{fig_num:04d}.pdf'
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Saved: {filename}")

   
    # ========================================================================
    # Example 2: FRF Mode - Frequency Response Functions
    # ========================================================================
    fig_num = 2
    print("\n" + "="*70)
    print("Generating synthetic frequency response function data...")
    print("="*70)
    
    # Create synthetic FRF data
    f_frf = np.logspace(0, 2, 100)  # 1 to 100 Hz
    
    # Two systems with resonances at different frequencies
    omega_n1 = 2 * np.pi * 15  # Natural frequency 1
    omega_n2 = 2 * np.pi * 45  # Natural frequency 2
    zeta = 0.05  # Damping ratio
    
    # FRF matrix (nf x m x r) - 2 outputs, 2 inputs
    S_frf = np.zeros((len(f_frf), 2, 2), dtype=complex)
    
    for ii in range(2):
        for jj in range(2):
            omega_n = omega_n1 if ii == 0 else omega_n2
            omega = 2 * np.pi * f_frf
            
            # Second-order system FRF
            H = 1 / (omega_n**2 - omega**2 + 2j*zeta*omega_n*omega)
            
            # Add coupling between channels
            if ii != jj:
                H = H * 0.3  # Reduced coupling
            
            S_frf[:, ii, jj] = H
    
    # Create two variants for comparison
    S_frf_theory = S_frf.copy()
    S_frf_experimental = S_frf * (1 + 0.1*np.random.randn(*S_frf.shape))
    
    fig2 = plot_spectra(fa=f_frf, Sa=S_frf_theory,
                       fb=f_frf, Sb=S_frf_experimental,
                       frf_psd='FRF', fig_num=fig_num)

    fig2.suptitle('Frequency Response Functions - Theory vs Experiment', 
                 fontsize=10, fontweight='bold')

    # Display plots
    if not interactive:
        plt.show()
    
    # Save plots to .pdf
    if pdf_plots:
        filename = f'plot_spectra-{fig_num:04d}.pdf'
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Saved: {filename}")

    print("\nFigure 2: FRF comparison")
    print("  - Black lines: Theoretical FRF")
    print("  - Black circles: Experimental FRF")
    print("  - Shows magnitude vs frequency on log-log scale")
    print("  - Resonance peaks visible at natural frequencies")
    
    # ========================================================================
    # Example 3: Comparison of three datasets (PSD mode)
    # ========================================================================
    fig_num = 3
    print("\n" + "="*70)
    print("Generating three-dataset comparison...")
    print("="*70)
    
    # Create three variations of spectral data
    S_model = S_psd.copy()
    S_exp1 = S_psd * (1 + 0.15*np.random.randn(*S_psd.shape))
    S_exp2 = S_psd * (1 + 0.25*np.random.randn(*S_psd.shape))
    
    fig3 = plot_spectra(fa=f, Sa=S_model,
                       fb=f, Sb=S_exp1,
                       fc=f, Sc=S_exp2,
                       frf_psd='PSD', fig_num=fig_num)
    fig3.suptitle('Three-Dataset Comparison: Model vs Experiments', 
                 fontsize=10, fontweight='bold', y=0.995)
    
    print("\nFigure 3: Three-dataset comparison")
    print("  - Black: Model prediction")
    print("  - Green: Experiment 1")
    print("  - Red: Experiment 2")
    
    # Display plots
    if not interactive:
        plt.show()
    
    # Save plots to .pdf
    if pdf_plots:
        filename = f'plot_spectra-{fig_num:04d}.pdf'
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        print(f"Saved: {filename}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("PLOT_SPECTRA DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nKey features demonstrated:")
    print("  1. PSD mode: Cross-spectral density matrices")
    print("     - Diagonal: auto-spectra (real)")
    print("     - Off-diagonal: cross-spectra (real + imaginary)")
    print("  2. FRF mode: Frequency response functions")
    print("     - Log-log magnitude plots")
    print("  3. Multi-dataset comparison (up to 3)")
    print("\nApplications:")
    print("  - MIMO system identification")
    print("  - Modal analysis")
    print("  - Structural dynamics")
    print("  - Multi-channel signal processing")
    print("="*70 + "\n")
    
    # Keep plots open
    input("Press Enter to close plots and exit...")
    plt.close('all')
