"""
Cross Spectral Density Estimation using Welch's Method

This module provides a transparent implementation of Welch's averaged periodogram
method for cross-spectral density estimation. The algorithm structure is kept 
explicit to show students exactly how the CSD is computed through overlapping 
windowed segments.

The Cross Spectral Density (CSD) is a generalization of the Power Spectral 
Density (PSD). When both input signals are identical, CSD reduces to PSD.
"""

import numpy as np
from scipy.stats import chi2
from scipy.signal import detrend
from typing import Tuple, Optional, Union


def csd(x: np.ndarray,
        y: np.ndarray,
        Fs: float, 
        nfft: Optional[int] = None, 
        dflag: str = 'linear',
        window: str = 'sine') -> Union[Tuple[np.ndarray, np.ndarray], 
                                        Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Estimate the positive frequency part of the double-sided Cross Spectral 
    Density between two signal vectors x and y using Welch's averaged 
    periodogram method.
    
    The signals x and y are divided into overlapping segments of length nfft, 
    each of which is detrended, then windowed. The cross-spectrum of the length 
    nfft DFTs of the segments are averaged to form Pxy.
    
    For real-valued signals, Pxy is normalized to agree with Parseval's theorem.
    When x and y are identical, this function reduces to the PSD estimate.
    
    Parameters
    ----------
    x : array_like
        First input signal vector (1D array)
    y : array_like
        Second input signal vector (1D array), must have same length as x
    Fs : float
        Sampling frequency (1 / delta_t)
    nfft : int, optional
        Number of points used in the FFT. Default: 2^(floor(log2(n))-2)
    dflag : {'none', 'mean', 'linear'}, optional
        Detrending type. Default: 'linear'
        - 'none': No detrending
        - 'mean': Remove mean value
        - 'linear': Remove linear trend
    window : str, optional
        Window function type. Default: 'sine'
        Options: 'dirichlet', 'tapered', 'sine', 'lanczos', 
                 'hamming', 'hann', 'gauss'
    
    Returns
    -------
    Pxy : ndarray
        Cross spectral density (complex-valued in general)
        Length: nfft/2+1 for nfft even, (nfft+1)/2 for nfft odd, 
                or nfft if signals are complex
    f : ndarray
        Frequency values corresponding to Pxy
    Pxyc : ndarray, optional (if requested via multiple return values)
        Confidence interval bounds for magnitude [lower, upper] at 95% confidence
        Shape: (len(Pxy), 2)
    
    Notes
    -----
    This implementation uses:
    - 50% overlap between segments (nOverlap = nWindow / 2)
    - Segments are detrended before windowing
    - Normalization factor: K * norm(window)^2 for asymptotically unbiased estimates
    
    The cross-spectrum is computed as:
        Pxy = FFT(x) * conj(FFT(y))
    
    When x == y, this reduces to the power spectral density:
        Pxx = FFT(x) * conj(FFT(x)) = |FFT(x)|^2
    
    The confidence intervals apply to the magnitude |Pxy| and use Kay's formula 
    (p. 76, eqn 4.16) with a chi-squared distribution and reduction factor RF = 9*K/11.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Generate two correlated signals
    >>> Fs = 1000  # Sampling frequency
    >>> t = np.arange(0, 1, 1/Fs)
    >>> x = np.sin(2*np.pi*50*t) + 0.2*np.random.randn(len(t))
    >>> y = np.sin(2*np.pi*50*t + np.pi/4) + 0.2*np.random.randn(len(t))  # Phase shifted
    >>> Pxy, f = csd(x, y, Fs)
    >>> # With confidence intervals
    >>> Pxy, f, Pxyc = csd(x, y, Fs)
    >>> # For auto-spectrum (PSD), use same signal for both inputs
    >>> Pxx, f = csd(x, x, Fs)
    
    References
    ----------
    .. [1] Welch, P.D. (1967) "The use of fast Fourier transform for the 
           estimation of power spectra: A method based on time averaging over 
           short, modified periodograms", IEEE Trans. Audio Electroacoust. 
           vol. 15, pp. 70-73.
    .. [2] Kay, S.M. (1988) "Modern Spectral Estimation: Theory and Application",
           Prentice-Hall.
    .. [3] Bendat, J.S. and Piersol, A.G. (2010) "Random Data: Analysis and 
           Measurement Procedures", 4th ed., Wiley.
    """
    # Ensure x and y are 1D column vectors
    x = np.atleast_1d(x).flatten()
    y = np.atleast_1d(y).flatten()
    
    # Check that x and y have the same length
    if len(x) != len(y):
        raise ValueError(f"x and y must have the same length. "
                        f"Got len(x)={len(x)}, len(y)={len(y)}")
    
    n = len(x)  # number of data points
    
    # Set default nfft if not provided
    if nfft is None or nfft == 0:
        nfft = 2 ** (int(np.floor(np.log2(n))) - 2)
    
    # Make nfft an even number
    if nfft % 2 == 1:
        nfft = nfft + 1
    
    # Segment parameters
    nWindow = nfft  # number of points in a segment of data
    nOverlap = nWindow // 2  # 50% overlap
    index = np.arange(nWindow)
    Pw = nWindow // 10  # taper width for 'tapered' window
    
    # ========================================================================
    # Window function generation
    # ========================================================================
    if window == 'dirichlet':
        # Rectangular window (no windowing)
        win = np.ones(nWindow)
        
    elif window == 'tapered':
        # Tapered rectangular window
        left_taper = 0.5 * (1 - np.cos(np.pi * np.arange(Pw + 1) / Pw))
        right_taper = 0.5 * (1 + np.cos(np.pi * np.arange(Pw + 1) / Pw))
        middle = np.ones(nWindow - 2 * Pw - 2)
        win = np.concatenate([left_taper, middle, right_taper])
        
    elif window == 'sine':
        # Sine window (good for PSD estimates)
        win = np.sin(np.pi * (index + 0.5) / nWindow)
        
    elif window == 'lanczos':
        # Lanczos window (sinc window)
        win = np.sinc(2 * index / (nWindow - 1) - 1)
        
    elif window == 'hamming':
        # Hamming window
        win = 0.53836 - 0.46164 * np.cos(2 * np.pi * index / nWindow)
        
    elif window == 'hann':
        # Hann window (good for TFE estimates)
        win = (1 + np.cos(2 * np.pi * (index - nWindow / 2) / nWindow)) / 2
        
    elif window == 'gauss':
        # Gaussian window
        win = np.exp(-0.5 * ((index - 0.5 - nWindow / 2) / (0.2 * nWindow)) ** 2)
        
    else:
        raise ValueError(f"Unknown window type: '{window}'. Choose from: "
                        "'dirichlet', 'tapered', 'sine', 'lanczos', "
                        "'hamming', 'hann', 'gauss'")
    
    # Calculate number of windows
    K = int(np.fix((n - nOverlap) / (nWindow - nOverlap)))

    print(f' csd: {K} windows of {nfft} points each')
    
    # Zero-pad x and y if they have length less than the window length
    if n < nWindow:
        x = np.pad(x, (0, nWindow - n), mode='constant')
        y = np.pad(y, (0, nWindow - n), mode='constant')
        n = nWindow
    
    # ========================================================================
    # Normalizing scale factor ==> asymptotically unbiased
    # ========================================================================
    KNW2 = K * np.linalg.norm(win) ** 2
    
    # Alternative: KNW2 = K * np.sum(win) ** 2  # ==> peaks are about right
    
    # Initialize cross spectral density array (complex in general)
    Pxy = np.zeros(nfft, dtype=complex)
    
    # ========================================================================
    # Main loop: Process each overlapping segment
    # ========================================================================
    segment_start = 0
    for k in range(K):
        # Extract current segments from both signals
        segment_indices = segment_start + index
        x_segment = x[segment_indices]
        y_segment = y[segment_indices]
        
        # Detrend and window both segments
        if dflag == 'linear':
            # Remove linear trend (polyfit degree 1)
            x_detrended = detrend(x_segment, type='linear')
            y_detrended = detrend(y_segment, type='linear')
            xw = win * x_detrended
            yw = win * y_detrended
            
        elif dflag == 'mean':
            # Remove mean value (polyfit degree 0)
            x_detrended = detrend(x_segment, type='constant')
            y_detrended = detrend(y_segment, type='constant')
            xw = win * x_detrended
            yw = win * y_detrended
            
        else:  # dflag == 'none'
            # No detrending
            xw = win * x_segment
            yw = win * y_segment
        
        # Compute FFT of both windowed segments
        Xw = np.fft.fft(xw, nfft)
        Yw = np.fft.fft(yw, nfft)
        
        # Compute cross-spectrum: X * conj(Y)
        Pxy_segment = Xw * np.conj(Yw)
        Pxy = Pxy + Pxy_segment
        
        # Advance to the next segment (with overlap)
        segment_start = segment_start + (nWindow - nOverlap)
    
    # ========================================================================
    # Return the positive frequency part of the cross spectral density
    # ========================================================================
    # Check if both signals are real
    if not np.any(np.imag(x) != 0) and not np.any(np.imag(y) != 0):  
        if nfft % 2 == 1:  # nfft odd
            select = np.arange((nfft + 1) // 2)
        else:  # nfft even
            select = np.arange(nfft // 2 + 1)  # include DC and Nyquist
        Pxy = Pxy[select]
    else:  # x or y is complex
        select = np.arange(nfft)
    
    # Frequency vector
    f = select * Fs / nfft
    
    # ========================================================================
    # Normalize spectral density to agree with Parseval's theorem
    # ========================================================================
    Pxy = Pxy / (KNW2 * Fs)
    
    # ========================================================================
    # Compute confidence intervals for magnitude if requested
    # ========================================================================
    if K > 0:
        # Confidence interval from Kay, p. 76, eqn 4.16
        # Note: These apply to the magnitude |Pxy|
        p = 0.95  # confidence level
        alpha = 1 - p
        RF = 9 * K / 11  # reduction factor
        dof = 2 * K * RF  # degrees of freedom
        
        # Lower and upper confidence bounds for magnitude
        chi2_lower = chi2.ppf(alpha / 2, dof)
        chi2_upper = chi2.ppf(1 - alpha / 2, dof)
        
        Pxy_mag = np.abs(Pxy)
        Pxyc = np.column_stack([
            Pxy_mag * dof / chi2_upper,  # lower bound
            Pxy_mag * dof / chi2_lower   # upper bound
        ])
    else:
        Pxyc = np.zeros((len(Pxy), 2))
    
    return Pxy, f, Pxyc


# ============================================================================
# Example usage and testing
# ============================================================================
if __name__ == "__main__":
    """
    Demonstration of the CSD function with various scenarios:
    1. Auto-spectrum (CSD with x=y reduces to PSD)
    2. Cross-spectrum between correlated signals
    3. Coherence function computation
    """
    import matplotlib.pyplot as plt
    
    # Enable interactive mode
    plt.ion()
    plt.rcParams['text.usetex'] = False  # Set to True if LaTeX is installed
    
    # ========================================================================
    # Generate test signals
    # ========================================================================
    Fs = 1000  # Sampling frequency (Hz)
    T = 2.048  # Duration (seconds)
    nfft = 0   # Use default nfft
    t = np.arange(0, T, 1/Fs)
    
    # Common signal component (two sinusoids)
    f1, f2 = 50, 120  # Frequencies (Hz)
    common = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
    
    # Signal x: common + noise
    x = common + 0.2 * np.random.randn(len(t))
    
    # Signal y: phase-shifted common + different noise
    y = (np.sin(2 * np.pi * f1 * t + np.pi/4) + 
         0.5 * np.sin(2 * np.pi * f2 * t - np.pi/6) + 
         0.2 * np.random.randn(len(t)))
    
    # ========================================================================
    # Compute spectral estimates
    # ========================================================================
    
    # Auto-spectra (PSD)
    Pxx, f, Pxxc = csd(x, x, Fs, nfft, window='sine')
    Pyy, f, Pyyc = csd(y, y, Fs, nfft, window='sine')
    
    # Cross-spectrum
    Pxy, f, Pxyc = csd(x, y, Fs, nfft, window='sine')
    
    # Coherence function: γ²(f) = |Pxy|² / (Pxx * Pyy)
    # Coherence ranges from 0 (uncorrelated) to 1 (perfectly correlated)
    coherence = np.abs(Pxy)**2 / (Pxx * Pyy)
    
    # Phase of cross-spectrum
    phase = np.angle(Pxy, deg=True)
    
    # ========================================================================
    # Plotting
    # ========================================================================
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    
    # Plot 1: Time domain signals
    axes[0].plot(t, x, 'b-', linewidth=0.5, alpha=0.7, label='Signal x')
    axes[0].plot(t, y, 'r-', linewidth=0.5, alpha=0.7, label='Signal y')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Input Signals: Correlated Sinusoids + Noise')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 0.2])  # Show only first 0.2 seconds
    
    # Plot 2: Auto-spectra (PSD) with confidence intervals
    axes[1].semilogy(f, Pxx, 'b-', label='$P_{xx}$ (PSD of x)', linewidth=1.5)
    axes[1].semilogy(f, Pyy, 'r-', label='$P_{yy}$ (PSD of y)', linewidth=1.5)
    axes[1].fill_between(f, Pxxc[:, 0], Pxxc[:, 1], alpha=0.2, color='b')
    axes[1].axvline(f1, color='k', linestyle='--', alpha=0.3, linewidth=1)
    axes[1].axvline(f2, color='k', linestyle='--', alpha=0.3, linewidth=1)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Power Spectral Density (1/Hz)')
    axes[1].set_title('Auto-Spectra (Power Spectral Densities)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3, which='both')
    axes[1].set_xlim([0, Fs/2])
    
    # Plot 3: Cross-spectrum magnitude
    axes[2].semilogy(f, np.abs(Pxy), 'g-', label='$|P_{xy}|$ (CSD magnitude)', 
                     linewidth=1.5)
    axes[2].fill_between(f, Pxyc[:, 0], Pxyc[:, 1], alpha=0.2, color='g',
                         label='95% confidence')
    axes[2].axvline(f1, color='k', linestyle='--', alpha=0.3, linewidth=1)
    axes[2].axvline(f2, color='k', linestyle='--', alpha=0.3, linewidth=1)
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Cross Spectral Density Magnitude')
    axes[2].set_title('Cross-Spectrum Magnitude')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3, which='both')
    axes[2].set_xlim([0, Fs/2])
    
    # Plot 4: Coherence function
    axes[3].plot(f, coherence, 'purple', linewidth=1.5)
    axes[3].axvline(f1, color='k', linestyle='--', alpha=0.3, linewidth=1)
    axes[3].axvline(f2, color='k', linestyle='--', alpha=0.3, linewidth=1)
    axes[3].axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    axes[3].set_xlabel('Frequency (Hz)')
    axes[3].set_ylabel('Coherence $\\gamma^2(f)$')
    axes[3].set_title('Coherence Function (0 = uncorrelated, 1 = perfectly correlated)')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlim([0, Fs/2])
    axes[3].set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.show()
    
    # ========================================================================
    # Additional plot: Cross-spectrum phase
    # ========================================================================
    fig2, ax = plt.subplots(1, 1, figsize=(10, 4))
    
    ax.plot(f, phase, 'orange', linewidth=1.5)
    ax.axvline(f1, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(f2, color='k', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Phase (degrees)')
    ax.set_title('Cross-Spectrum Phase (Phase difference between x and y)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, Fs/2])
    ax.set_ylim([-180, 180])
    
    plt.tight_layout()
    plt.show()
    
    # ========================================================================
    # Verify that CSD reduces to PSD when x == y
    # ========================================================================
    print("\n" + "="*70)
    print("VERIFICATION: CSD with x=y should equal PSD")
    print("="*70)
    
    Pxx_via_csd, _, _ = csd(x, x, Fs, nfft, window='sine')
    
    # Check if they're the same (should be real and equal)
    max_diff = np.max(np.abs(Pxx_via_csd - Pxx))
    max_imag = np.max(np.abs(np.imag(Pxx_via_csd)))
    
    print(f"Max difference between CSD(x,x) and PSD(x): {max_diff:.2e}")
    print(f"Max imaginary part of CSD(x,x):             {max_imag:.2e}")
    print("\nResult: " + ("✓ PASS" if max_diff < 1e-10 and max_imag < 1e-10 else "✗ FAIL"))
    print("="*70 + "\n")
    
    # ========================================================================
    # Verify Parseval's theorem for cross-spectrum
    # ========================================================================
    print("="*70)
    print("PARSEVAL'S THEOREM FOR CROSS-SPECTRUM")
    print("="*70)
    
    # Time domain: ∫ x(t)*conj(y(t)) dt / T
    time_cross = np.trapz(x * np.conj(y), t) / T
    
    # Frequency domain: 2 * ∫ Pxy(f) df  (for positive frequencies)
    freq_cross = 2 * np.trapz(Pxy, f)
    
    print(f"Time domain cross-energy:      {time_cross:.6f}")
    print(f"Frequency domain cross-energy: {freq_cross:.6f}")
    print(f"Relative error:                {abs(time_cross - freq_cross)/abs(time_cross) * 100:.3f}%")
    print("\nNote: Small errors are expected due to windowing and finite segments.")
    print("="*70 + "\n")
    
    # Keep plot windows open
    input("Press Enter to close plots and exit...")
    plt.close('all')
