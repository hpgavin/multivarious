"""
Power Spectral Density Estimation using Welch's Method

This module provides a transparent implementation of Welch's averaged periodogram
method for educational purposes. The algorithm structure is kept explicit to show
students exactly how the PSD is computed through overlapping windowed segments.
"""

import numpy as np
from scipy.stats import chi2
from scipy.signal import detrend
from typing import Tuple, Optional, Union


def psd(x: np.ndarray, 
        Fs: float, 
        nfft: Optional[int] = None, 
        dflag: str = 'linear',
        window: str = 'sine') -> Union[Tuple[np.ndarray, np.ndarray], 
                                        Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Estimate the positive frequency part of the double-sided Power Spectral 
    Density of a signal vector x using Welch's averaged periodogram method.
    
    The signal x is divided into overlapping segments of length nfft, each of 
    which is detrended, then windowed. The magnitude squared of the length nfft 
    DFTs of the segments are averaged to form Pxx.
    
    Pxx is normalized to agree with Parseval's theorem:
        trapz(t, x**2)/T  equals  2*trapz(f, Pxx)    ...  for 0 < f < Fs/2
    
    Parameters
    ----------
    x : array_like
        Vector of sampled input signal (1D array)
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
    Pxx : ndarray
        Real-valued auto-power spectral density (length nfft/2+1 for nfft even,
        (nfft+1)/2 for nfft odd, or nfft if signal x is complex)
    f : ndarray
        Frequency values corresponding to Pxx
    Pxxc : ndarray, optional (if requested via multiple return values)
        Confidence interval bounds [lower, upper] for 95% confidence level
        Shape: (len(Pxx), 2)
    
    Notes
    -----
    This implementation uses:
    - 50% overlap between segments (nOverlap = nWindow / 2)
    - Segments are detrended before windowing
    - Normalization factor: K * norm(window)^2 for asymptotically unbiased estimates
    
    The confidence intervals use Kay's formula (p. 76, eqn 4.16) with a 
    chi-squared distribution and reduction factor RF = 9*K/11.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Generate a test signal with two frequency components
    >>> Fs = 1000  # Sampling frequency
    >>> t = np.arange(0, 1, 1/Fs)
    >>> x = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*120*t) + 0.2*np.random.randn(len(t))
    >>> Pxx, f = psd(x, Fs)
    >>> # With confidence intervals
    >>> Pxx, f, Pxxc = psd(x, Fs)
    
    References
    ----------
    .. [1] Welch, P.D. (1967) "The use of fast Fourier transform for the 
           estimation of power spectra: A method based on time averaging over 
           short, modified periodograms", IEEE Trans. Audio Electroacoust. 
           vol. 15, pp. 70-73.
    .. [2] Kay, S.M. (1988) "Modern Spectral Estimation: Theory and Application",
           Prentice-Hall.
    """
    # Ensure x is a 1D column vector
    x = np.atleast_1d(x).flatten()
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

    print(f' psd: {K} windows of {nfft} points each')
    
    # Zero-pad x if it has length less than the window length
    if n < nWindow:
        x = np.pad(x, (0, nWindow - n), mode='constant')
        n = nWindow
    
    # ========================================================================
    # Normalizing scale factor ==> asymptotically unbiased
    # ========================================================================
    KNW2 = K * np.linalg.norm(win) ** 2
    
    # Alternative: KNW2 = K * np.sum(win) ** 2  # ==> peaks are about right
    
    # Initialize power spectral density array
    Pxx = np.zeros(nfft)
    
    # ========================================================================
    # Main loop: Process each overlapping segment
    # ========================================================================
    segment_start = 0
    for k in range(K):
        # Extract current segment
        segment_indices = segment_start + index
        x_segment = x[segment_indices]
        
        # Detrend and window the segment
        if dflag == 'linear':
            # Remove linear trend (polyfit degree 1)
            x_detrended = detrend(x_segment, type='linear')
            xw = win * x_detrended
            
        elif dflag == 'mean':
            # Remove mean value (polyfit degree 0)
            x_detrended = detrend(x_segment, type='constant')
            xw = win * x_detrended
            
        else:  # dflag == 'none'
            # No detrending
            xw = win * x_segment
        
        # Compute FFT and power
        Xx2 = np.abs(np.fft.fft(xw, nfft)) ** 2
        Pxx = Pxx + Xx2
        
        # Advance to the next segment (with overlap)
        segment_start = segment_start + (nWindow - nOverlap)
    
    # ========================================================================
    # Return the positive frequency part of the power spectral density
    # ========================================================================
    if not np.any(np.imag(x) != 0):  # x is real-valued
        if nfft % 2 == 1:  # nfft odd
            select = np.arange((nfft + 1) // 2)
        else:  # nfft even
            select = np.arange(nfft // 2 + 1)  # include DC and Nyquist
        Pxx = Pxx[select]
    else:  # x is complex
        select = np.arange(nfft)
    
    # Frequency vector
    f = select * Fs / nfft
    
    # ========================================================================
    # Normalize spectral density to agree with Parseval's theorem
    # ========================================================================
    Pxx = Pxx / (KNW2 * Fs)
    
    # ========================================================================
    # Compute confidence intervals if requested
    # ========================================================================
    if K > 0:
        # Confidence interval from Kay, p. 76, eqn 4.16
        p = 0.95  # confidence level
        alpha = 1 - p
        RF = 9 * K / 11  # reduction factor
        dof = 2 * K * RF  # degrees of freedom
        
        # Lower and upper confidence bounds
        chi2_lower = chi2.ppf(alpha / 2, dof)
        chi2_upper = chi2.ppf(1 - alpha / 2, dof)
        
        Pxxc = np.column_stack([
            Pxx * dof / chi2_upper,  # lower bound
            Pxx * dof / chi2_lower   # upper bound
        ])
    else:
        Pxxc = np.zeros((len(Pxx), 2))
    
    return Pxx, f, Pxxc


# ============================================================================
# Example usage and testing
# ============================================================================
if __name__ == "__main__":
    """
    Demonstration of the PSD function with various window types and 
    comparison of results.
    """
    import matplotlib.pyplot as plt
    
    # Enable interactive mode and LaTeX rendering
    plt.ion()
    plt.rcParams['text.usetex'] = False  # Set to True if LaTeX is installed
    
    # ========================================================================
    # Generate test signal
    # ========================================================================
    Fs = 1000  # Sampling frequency (Hz)
    T = 2.048  # Duration (seconds)
    nfft = 0   # number of points in the FFT
    t = np.arange(0, T, 1/Fs)
    
    # Signal: two sinusoids + noise
    f1, f2 = 50, 120  # Frequencies (Hz)
    x = (np.sin(2 * np.pi * f1 * t) + 
         0.5 * np.sin(2 * np.pi * f2 * t) + 
         0.2 * np.random.randn(len(t)))
    
    # ========================================================================
    # Compute PSD with different windows
    # ========================================================================
    windows = ['sine', 'hann', 'hamming']
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Time domain signal
    axes[0].plot(t, x, 'b-', linewidth=0.5, alpha=0.7)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Input Signal: Two Sinusoids + Noise')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 0.2])  # Show only first 0.2 seconds
    
    # Plot 2: Power Spectral Density with confidence intervals
    for window_type in windows:
        Pxx, f, Pxxc = psd(x, Fs, nfft, window=window_type)
        
        # Plot PSD
        axes[1].semilogy(f, Pxx, label=f'{window_type} window', linewidth=1.5)
    
    # Add confidence interval for the sine window
    Pxx_sine, f_sine, Pxxc_sine = psd(x, Fs, nfft, window='sine')
    axes[1].fill_between(f_sine, Pxxc_sine[:, 0], Pxxc_sine[:, 1], 
                         alpha=0.2, label='50% confidence interval (sine)')
    
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Power Spectral Density (1/Hz)')
    axes[1].set_title('Power Spectral Density Estimate (Welch Method)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3, which='both')
    axes[1].set_xlim([0, Fs/2])
    
    # Add vertical lines at true frequencies
    axes[1].axvline(f1, color='r', linestyle='--', alpha=0.5, linewidth=1)
    axes[1].axvline(f2, color='r', linestyle='--', alpha=0.5, linewidth=1)
    axes[1].text(f1, axes[1].get_ylim()[1]/2, f'  {f1} Hz', 
                rotation=90, va='bottom', color='r')
    axes[1].text(f2, axes[1].get_ylim()[1]/2, f'  {f2} Hz', 
                rotation=90, va='bottom', color='r')
    
    plt.tight_layout()
    plt.show()
    
    # ========================================================================
    # Verify Parseval's theorem
    # ========================================================================
    print("\n" + "="*70)
    print("PARSEVAL'S THEOREM VERIFICATION")
    print("="*70)
    
    # Time domain energy
    time_energy = np.trapz(x**2, t) / T
    
    # Frequency domain energy (positive frequencies only, so multiply by 2)
    Pxx, f, _ = psd(x, Fs, nfft, window='sine')
    freq_energy = 2 * np.trapz(Pxx, f)
    
    print(f"Time domain energy:      {time_energy:.6f}")
    print(f"Frequency domain energy: {freq_energy:.6f}")
    print(f"Relative error:          {abs(time_energy - freq_energy)/time_energy * 100:.3f}%")
    print("\nNote: Small errors are expected due to windowing and finite segments.")
    print("="*70 + "\n")
    
    # Keep plot window open
    input("Press Enter to close plots and exit...")
    plt.close('all')
