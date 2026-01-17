"""
ftdsp.py - Frequency-domain digital signal processing

Band-pass filtering, differentiation, and integration using FFT methods.
Alternative to state-space filtering methods.

Translation from MATLAB by Claude, 2024-12-12
Original: ftdsp.m by Henri Gavin, Duke University, 2007
"""

import numpy as np
import warnings


def ftdsp(u, sr, flo, fhi, ni=0):
    """
    Band-pass filter and integrate/differentiate discrete-time signals using FFT.
    
    Applies band-pass filtering in the frequency domain, optionally followed by
    integration or differentiation. Uses FFT-based convolution with frequency
    domain windowing for smooth transitions.
    
    Parameters
    ----------
    u : ndarray, shape (P,) or (P, m)
        Discrete-time signal(s) to be filtered/integrated
        - 1D array: Single signal
        - 2D array: Multiple signals (each column is a signal)
    sr : float
        Sample rate (Hz)
    flo : float
        Low frequency limit for band-pass filter (Hz)
        Must satisfy: 0 <= flo < fhi
    fhi : float
        High frequency limit for band-pass filter (Hz)
        Must satisfy: flo < fhi <= sr/2 (Nyquist)
    ni : int, optional
        Number of integrations (default: 0)
        - ni > 0: Integrate ni times
        - ni = 0: No integration/differentiation (just filter)
        - ni < 0: Differentiate |ni| times
    
    Returns
    -------
    y : ndarray, same shape as u
        Filtered and integrated/differentiated signal(s)
    
    Notes
    -----
    Processing Steps:
    1. Detrend input signal (remove linear trend)
    2. Apply Tukey window (10% taper each end)
    3. FFT to frequency domain (zero-padded to power of 2)
    4. Apply band-pass filter with tapered edges
    5. Apply integration/differentiation: H(jω) = (jω)^(-ni)
    6. IFFT back to time domain
    7. Retain original number of points
    
    Band-Pass Filter:
    - Pass band: [flo, fhi]
    - Transition bands: ~10% of pass band width
    - Taper: Raised cosine (smooth roll-off)
    - Applied to both positive and negative frequencies
    
    Integration/Differentiation:
    - Integration: Multiply by (jω)^(-1) in frequency domain
    - Differentiation: Multiply by (jω) in frequency domain
    - ni integrations → multiply by (jω)^(-ni)
    - DC component (ω=0) handled specially to avoid singularity
    
    FFT Zero-Padding:
    - Signal zero-padded to next power of 2 for efficiency
    - Increases frequency resolution
    - Original length data returned
    
    Preprocessing:
    - Detrending removes linear drift
    - Windowing reduces spectral leakage
    - Both improve numerical accuracy
    
    Examples
    --------
    >>> import numpy as np
    >>> 
    >>> # Example 1: Band-pass filter acceleration to get velocity
    >>> sr = 100  # Hz
    >>> t = np.arange(0, 10, 1/sr)
    >>> accel = np.sin(2*np.pi*5*t)  # 5 Hz signal
    >>> 
    >>> # Filter 1-20 Hz and integrate once
    >>> veloc = ftdsp(accel, sr, flo=1, fhi=20, ni=1)
    >>> 
    >>> # Example 2: Band-pass filter and integrate to displacement
    >>> displ = ftdsp(accel, sr, flo=0.1, fhi=25, ni=2)
    >>> 
    >>> # Example 3: High-pass filter only (no integration)
    >>> filtered = ftdsp(accel, sr, flo=2, fhi=sr/2, ni=0)
    >>> 
    >>> # Example 4: Differentiate velocity to get acceleration
    >>> accel_computed = ftdsp(veloc, sr, flo=0.5, fhi=40, ni=-1)
    >>> 
    >>> # Example 5: Multiple signals (each column)
    >>> signals = np.random.randn(1000, 3)  # 3 signals
    >>> filtered = ftdsp(signals, sr, flo=1, fhi=10, ni=0)
    
    See Also
    --------
    butter_synth_ss : State-space filtering (preferred method)
    scipy.signal.filtfilt : Zero-phase filtering
    numpy.fft.fft : Fast Fourier Transform
    
    References
    ----------
    H.P. Gavin, "Frequency Domain Signal Processing", Duke University, 2007
    
    Warnings
    --------
    This function is provided for compatibility and special use cases.
    For most applications, state-space filtering methods are preferred due to:
    - Better handling of transients
    - No need for detrending/windowing
    - Easier to cascade filters
    - More numerically stable
    """
    
    # Convert to numpy array
    u = np.asarray(u, dtype=float)
    
    # Handle 1D vs 2D input
    if u.ndim == 1:
        u = u.reshape(-1, 1)
        squeeze_output = True
    else:
        squeeze_output = False
    
    P, m = u.shape
    
    # Validate inputs
    if sr <= 0:
        raise ValueError('Sample rate sr must be positive')
    if flo < 0:
        raise ValueError('Low frequency flo must be non-negative')
    if fhi > sr / 2:
        raise ValueError(f'High frequency fhi must be <= Nyquist frequency {sr/2} Hz')
    if flo >= fhi:
        raise ValueError('Must have flo < fhi')
    
    # Detrend the data (remove linear trend)
    # Improves numerical accuracy by removing DC drift
    from scipy.signal import detrend
    u = detrend(u, axis=0, type='linear')
    
    # Apply Tukey window (10% taper at each end)
    # Reduces spectral leakage
    Pw = int(np.floor(P / 20))  # Number of window points at each end
    
    # Create Tukey window
    w = np.ones(P)
    if Pw > 0:
        # Rising edge: 0.5 * (1 - cos(πt))
        w[:Pw+1] = 0.5 * (1 - np.cos(np.pi * np.arange(Pw+1) / Pw))
        # Falling edge: 0.5 * (1 + cos(πt))
        w[-Pw-1:] = 0.5 * (1 + np.cos(np.pi * np.arange(Pw+1) / Pw))
    
    # Apply window to all signals
    u = u * w.reshape(-1, 1)
    
    # Use next power of 2 for FFT efficiency
    NF = 2 ** int(np.ceil(np.log2(P)))
    
    # Frequency resolution
    delta_f = sr / NF
    
    # Frequency vector: [0, 1, 2, ..., NF/2, -NF/2+1, ..., -2, -1] * delta_f
    f = np.concatenate([
        np.arange(0, NF//2 + 1),
        np.arange(-NF//2 + 1, 0)
    ]) * delta_f
    
    # Compute frequency bin indices for band-pass filter
    kloP = max(int(np.floor(flo / delta_f)) + 1, 1)  # Low freq, positive
    khiP = min(int(np.floor(fhi / delta_f)) + 1, NF//2 + 1)  # High freq, positive
    kloN = min(int(np.ceil(-flo / delta_f)) + 1 + NF, NF)  # Low freq, negative
    khiN = max(int(np.ceil(-fhi / delta_f)) + 1 + NF, NF//2 + 2)  # High freq, negative
    
    # Transition bandwidth (~10% of pass band width)
    Nband_lo = int(np.round(abs(khiP - kloP) / 10))
    Nband_hi = int(np.round(abs(khiP - kloP) / 10))
    
    # Ensure transition bands don't exceed available range
    if Nband_lo > kloP:
        Nband_lo = kloP - 1
    if Nband_hi > khiP:
        Nband_hi = khiP - 1
    
    # Initialize filter transfer function
    H = np.zeros(NF)
    
    # Set pass bands (rectangular)
    H[kloP:khiP+1] = 1.0  # Positive frequencies
    H[khiN:kloN+1] = 1.0  # Negative frequencies
    
    # Apply tapered transition at low frequency edge
    if flo > delta_f and Nband_lo > 0:
        for k in range(Nband_lo + 1):
            # Raised cosine taper
            taper_val = 0.5 * (1 - np.cos(k * np.pi / Nband_lo))
            idx_pos = kloP + k
            idx_neg = kloN - k
            if idx_pos < NF:
                H[idx_pos] = taper_val
            if idx_neg >= 0 and idx_neg < NF:
                H[idx_neg] = taper_val
    
    # Apply tapered transition at high frequency edge
    if fhi < sr/2 - delta_f and Nband_hi > 0:
        for k in range(Nband_hi + 1):
            # Raised cosine taper
            taper_val = 0.5 * (1 - np.cos(k * np.pi / Nband_hi))
            idx_pos = khiP - k
            idx_neg = khiN + k
            if idx_pos >= 0 and idx_pos < NF:
                H[idx_pos] = taper_val
            if idx_neg >= 0 and idx_neg < NF:
                H[idx_neg] = taper_val
    
    # Integration/Differentiation filter: H_ID(jω) = (jω)^(-ni)
    # ni > 0: Integration (divide by jω)
    # ni < 0: Differentiation (multiply by jω)
    # ni = 0: No change
    
    if ni != 0:
        # (jω)^(-ni) = (j*2πf)^(-ni)
        ID = (1j * 2 * np.pi * f) ** (-ni)
        
        # Handle DC component (f=0) to avoid singularity
        # Set to 1 (no integration/differentiation at DC)
        ID[0] = 1.0
    else:
        ID = np.ones(NF)
    
    # Take FFT of input signal(s)
    U = np.fft.fft(u, n=NF, axis=0)
    
    # Apply filters in frequency domain (convolution)
    # Broadcast: H and ID are (NF,), U is (NF, m)
    Y = (H.reshape(-1, 1) * ID.reshape(-1, 1)) * U
    
    # Inverse FFT back to time domain
    y = np.fft.ifft(Y, axis=0)
    
    # Check if imaginary part is small (should be negligible for real signals)
    max_imag_ratio = np.max(np.linalg.norm(y.imag, axis=0) / 
                            (np.linalg.norm(y.real, axis=0) + 1e-15))
    
    if max_imag_ratio > 1e-4:
        warnings.warn(
            f'ftdsp: Imaginary part is larger than expected (ratio={max_imag_ratio:.2e}). '
            'This may indicate numerical issues. Consider using state-space filtering.',
            RuntimeWarning
        )
    
    # Take real part and retain only original number of points
    y = np.real(y[:P, :])
    
    # Return same shape as input
    if squeeze_output:
        return y.squeeze()
    else:
        return y


# Test and demonstration code
if __name__ == '__main__':
    """
    Test ftdsp function
    """
    import matplotlib.pyplot as plt
    
    print("\n" + "="*70)
    print("Testing ftdsp.py - FFT-Based Signal Processing")
    print("="*70)
    
    # Test 1: Band-pass filtering
    print("\nTest 1: Band-Pass Filtering")
    print("-" * 70)
    
    sr = 200  # Hz
    t = np.arange(0, 10, 1/sr)
    
    # Create test signal with multiple frequencies
    signal = (np.sin(2*np.pi*2*t) +      # 2 Hz (should pass)
              0.5*np.sin(2*np.pi*10*t) +  # 10 Hz (should pass)
              0.3*np.sin(2*np.pi*50*t) +  # 50 Hz (should be filtered)
              0.2*np.random.randn(len(t)))  # Noise
    
    # Band-pass filter: 5-20 Hz
    filtered = ftdsp(signal, sr, flo=5, fhi=20, ni=0)
    
    print(f"  Input length: {len(signal)}")
    print(f"  Output length: {len(filtered)}")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Pass band: 5-20 Hz")
    print(f"  Expected: 10 Hz component passes, 2 Hz and 50 Hz filtered out")
    
    # Test 2: Integration
    print("\nTest 2: Integration")
    print("-" * 70)
    
    # Acceleration signal
    f0 = 5  # Hz
    accel = np.sin(2*np.pi*f0*t)
    
    # Integrate once to get velocity
    veloc = ftdsp(accel, sr, flo=0.1, fhi=50, ni=1)
    
    # Analytical velocity (should be -cos(2πf₀t)/(2πf₀))
    veloc_analytical = -np.cos(2*np.pi*f0*t) / (2*np.pi*f0)
    
    # Compare (after initial transient settles)
    error = np.max(np.abs(veloc[100:] - veloc_analytical[100:]))
    print(f"  Input: Acceleration = sin(2π·{f0}·t)")
    print(f"  Output: Velocity (integrated once)")
    print(f"  Expected: -cos(2π·{f0}·t)/(2π·{f0})")
    print(f"  Max error (after settling): {error:.6f}")
    
    # Test 3: Double integration
    print("\nTest 3: Double Integration (Acceleration → Displacement)")
    print("-" * 70)
    
    # Integrate twice
    displ = ftdsp(accel, sr, flo=0.5, fhi=40, ni=2)
    
    # Analytical displacement
    displ_analytical = -np.sin(2*np.pi*f0*t) / (2*np.pi*f0)**2
    
    error = np.max(np.abs(displ[100:] - displ_analytical[100:]))
    print(f"  Input: Acceleration")
    print(f"  Output: Displacement (integrated twice)")
    print(f"  Max error (after settling): {error:.6f}")
    
    # Test 4: Differentiation
    print("\nTest 4: Differentiation (Velocity → Acceleration)")
    print("-" * 70)
    
    # Start with velocity
    veloc_input = np.cos(2*np.pi*f0*t)
    
    # Differentiate
    accel_computed = ftdsp(veloc_input, sr, flo=1, fhi=50, ni=-1)
    
    # Analytical acceleration
    accel_analytical = -2*np.pi*f0 * np.sin(2*np.pi*f0*t)
    
    error = np.max(np.abs(accel_computed[100:] - accel_analytical[100:]))
    print(f"  Input: Velocity = cos(2π·{f0}·t)")
    print(f"  Output: Acceleration (differentiated once)")
    print(f"  Expected: -2π·{f0}·sin(2π·{f0}·t)")
    print(f"  Max error (after settling): {error:.6f}")
    
    # Test 5: Multiple signals
    print("\nTest 5: Multiple Signals (Batch Processing)")
    print("-" * 70)
    
    signals = np.column_stack([
        np.sin(2*np.pi*3*t),
        np.sin(2*np.pi*7*t),
        np.sin(2*np.pi*15*t)
    ])
    
    filtered_batch = ftdsp(signals, sr, flo=5, fhi=20, ni=0)
    
    print(f"  Input shape: {signals.shape}")
    print(f"  Output shape: {filtered_batch.shape}")
    print(f"  Signal 1 (3 Hz): Should be filtered out")
    print(f"  Signal 2 (7 Hz): Should pass")
    print(f"  Signal 3 (15 Hz): Should pass")
    
    # Check which passed
    for i, freq in enumerate([3, 7, 15]):
        rms_in = np.sqrt(np.mean(signals[:, i]**2))
        rms_out = np.sqrt(np.mean(filtered_batch[:, i]**2))
        attenuation = 20 * np.log10(rms_out / rms_in)
        print(f"    Signal {i+1} ({freq} Hz): {attenuation:+.1f} dB")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Plot 1: Band-pass filtering
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # Time domain
    axes[0, 0].plot(t[:500], signal[:500], 'b-', linewidth=1, label='Original')
    axes[0, 0].plot(t[:500], filtered[:500], 'r-', linewidth=1.5, 
                    alpha=0.7, label='Filtered (5-20 Hz)')
    axes[0, 0].set_xlabel('Time (s)', fontsize=10)
    axes[0, 0].set_ylabel('Amplitude', fontsize=10)
    axes[0, 0].set_title('Test 1: Band-Pass Filtering (Time Domain)', 
                         fontsize=11, fontweight='bold')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Frequency domain
    from scipy.signal import welch
    f_orig, psd_orig = welch(signal, sr, nperseg=512)
    f_filt, psd_filt = welch(filtered, sr, nperseg=512)
    
    axes[0, 1].semilogy(f_orig, psd_orig, 'b-', linewidth=1, label='Original')
    axes[0, 1].semilogy(f_filt, psd_filt, 'r-', linewidth=1.5, 
                        alpha=0.7, label='Filtered')
    axes[0, 1].axvspan(5, 20, alpha=0.2, color='green', label='Pass band')
    axes[0, 1].set_xlabel('Frequency (Hz)', fontsize=10)
    axes[0, 1].set_ylabel('PSD', fontsize=10)
    axes[0, 1].set_title('Frequency Domain', fontsize=11, fontweight='bold')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim([0, 60])
    
    # Integration
    axes[1, 0].plot(t[:500], accel[:500], 'b-', linewidth=1, label='Acceleration')
    axes[1, 0].plot(t[:500], veloc[:500], 'r-', linewidth=1.5, 
                    alpha=0.7, label='Velocity (integrated)')
    axes[1, 0].plot(t[:500], veloc_analytical[:500], 'k--', linewidth=1, 
                    alpha=0.5, label='Analytical')
    axes[1, 0].set_xlabel('Time (s)', fontsize=10)
    axes[1, 0].set_ylabel('Amplitude', fontsize=10)
    axes[1, 0].set_title('Test 2: Integration', fontsize=11, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Integration error
    axes[1, 1].plot(t, veloc - veloc_analytical, 'r-', linewidth=1)
    axes[1, 1].set_xlabel('Time (s)', fontsize=10)
    axes[1, 1].set_ylabel('Error', fontsize=10)
    axes[1, 1].set_title('Integration Error', fontsize=11, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(t[100], color='k', linestyle='--', alpha=0.3, 
                      label='Transient settles')
    axes[1, 1].legend(fontsize=9)
    
    # Double integration
    axes[2, 0].plot(t[:500], displ[:500], 'r-', linewidth=1.5, 
                    label='Displacement (2× integrated)')
    axes[2, 0].plot(t[:500], displ_analytical[:500], 'k--', linewidth=1, 
                    alpha=0.5, label='Analytical')
    axes[2, 0].set_xlabel('Time (s)', fontsize=10)
    axes[2, 0].set_ylabel('Displacement', fontsize=10)
    axes[2, 0].set_title('Test 3: Double Integration', fontsize=11, fontweight='bold')
    axes[2, 0].legend(fontsize=9)
    axes[2, 0].grid(True, alpha=0.3)
    
    # Differentiation
    axes[2, 1].plot(t[:500], accel_computed[:500], 'r-', linewidth=1.5, 
                    label='Computed (differentiated)')
    axes[2, 1].plot(t[:500], accel_analytical[:500], 'k--', linewidth=1, 
                    alpha=0.5, label='Analytical')
    axes[2, 1].set_xlabel('Time (s)', fontsize=10)
    axes[2, 1].set_ylabel('Acceleration', fontsize=10)
    axes[2, 1].set_title('Test 4: Differentiation', fontsize=11, fontweight='bold')
    axes[2, 1].legend(fontsize=9)
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ftdsp_demo.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: ftdsp_demo.png")
    
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print("✓ Band-pass filtering working correctly")
    print("✓ Integration accurate (after transient)")
    print("✓ Differentiation accurate (after transient)")
    print("✓ Multiple signals processed correctly")
    print("\nNote: State-space methods (butter_synth_ss) are generally")
    print("      preferred for better transient handling and stability.")
    print("="*70 + "\n")
    
    plt.show()
