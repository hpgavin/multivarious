"""
taper.py - Taper the initial and final points of time series

Provides smooth windowing functions for time series to reduce edge effects
and transients in filtering applications.

Supports:
- Planck window - C∞ continuous (infinitely smooth) - DEFAULT
- Tukey window (cosine taper) - C¹ continuous

Translation from MATLAB by Claude, 2024-11-18
Enhanced with Planck window for superior transient response
"""

import numpy as np


def taper(u, Ni=None, Nf=None, window='planck'):
    """
    Taper the initial Ni points and final Nf points of m time series.
    
    Applies a smooth window function to gradually ramp signals from/to zero
    at the edges. Particularly important for:
    - Recursive filtering (removes initial condition transients)
    - Time-domain filtering (gentle startup/shutdown)
    - Reducing edge effects in signal processing
    
    Parameters
    ----------
    u : ndarray, shape (m, N)
        A set of m time series, as row vectors.
        Each row is a separate time series of length N.
    Ni : int, optional
        Number of initial points to taper.
        Default: floor(N/20) (5% of signal length)
    Nf : int, optional
        Number of final points to taper.
        Default: floor(N/20) (5% of signal length)
    window : str, optional
        Window type: 'planck' or 'tukey'
        Default: 'planck' (infinitely smooth, best for filtering)
    
    Returns
    -------
    u_tapered : ndarray, shape (m, N)
        The set of m tapered time series, as row vectors.
    
    Notes
    -----
    **Planck Window** (default, recommended for filtering):
        - C∞ continuous (infinitely differentiable)
        - All derivatives approach zero at edges
        - Minimal ringing in transient response
        - Superior for recursive filters and time-domain filtering
        - Based on Planck function: w(x) = 1/(1 + exp(Z/x + Z/(1-x)))
    
    **Tukey Window** (cosine taper, better for FFT):
        - C¹ continuous (once differentiable)
        - Simpler computation
        - More commonly used in spectral analysis
        - Formula: w(x) = 0.5 * (1 - cos(π·x))
    
    Examples
    --------
    >>> import numpy as np
    >>> # Best for filtering: Planck window (default)
    >>> signal = np.random.randn(1, 1000)
    >>> tapered = taper(signal, Ni=50, Nf=50)  # Uses Planck
    
    >>> # For FFT analysis: Tukey window
    >>> tapered = taper(signal, Ni=50, Nf=50, window='tukey')
    
    References
    ----------
    Planck window:
    - https://en.wikipedia.org/wiki/Window_function#Planck-taper_window
    - Used in gravitational wave data analysis (LIGO)
    - Preferred for removing initial condition transients
    """
    
    # Ensure u is 2D array
    u = np.asarray(u)
    if u.ndim == 1:
        u = u.reshape(1, -1)
    
    m, N = u.shape
    
    # Set default taper lengths (5% of signal length)
    if Nf is None:
        Nf = np.floor(N / 20)
    
    if Ni is None:
        Ni = np.floor(N / 20)
    
    # Ensure taper lengths are valid
    Ni = int(Ni)
    Nf = int(Nf)
    if Ni + Nf + 2 > N:
        raise ValueError(f'Taper lengths too large: Ni={Ni}, Nf={Nf}, N={N}. '
                        f'Need Ni + Nf + 2 <= N')
    
    # Build taper window based on chosen type
    if window.lower() == 'planck':
        w = _planck_window(N, Ni, Nf)
    elif window.lower() == 'tukey':
        w = _tukey_window(N, Ni, Nf)
    else:
        raise ValueError(f"Unknown window type '{window}'. Use 'planck' or 'tukey'.")
    
    # Apply taper to all time series (broadcast across rows)
    u_tapered = u * w[np.newaxis, :]
    
    return u_tapered


def _planck_window(N, Ni, Nf):
    """
    Generate Planck-taper window (C∞ continuous).
    
    Implements the Planck-taper window as defined in:
    https://en.wikipedia.org/wiki/Window_function#Planck-taper_window
    
    The Planck window is infinitely differentiable everywhere, making it
    ideal for applications requiring maximum smoothness, such as:
    - Removing transients from recursive filters
    - Time-domain filtering with gentle startup
    - Applications sensitive to derivative discontinuities
    
    Formula (for left taper, 0 < n < Ni):
        Z_L = Ni * (1/n + 1/(n - Ni))
        w[n] = 1 / (exp(Z_L) + 1)
    
    Parameters
    ----------
    N : int
        Total length of window
    Ni : int
        Number of initial taper points
    Nf : int
        Number of final taper points
    
    Returns
    -------
    w : ndarray, shape (N,)
        Planck window values
    """
    
    w = np.ones(N)
    
    # Initial taper (0 to Ni) - left edge
    if Ni > 0:
        for n in range(Ni):
            # Avoid division by zero at endpoints
            if n == 0:
                w[n] = 0.0
            else:
                # Z_L = Ni * (1/n + 1/(n - Ni))
                Z_L = Ni * (1.0 / n + 1.0 / (n - Ni))
                # Clip to avoid overflow
                Z_L = np.clip(Z_L, -700, 700)
                w[n] = 1.0 / (np.exp(Z_L) + 1.0)
    
    # Final taper (N-Nf to N) - right edge (mirrored)
    if Nf > 0:
        for n in range(N - Nf, N):
            # Mirror the left taper
            n_mirror = N - 1 - n  # Distance from right edge
            if n_mirror == 0:
                w[n] = 0.0
            else:
                # Same formula but mirrored
                Z_R = Nf * (1.0 / n_mirror + 1.0 / (n_mirror - Nf))
                Z_R = np.clip(Z_R, -700, 700)
                w[n] = 1.0 / (np.exp(Z_R) + 1.0)
    
    return w


def _tukey_window(N, Ni, Nf):
    """
    Generate Tukey (cosine-taper) window (C¹ continuous).
    
    The traditional cosine taper. Simpler than Planck but less smooth.
    Good for spectral analysis (FFT) applications.
    
    Parameters
    ----------
    N : int
        Total length of window
    Ni : int
        Number of initial taper points
    Nf : int
        Number of final taper points
    
    Returns
    -------
    w : ndarray, shape (N,)
        Tukey window values
    """
    
    # Initial taper: smooth rise from 0 to 1
    initial_taper = 0.5 * (1 - np.cos(np.pi * np.arange(Ni + 1) / (Ni + 1)))
    
    # Middle section: ones (no tapering)
    middle_ones = np.ones(N - Ni - Nf - 2)
    
    # Final taper: smooth fall from 1 to 0
    final_taper = 0.5 * (1 + np.cos(np.pi * np.arange(Nf + 1) / (Nf + 1)))
    
    # Concatenate all sections
    w = np.concatenate([initial_taper, middle_ones, final_taper])
    
    return w


# Standalone test functions
def compare_windows(Ni=50, Nf=50, N=500):
    """
    Compare Planck and Tukey windows side-by-side.
    
    Returns comparison figure showing windows and their derivatives.
    """
    import matplotlib.pyplot as plt
    
    w_planck = _planck_window(N, Ni, Nf)
    w_tukey = _tukey_window(N, Ni, Nf)
    
    dw_planck = np.gradient(w_planck)
    d2w_planck = np.gradient(dw_planck)
    dw_tukey = np.gradient(w_tukey)
    d2w_tukey = np.gradient(dw_tukey)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    axes[0].plot(w_planck, 'b-', linewidth=2, label='Planck (C∞)')
    axes[0].plot(w_tukey, 'r--', linewidth=2, label='Tukey (C¹)')
    axes[0].set_ylabel('Window Value', fontsize=12)
    axes[0].set_title('Window Function Comparison', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([-0.05, 1.05])
    
    axes[1].plot(dw_planck, 'b-', linewidth=2, label='Planck 1st derivative')
    axes[1].plot(dw_tukey, 'r--', linewidth=2, label='Tukey 1st derivative')
    axes[1].set_ylabel('First Derivative', fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color='k', linewidth=0.5, alpha=0.5)
    
    axes[2].plot(d2w_planck, 'b-', linewidth=2, label='Planck 2nd derivative')
    axes[2].plot(d2w_tukey, 'r--', linewidth=2, label='Tukey 2nd derivative')
    axes[2].set_ylabel('Second Derivative', fontsize=12)
    axes[2].set_xlabel('Sample Index', fontsize=12)
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(0, color='k', linewidth=0.5, alpha=0.5)
    
    for ax in axes:
        ax.axvspan(0, Ni, alpha=0.1, color='yellow')
        ax.axvspan(N-Nf, N, alpha=0.1, color='orange')
    
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    print("\n" + "="*70)
    print("Testing taper.py - Planck vs Tukey Comparison")
    print("="*70)
    
    # Test 1: Window comparison
    print("\nTest 1: Smoothness Comparison")
    fig = compare_windows(Ni=50, Nf=50, N=500)
    plt.savefig('planck_vs_tukey.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: planck_vs_tukey.png")
    print("  → Planck: Smooth derivatives (C∞)")
    print("  → Tukey: Discontinuous 2nd derivative (C¹)")
    
    # Test 2: Transient response
    print("\nTest 2: Transient Response")
    N = 1000
    t = np.linspace(0, 10, N)
    signal = (np.sin(2*np.pi*2*t) + 0.5*np.sin(2*np.pi*5*t)).reshape(1, -1)
    
    tapered_planck = taper(signal, Ni=100, Nf=100, window='planck')
    tapered_tukey = taper(signal, Ni=100, Nf=100, window='tukey')
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 8))
    axes[0].plot(t, signal[0], 'k-', linewidth=1, label='Original')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel('Original')
    axes[0].set_title('Startup Transient Comparison (zoomed to initial 20%)')
    
    axes[1].plot(t, tapered_planck[0], 'b-', linewidth=1.5, label='Planck')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].set_ylabel('Planck')
    
    axes[2].plot(t, tapered_tukey[0], 'r-', linewidth=1.5, label='Tukey')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)
    axes[2].set_ylabel('Tukey')
    axes[2].set_xlabel('Time (s)')
    
    for ax in axes:
        ax.set_xlim([0, 2])
    
    plt.tight_layout()
    plt.savefig('planck_transient.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: planck_transient.png")
    print("  → Planck provides gentler startup")
    
    print("\n" + "="*70)
    print("Recommendation: Use Planck (default) for recursive filters!")
    print("="*70 + "\n")
    
    plt.show()
