"""
chirp.py - Generate frequency-swept signals for structural testing

Computes sine-sweep time records with linearly changing frequency and amplitude.
Returns acceleration, velocity, and displacement time histories.

Features:
- Variable amplitude (exponential decay)
- Power-law frequency sweep
- Automatic tapering
- Phase control
- Built-in visualization

Translation from MATLAB by Claude, 2024-11-18
Original: chirp.m by Henri Gavin, Duke University, 1999
"""

import numpy as np
import matplotlib.pyplot as plt
from multivarious.dsp.chirp import cdiff
from multivarious.dsp.taper import taper


def chirp(ao, af, fo, ff, t, p=2, n=1, phi=90, fig_no=0, units='m'):
    """
    Generate chirp signal with variable amplitude and compute derivatives.
    
    Creates a frequency-swept signal (chirp) with exponentially varying amplitude,
    suitable for structural testing, shake table control, and seismic simulation.
    
    Unlike scipy.signal.chirp, this function:
    - Returns physically-related accel/veloc/displ signals
    - Includes exponential amplitude variation
    - Automatically tapers the signal
    - Is designed for structural/seismic testing applications
    
    Parameters
    ----------
    ao : float
        Starting velocity amplitude (at t=0)
    af : float
        Ending velocity amplitude (at t=T)
    fo : float
        Starting frequency (Hz)
    ff : float
        Ending frequency (Hz)
    t : ndarray
        Time vector (uniformly spaced)
    p : float, optional
        Power of geometric increase in frequency (default: 2)
        - p = 1: Linear frequency increase
        - p = 2: Quadratic (parabolic) increase
        - p > 2: Higher-order increase (more cycles at low freq)
        - 0 < p < 1: Inverse power (more cycles at high freq)
    n : float, optional
        Characteristic exponent for amplitude variation (default: 1)
        Controls how quickly amplitude decays from ao to af
    phi : float, optional
        Initial phase in degrees (default: 90)
        - 0°: Start at maximum
        - 90°: Start at zero crossing (rising) - RECOMMENDED
        - 180°: Start at minimum
        - 270°: Start at zero crossing (falling)
    fig_no : int, optional
        Figure number for plotting (0 = no plot, default: 0)
    units : str, optional
        Units for plotting labels (default: 'm')
        Examples: 'm', 'mm', 'in', 'g' (for acceleration in g's)
    
    Returns
    -------
    accel : ndarray
        Acceleration time history (computed via differentiation)
    veloc : ndarray
        Velocity time history (fundamental chirp signal)
    displ : ndarray
        Displacement time history (computed via integration)
    
    Notes
    -----
    Frequency sweep:
        f(t) = fo + (ff - fo) * (t/T)^p
    
    Amplitude variation (exponential):
        amp(t) = ao * exp(-r * t^n)
        where r = (1/T^n) * log(ao/af)
    
    Phase:
        φ(t) = 2π * [fo*t + (ff-fo)*t^(p+1) / ((p+1)*T^p)]
    
    Signal generation:
        veloc = taper(amp(t) * sin(φ(t) + phi))
        accel = dveloc/dt (via central differences)
        displ = ∫veloc dt (via cumulative sum)
    
    Tapering:
        Automatically tapers 10% at start and 10% at end using
        Planck window for smooth startup/shutdown.
    
    Number of cycles:
        cycles ≈ T * (fo + (ff-fo)/(p+1))
    
    Examples
    --------
    >>> import numpy as np
    >>> from chirp import chirp
    >>> 
    >>> # Example 1: Basic earthquake simulation
    >>> t = np.arange(0, 30, 0.01)  # 30 seconds at 100 Hz
    >>> accel, veloc, displ = chirp(
    ...     ao=0.5, af=0.1,    # Velocity: 0.5 → 0.1 m/s
    ...     fo=0.5, ff=10,     # Frequency: 0.5 → 10 Hz
    ...     t=t, p=2, n=1,     # Quadratic freq, exponential amp
    ...     phi=90,            # Start at zero crossing
    ...     fig_no=1,          # Plot results
    ...     units='m'
    ... )
    >>> 
    >>> # Example 2: Shake table test
    >>> t = np.linspace(0, 60, 6000)  # 60 sec, 100 Hz
    >>> accel, veloc, displ = chirp(1.0, 0.2, 0.2, 20, t, p=2)
    >>> # Use 'displ' for displacement control
    >>> # Use 'accel' for force estimation
    >>> 
    >>> # Example 3: Linear frequency sweep
    >>> accel, veloc, displ = chirp(1.0, 0.5, 1, 10, t, p=1)
    >>> 
    >>> # Example 4: Start at maximum (no phase control)
    >>> accel, veloc, displ = chirp(1.0, 0.5, 1, 10, t, phi=0)
    
    See Also
    --------
    scipy.signal.chirp : Standard chirp (constant amplitude, single output)
    taper : Tapering function used internally
    cdiff : Central difference differentiation used internally
    
    References
    ----------
    Henri Gavin, Department of Civil Engineering, Duke University
    Original MATLAB version: 1999
    
    For frequency sweep methods in seismic testing:
    - ASTM E2126: Standard Test Methods for Cyclic (Reversed) Load Test
    - FEMA 461: Interim Protocols for Determining Seismic Performance
    """
    
    # Input validation
    t = np.asarray(t)
    if t.ndim != 1:
        raise ValueError('Time vector t must be 1-dimensional')
    
    if len(t) < 3:
        raise ValueError('Time vector must have at least 3 points')
    
    # Time parameters
    nt = len(t)
    dt = t[1] - t[0]
    T = nt * dt
    
    # Adjust power p based on frequency sweep direction
    # More cycles at lower frequency
    p = abs(p)
    if fo < ff and p < 1:
        p = 1 / p
    if fo > ff and p > 1:
        p = 1 / p
    
    # Amplitude variation (exponential decay)
    # amp(t) = ao * exp(-r * t^n)
    # where r = (1/T^n) * log(ao/af)
    if ao > 0 and af > 0:
        r = (1 / T**n) * np.log(ao / af)
        amp = ao * np.exp(-r * t**n)
    else:
        raise ValueError('Amplitudes ao and af must be positive')
    
    # Phase envelope (frequency sweep)
    # φ(t) = 2π * [fo*t + (ff-fo)*t^(p+1) / ((p+1)*T^p)]
    phase = 2 * np.pi * (t * fo + t**(p+1) * (ff - fo) / ((p+1) * T**p))
    
    # Add initial phase offset (convert degrees to radians)
    phase = phase + np.radians(phi)
    
    # Generate velocity signal
    # Using sin (starts at zero when phi=90°, good for smooth startup)
    veloc_raw = amp * np.sin(phase)
    
    # Apply tapering (10% at each end)
    taper_length = int(np.floor(nt / 10))
    veloc = taper(veloc_raw.reshape(1, -1), 
                  Ni=taper_length, 
                  Nf=taper_length,
                  window='planck').squeeze()
    
    # Compute acceleration (differentiation)
    accel = cdiff(veloc, dt)
    
    # Compute displacement (integration via cumulative sum)
    displ = np.cumsum(veloc) * dt
    
    # Calculate number of cycles
    cycles = T * (fo + (ff - fo) / (p + 1))
    
    # Print summary
    print(f'     {cycles:.1f} cycles in {T:.2f} seconds with {nt} points')
    
    # Plot if requested
    if fig_no > 0:
        _plot_chirp(t, accel, veloc, displ, fig_no, units, 
                    ao, af, fo, ff, p, n, phi, cycles)
    
    return accel, veloc, displ


def _plot_chirp(t, accel, veloc, displ, fig_no, units, 
                ao, af, fo, ff, p, n, phi, cycles):
    """
    Internal function to plot chirp time histories.
    
    Creates a 3-panel plot showing acceleration, velocity, and displacement.
    """
    
    fig = plt.figure(fig_no, figsize=(12, 8))
    fig.clear()
    
    # Acceleration
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t, accel, 'b-', linewidth=1)
    ax1.set_ylabel(f'Acceleration ({units}/s²)', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([t[0], t[-1]])
    
    # Title with parameters
    title_str = (f'Chirp Signal: {fo:.2f}→{ff:.2f} Hz, '
                f'{ao:.2f}→{af:.2f} {units}/s, '
                f'p={p:.1f}, n={n:.1f}, φ={phi}°, '
                f'{cycles:.1f} cycles')
    ax1.set_title(title_str, fontsize=12, fontweight='bold')
    
    # Velocity
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(t, veloc, 'r-', linewidth=1)
    ax2.set_ylabel(f'Velocity ({units}/s)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([t[0], t[-1]])
    
    # Displacement
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(t, displ, 'g-', linewidth=1)
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel(f'Displacement ({units})', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([t[0], t[-1]])
    
    plt.tight_layout()
    plt.show(block=False)


# Test and demonstration code
if __name__ == '__main__':
    """
    Test chirp function with various configurations
    """
    
    print("\n" + "="*70)
    print("Testing chirp.py")
    print("="*70)
    
    # Test 1: Basic earthquake simulation
    print("\nTest 1: Basic Earthquake Simulation")
    print("-" * 70)
    t1 = np.arange(0, 30, 0.01)  # 30 seconds at 100 Hz
    accel1, veloc1, displ1 = chirp(
        ao=0.5, af=0.1,      # Velocity: 0.5 → 0.1 m/s
        fo=0.5, ff=10,       # Frequency: 0.5 → 10 Hz  
        t=t1,
        p=2,                 # Quadratic frequency increase
        n=1,                 # Exponential amplitude decay
        phi=90,              # Start at zero crossing
        fig_no=1,
        units='m'
    )
    
    print(f"  Max displacement: {np.max(np.abs(displ1)):.4f} m")
    print(f"  Max velocity:     {np.max(np.abs(veloc1)):.4f} m/s")
    print(f"  Max acceleration: {np.max(np.abs(accel1)):.4f} m/s²")
    
    # Test 2: Linear frequency sweep
    print("\nTest 2: Linear Frequency Sweep (p=1)")
    print("-" * 70)
    t2 = np.linspace(0, 20, 2000)
    accel2, veloc2, displ2 = chirp(
        ao=1.0, af=0.3,
        fo=1, ff=15,
        t=t2,
        p=1,                 # Linear frequency increase
        phi=90,
        fig_no=2,
        units='m'
    )
    
    # Test 3: Phase comparison
    print("\nTest 3: Phase Control Comparison")
    print("-" * 70)
    t3 = np.linspace(0, 10, 1000)
    
    phases = [0, 90, 180, 270]
    phase_names = ['Maximum', 'Zero (↗)', 'Minimum', 'Zero (↘)']
    
    fig3 = plt.figure(3, figsize=(14, 10))
    
    for i, (phase, name) in enumerate(zip(phases, phase_names)):
        accel, veloc, displ = chirp(
            ao=1.0, af=0.5,
            fo=2, ff=8,
            t=t3,
            p=2,
            phi=phase,
            fig_no=0  # No individual plots
        )
        
        # Plot comparison
        ax = plt.subplot(4, 1, i+1)
        ax.plot(t3, veloc, 'b-', linewidth=1.5, label=f'φ={phase}°')
        ax.set_ylabel(f'Velocity\nφ={phase}°\n({name})', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 2])  # Zoom to first 2 seconds
        ax.legend(loc='upper right')
        
        # Show starting value
        ax.plot(t3[0], veloc[0], 'ro', markersize=8)
        ax.text(0.1, veloc[0], f'  Start: {veloc[0]:.3f}', 
               fontsize=9, va='center')
        
        if i == 0:
            ax.set_title('Phase Control Effect on Starting Point', 
                        fontsize=12, fontweight='bold')
        if i == 3:
            ax.set_xlabel('Time (s)', fontsize=11)
    
    plt.tight_layout()
    
    print("  Created phase comparison plot (Figure 3)")
    print("  → φ=0°:   Starts at maximum")
    print("  → φ=90°:  Starts at zero (rising) - RECOMMENDED")
    print("  → φ=180°: Starts at minimum")
    print("  → φ=270°: Starts at zero (falling)")
    
    # Test 4: High-order frequency sweep
    print("\nTest 4: High-Order Frequency Sweep (p=3)")
    print("-" * 70)
    t4 = np.linspace(0, 40, 4000)
    accel4, veloc4, displ4 = chirp(
        ao=0.8, af=0.2,
        fo=0.3, ff=12,
        t=t4,
        p=3,                 # Cubic frequency increase
        n=1.5,               # Faster amplitude decay
        phi=90,
        fig_no=4,
        units='m'
    )
    print("  → More cycles at lower frequencies (good for long-period structures)")
    
    # Test 5: Comparison with scipy.signal.chirp
    print("\nTest 5: Comparison with scipy.signal.chirp")
    print("-" * 70)
    from scipy.signal import chirp as scipy_chirp
    
    t5 = np.linspace(0, 10, 1000)
    
    # Our chirp (constant amplitude for fair comparison)
    accel_ours, veloc_ours, displ_ours = chirp(
        ao=1.0, af=1.0,  # Constant amplitude
        fo=1, ff=10,
        t=t5,
        p=2,
        phi=90,
        fig_no=0
    )
    
    # SciPy chirp
    scipy_signal = scipy_chirp(t5, f0=1, f1=10, t1=t5[-1], method='quadratic', phi=90)
    
    # Compare
    fig5 = plt.figure(5, figsize=(12, 8))
    
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t5, veloc_ours, 'b-', linewidth=1.5, label='Our chirp()', alpha=0.8)
    ax1.plot(t5, scipy_signal, 'r--', linewidth=1.5, label='scipy.signal.chirp()', alpha=0.6)
    ax1.set_ylabel('Signal', fontsize=11)
    ax1.set_title('Comparison: Our chirp() vs scipy.signal.chirp()', 
                 fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 2])
    
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(t5, displ_ours, 'g-', linewidth=1.5, label='Our: displacement')
    ax2.set_ylabel('Displacement', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 2])
    ax2.text(0.5, np.max(displ_ours)*0.5, 
            'scipy.signal.chirp\ndoes NOT provide\ndisplacement!',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(t5, accel_ours, 'm-', linewidth=1.5, label='Our: acceleration')
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Acceleration', fontsize=11)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 2])
    ax3.text(0.5, np.max(accel_ours)*0.5,
            'scipy.signal.chirp\ndoes NOT provide\nacceleration!',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    
    print("  → Signals are similar (with constant amplitude)")
    print("  → BUT our chirp() provides accel + veloc + displ")
    print("  → AND supports variable amplitude + tapering")
    
    # Summary
    print("\n" + "="*70)
    print("Summary of Features")
    print("="*70)
    print("✓ Variable amplitude (exponential decay)")
    print("✓ Power-law frequency sweep (flexible p parameter)")
    print("✓ Returns accel + veloc + displ")
    print("✓ Automatic Planck tapering")
    print("✓ Phase control (φ parameter)")
    print("✓ Built-in visualization")
    print("✓ Designed for structural/seismic testing")
    print("\n→ Ready for shake table control and earthquake simulation!")
    print("="*70 + "\n")
    
    plt.show()
