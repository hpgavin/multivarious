"""
eqgm_1d.py - Synthetic Earthquake Ground Motion Generator
==========================================================

Generate artificial (synthetic) earthquake ground motion records as the
response of a linear time-invariant system driven by filtered white noise
with a prescribed temporal envelope.

Translation from MATLAB to Python, 2025-11-24
Original by H.P. Gavin, Duke University, 2007-2020
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_lyapunov
from multivarious.lti.lsym import lsym
from multivarious.lti.dlsym import dlsym
from multivarious.dsp.butter_synth_ss import butter_synth_ss
from multivarious.dsp.accel2displ import accel2displ
from multivarious.dsp.taper import taper 


def eqgm_1d(PGA=3.5, fg=1.5, zg=0.9, aa=4.0, ta=2.0, t=None, fig_no=0, seed=None):
    """
    Generate synthetic earthquake ground motion record.
    
    Creates a realistic artificial earthquake acceleration time history
    by filtering white noise through a second-order system (representing
    ground dynamics) and applying a temporal envelope function.
    
    Parameters
    ----------
    PGA : float, optional
        Peak ground acceleration (m/s²), default: 3.5
    fg : float, optional
        Ground frequency (Hz), default: 1.5
    zg : float, optional
        Ground damping ratio, default: 0.9
    aa : float, optional
        Envelope rise-time parameter, default: 4.0
    ta : float, optional
        Envelope decay time constant (s), default: 2.0
    t : ndarray, optional
        Time vector (s), default: np.arange(1, 3001) * 0.01
    fig_no : int, optional
        Figure number for plotting (0 = no plots), default: 0
    seed : int, optional
        Random seed for reproducibility, default: None (random)
    
    Returns
    -------
    accel : ndarray, shape (1, n)
        Simulated earthquake ground acceleration (m/s²)
    veloc : ndarray, shape (1, n)
        Simulated earthquake ground velocity (m/s)
    displ : ndarray, shape (1, n)
        Simulated earthquake ground displacement (m)
    scale : float
        Scaling factor applied (currently returns 1.0)
    Ag : ndarray, shape (2, 2)
        Ground motion model state matrix
    Bg : ndarray, shape (2, 1)
        Ground motion model input matrix
    Cg : ndarray, shape (1, 2)
        Ground motion model output matrix
    
    Notes
    -----
    **Suggested Ground Motion Parameters:**
    
    Based on ATC-63 data sets:
    
    ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬────────────┐
    │ PGA  │ PGV  │  fg  │  zg  │  aa  │  ta  │ P2R  │  Dataset   │
    ├──────┼──────┼──────┼──────┼──────┼──────┼──────┼────────────┤
    │ 3.5  │ 0.33 │ 1.5  │ 0.9  │ 4.0  │ 2.0  │ 2.7  │ ATC-63-FF  │
    │ 5.0  │ 0.52 │ 1.3  │ 1.1  │ 3.0  │ 2.0  │ 2.7  │ ATC-63-NFNP│
    │ 4.7  │ 0.80 │ 0.5  │ 1.8  │ 1.0  │ 2.0  │ 2.5  │ ATC-63-NFP │
    └──────┴──────┴──────┴──────┴──────┴──────┴──────┴────────────┘
    
    FF = Far-Field
    NFNP = Near-Field, No Pulse
    NFP = Near-Field, Pulse
    
    **Ground Motion Model:**
    
    The acceleration is generated as the output of:
        ẍ + 4πζ_g f_g ẋ + (2πf_g)²x = a_bar · w(t)
    
    where w(t) is band-limited white noise (0-25 Hz) with envelope.
    
    **Envelope Function:**
    
    Combines exponential rise/decay with Planck tapers:
    
    The envelope shapes the temporal evolution to match real earthquakes:
    - Initial buildup (rise time ∝ aa)
    - Peak intensity
    - Gradual decay (time constant ta)
    
    **Bias and Drift Removal:**
    
    Uses subtract running average (SRA) method to ensure realistic
    baseline correction, similar to strong motion processing.
    
    Examples
    --------
    >>> import numpy as np
    >>> from eqgm_1d import eqgm_1d
    >>> 
    >>> # Example 1: Far-field ground motion
    >>> accel, veloc, displ, scale, Ag, Bg, Cg = eqgm_1d(
    ...     PGA=3.5, fg=1.5, zg=0.9, aa=4.0, ta=2.0, fig_no=1
    ... )
    >>> 
    >>> # Example 2: Near-field pulse-like ground motion
    >>> t = np.arange(1, 5001) * 0.01  # 50 seconds
    >>> accel, veloc, displ, _, _, _, _ = eqgm_1d(
    ...     PGA=4.7, fg=0.5, zg=1.8, aa=1.0, ta=2.0, t=t, fig_no=2
    ... )
    >>> 
    >>> # Example 3: Reproducible simulation with seed
    >>> accel, veloc, displ, _, _, _, _ = eqgm_1d(
    ...     PGA=5.0, fg=1.3, zg=1.1, seed=42, fig_no=0
    ... )
    
    See Also
    --------
    lsym : Continuous-time linear system simulation
    butter_synth_ss : Butterworth filter synthesis
    accel2displ : Acceleration baseline correction
    
    References
    ----------
    [1] ATC-63, "Quantification of Building Seismic Performance Factors",
        Applied Technology Council, 2008
    [2] H.P. Gavin, "Earthquake Ground Motion Simulation", Duke University
    """
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Default time vector
    if t is None:
        t = np.arange(0, 3001) * 0.01  # 30 seconds, dt=0.01
    
    t = np.asarray(t).flatten()
    dt = t[1] - t[0]  # Time step
    P = len(t)        # Number of points
    
    # Envelope parameters
    t0 = 2.0  # Initial time of no motion (s)
    T = t0 + 5.74 * aa**0.42 * ta  # Total duration
    
    # Check if time series is long enough
    if np.max(t) < T and fig_no > 0:
        P_needed = int(np.floor(T / dt))
        print(f'eqgm_1d: time series may not be long enough, '
              f'T={T:.0f}, P_needed={P_needed}')
    
    Pz = int(np.floor(t0 / dt))  # Zero-motion points
    Pt = int(np.round(2 * Pz))   # Taper points
    
    # ========== Envelope Function ==========
    # Exponential rise and decay with cosine tapers
    
    envlp = np.zeros(P)
    
    # Main envelope: exponential rise and decay
    for p in range(Pz-1, P):
        tau = (t[p] - t0) / (aa * ta)
        envlp[p] = tau**aa * np.exp(aa - (t[p] - t0) / ta)
    
    # Add small backgorund seismic noise 
    envlp = envlp + 0.005 * PGA 
    
    # Apply (infinitely smooth) Planck tapers at beginning and at end
    envlp = taper(envlp, Ni=Pt, Nf=Pt, window='planck')[0]

    # ========== Peak-to-RMS Model ==========
    # Empirical model for peak-to-RMS ratio based on fg, ta, zg, aa
    
    Tpk = aa * ta
    tafg = Tpk * fg
    lzg = np.log10(zg)
    
    # Regression coefficients
    c = np.array([
        2.0327e+00,
        2.2535e-01,
        1.9215e-02,
        2.7183e-01,
        2.3593e-02,
       -3.1858e-02,
       -4.4708e-03,
       -2.4680e-02,
       -6.3575e-04,
       -2.8971e-01
    ])
    
    # Design matrix for P2R prediction
    B = np.array([1, tafg, aa, lzg, tafg*aa, tafg*lzg, aa*lzg, 
                  tafg**2, aa**2, lzg**2])
    
    P2R = B @ c  # Peak-to-RMS ratio
    
    RMSa = PGA / P2R
    a_bar = PGA / P2R / np.sqrt(2 * np.pi * zg * fg)
    
    # ========== Linear State-Space Model ==========
    # Second-order system: ẍ + 2ζωₙẋ + ωₙ²x = a_bar·w
    
    wg = 2 * np.pi * fg  # Ground frequency (rad/s)
    
    Ag = np.array([[0.0, 1.0],
                   [-wg**2, -2*wg*zg]])
    Bg = np.array([[0.0],
                   [a_bar]])
    Cg = np.array([[0.0, 2*wg*zg]])
    
    # Verify RMS acceleration using Lyapunov equation
    # A*Σ + Σ*A' + B*B' = 0
    Sigma = solve_continuous_lyapunov( Ag , -Bg @ Bg.T )
    RMSa_check = np.sqrt(Cg @ Sigma @ Cg.T)[0, 0]
    
    if fig_no > 0:
        print(f'Target RMS accel: {RMSa:.4f} m/s²')
        print(f'Computed RMS accel: {RMSa_check:.4f} m/s²')
    
    # ========== Generate White Noise ==========
    # Unit Gaussian white noise has PSD = 1.0 and variance = 1/dt
    
    u = np.random.randn(P) / np.sqrt(dt)
    
    # Taper the start and end of noise
    nR = int(np.round(2 / dt))
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(1, nR+1) / nR))
    u[:nR] = ramp * u[:nR]
    u[-nR:] = ramp[::-1] * u[-nR:]
    
    # ========== Band-Limited White Noise (0-25 Hz) ==========
    # Use 9th-order Butterworth low-pass filter
    
    A_filt, B_filt, C_filt, D_filt, _ = butter_synth_ss(
        N=9, fc=25.0, fs=1/dt, filter_type='low'
    )
    
    # Forward-backward filtering for zero phase distortion
    u = dlsym(A_filt, B_filt, C_filt, D_filt, u[np.newaxis, :], t).flatten()
    u = dlsym(A_filt, B_filt, C_filt, D_filt, u[::-1][np.newaxis, :], t).flatten()
    u = u[::-1]
    
    # ========== Apply Envelope ==========
    u = envlp * u
    
    # ========== Pass Through Ground Motion Model ==========
    accel = lsym(Ag, Bg, Cg, np.array([[0.0]]), u[np.newaxis, :], t, ntrp='foh')
    
    # ========== Remove Bias and Drift ==========
    # Find time when acceleration drops below 98% of peak
    t0_drift = 1.2 * np.max(np.where(np.abs(accel) / np.max(np.abs(accel)) > 0.98)[1]) * dt
    ta_drift = 0.3
    
    accel, veloc, displ  = accel2displ(accel, t, method='SRA', aa=0.5*dt, 
                               t0=t0_drift, tt=ta_drift)
    
    # Compute peak values
    Amax = np.max(np.abs(accel))
    Vmax = np.max(np.abs(veloc))
    Dmax = np.max(np.abs(displ))
    
    scale = 1.0  # Placeholder for optional scaling
    
    # ========== Plotting ==========
    if fig_no > 0:
        T_plot = min(30, np.max(t))  # Maximum time to display
        idx_plot = t <= T_plot
        
        # Figure 1: Envelope visualization
        plt.figure(fig_no + 1, figsize=(12, 6))
        plt.clf()
        
        idx_max = np.argmax(np.abs(accel))
        
        plt.plot(t, accel.flatten(), 'b-', linewidth=1.5, label='Acceleration')
        plt.plot(t, envlp * RMSa, 'r-', linewidth=2, alpha=0.7, label='RMS envelope')
        plt.plot(t, -envlp * RMSa, 'r-', linewidth=2, alpha=0.7)
        
        plt.axhline(RMSa, color='k', linestyle='--', linewidth=1, label='RMS')
        plt.axhline(-RMSa, color='k', linestyle='--', linewidth=1)
        plt.axhline(PGA, color='g', linestyle='--', linewidth=1.5, label='PGA')
        plt.axhline(-PGA, color='g', linestyle='--', linewidth=1.5)
        
        plt.plot(t[idx_max], accel.flatten()[idx_max], '*', 
                markersize=15, color='green', markeredgecolor='darkgreen',
                markeredgewidth=2)
        
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Acceleration (m/s²)', fontsize=12)
        plt.title(f'Synthetic Earthquake Ground Motion (PGA={PGA:.2f} m/s²)', 
                 fontsize=13, fontweight='bold')
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, T_plot])
        plt.tight_layout()
        
        # Figure 2: Three-panel plot
        plt.figure(fig_no, figsize=(12, 10))
        plt.clf()
        
        plt.subplot(311)
        plt.plot(t[idx_plot], Amax * envlp[idx_plot], '-r', linewidth=1.5, alpha=0.7)
        plt.plot(t[idx_plot], -Amax * envlp[idx_plot], '-r', linewidth=1.5, alpha=0.7)
        plt.plot(t[idx_plot], accel.flatten()[idx_plot], '-b', linewidth=1)
        plt.ylabel('Accel (m/s²)', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, T_plot])
        
        plt.subplot(312)
        plt.plot(t[idx_plot], veloc.flatten()[idx_plot], '-b', linewidth=1)
        plt.ylabel('Veloc (m/s)', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, T_plot])
        
        plt.subplot(313)
        plt.plot(t[idx_plot], displ.flatten()[idx_plot], '-b', linewidth=1)
        plt.ylabel('Displ (m)', fontsize=11)
        plt.xlabel('Time (s)', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim([0, T_plot])
        
        plt.tight_layout()
        
        # Print summary
        print('\n' + '='*60)
        print('Synthetic Earthquake Ground Motion Summary')
        print('='*60)
        print(f'Peak Ground Acceleration:  {Amax:.4f} m/s²')
        print(f'Peak Ground Velocity:      {Vmax:.4f} m/s')
        print(f'Peak Ground Displacement:  {Dmax:.4f} m')
        print(f'Ground frequency:          {fg:.2f} Hz')
        print(f'Ground damping:            {zg:.3f}')
        print(f'Envelope rise parameter:   {aa:.2f}')
        print(f'Envelope decay time:       {ta:.2f} s')
        print(f'Peak-to-RMS ratio:         {P2R:.3f}')
        print('='*60 + '\n')
    
    return accel, veloc, displ, scale, Ag, Bg, Cg


# ============================================================================

# Test and demonstration code
if __name__ == '__main__':
    """
    Test eqgm_1d function with various parameter sets
    """
    print("\n" + "="*70)
    print("Testing eqgm_1d.py - Synthetic Earthquake Ground Motion")
    print("="*70)
    
    # Test 1: Far-field ground motion (ATC-63-FF)
    print("\nTest 1: Far-Field Ground Motion (ATC-63-FF)")
    print("-" * 70)
    
    accel, veloc, displ, scale, Ag, Bg, Cg = eqgm_1d(
        PGA=3.5, fg=1.5, zg=0.9, aa=4.0, ta=2.0,
        fig_no=1, seed=42
    )
    
    # Test 2: Near-field pulse (ATC-63-NFP)
    print("\nTest 2: Near-Field Pulse Ground Motion (ATC-63-NFP)")
    print("-" * 70)
    
    t_long = np.arange(1, 5001) * 0.01  # 50 seconds
    
    accel2, veloc2, displ2, _, _, _, _ = eqgm_1d(
        PGA=4.7, fg=0.5, zg=1.8, aa=1.0, ta=2.0,
        t=t_long, fig_no=3, seed=123
    )
    
    plt.show()
    
    print("\n" + "="*70)
