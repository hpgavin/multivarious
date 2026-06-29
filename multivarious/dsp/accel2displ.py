#! /usr/bin/env -S python3 -i 
import numpy as np
from scipy.integrate import cumulative_trapezoid

def accel2displ(accel, t, method='SRA', aa=None, t0=None, tt=None):
    """
    Remove bias and drift from an acceleration time series.
    
    Parameters:
        accel  : acceleration time series as a row vector (n x N) or (N,)
        t      : time vector (n,) 
        method : 'SRA' - subtract running average from accel (default)
                 'ZFV' - correct accel for nearly zero final displ and velocity
        aa     : forgetting factor for SRA (default: 0.5*dt)
                 with aa = 1.0*dt, response time of filter is  5 s
                 with aa = 0.5*dt, response time of filter is 10 s
                 with aa = 0.2*dt, response time of filter is 25 s
                 with aa = 0.1*dt, response time of filter is 50 s
        t0     : midpoint for ZFV (default: T/2)
        tt     : timespan for ZFV (default: T/2)
    
    Returns:
        accel : acceleration with bias and drift reduced    (n,N) or (N,)
        veloc : velocity by trapezoidal rule from accel     (n,N) or (N,)
        displ : displacement by trapezoidal rule from accel (n,N) or (N,)
    
    Author: HP Gavin, 2023-01-23
    """
    
    # Convert to numpy arrays
    accel = np.asarray(accel)
    t = np.asarray(t)
    
    # Handle 1D input - ensure row vector (1 x n)
    if accel.ndim == 1:
        accel = accel[np.newaxis, :]
    
    nA, nT = accel.shape  # number of channels, number of data points
    dt = t[1] - t[0]      # time step value
    T = t[-1]             # time duration of signal
    
    # Set defaults
    if aa is None:
        aa = 1.0 * dt
    if t0 is None:
        t0 = T / 2
    if tt is None:
        tt = T / 2

    aa = min( aa, 0.9 )  # forgetting factor must be 0 < aa < 1
    
    # Make a copy to avoid modifying input
    accel = accel.copy()
    
    if method == 'SRA':   # Subtract a running average of the acceleration
        
        avgA = accel[:,0] # initialize running average = initial acceleration
        
        for k in range(1, nT):
            avgA = (1 - aa) * avgA + aa * accel[:, k] #  update  running average
            accel[:, k] = accel[:, k] - avgA          # subtract running average
    
    else:  # 'ZFV' - correct acceleration so final displ and veloc are near zero
        
        exp_ts = np.exp(-(t - t0) / tt)  # negative exponent of scaled time
        
        ac = 1.0 / (1 + exp_ts)  # sigmoidal logistic function
        vc = exp_ts * (1 + exp_ts)**(-2.0) / tt  # d(ac)/dt
        dc = exp_ts * (1 + exp_ts)**(-3.0) * (exp_ts - 1) / tt**2  # d(vc)/dt
        
        for iter in range(50):  # iteratively converge on acceleration correction
            velT = np.sum(accel, axis=1, keepdims=True) * dt
            dspT = (accel @ np.arange(nT, 0, -1)[:, np.newaxis]) * dt**2
            accel = accel - velT * vc - dspT * dc
            
            if np.abs(velT) < 0.01 and np.abs(dspT) < 0.01:
                break
    
    # Compute velocity from acceleration via cumulative trapezoidal integration
    veloc = cumulative_trapezoid(accel, axis=1, initial=0) * dt
    
    # Compute displacement from velocity via cumulative trapezoidal integration
    displ = cumulative_trapezoid(veloc, axis=1, initial=0) * dt

    # return a (N,) array if accel is a 1-dimensional array.  
    if accel.ndim == 2 and accel.shape[0] == 1:
        accel = accel[0,:]
        veloc = veloc[0,:]
        displ = displ[0,:]

    return accel, veloc, displ


# ------------------------------------------------------------------------
# HP Gavin 2023-01-23

# Example usage and test
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Generate synthetic acceleration data with bias and drift
    t = np.linspace(0, 10, 1000)
    dt = t[1] - t[0]
    
    # True acceleration (sinusoidal)
    accel_true = np.sin(2 * np.pi * 1.0 * t)
    
    # Add bias and linear drift
    bias = 0.1
    drift = 0.02 * t
    accel_noisy = accel_true + bias + drift
    
    # Test SRA method
    accel_sra, veloc_sra, displ_sra = accel2displ(accel_noisy, t, method='SRA', aa=1.5*dt)
    
    # Test ZFV method
    accel_zfv, veloc_zfv, displ_zfv = accel2displ(accel_noisy, t, method='ZFV')
    
    # Plot results
    plt.ion()
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    # SRA method
    axes[0, 0].plot(t, accel_noisy, 'r-', alpha=0.5, label='Noisy')
    axes[0, 0].plot(t, accel_sra, 'b-', label='Corrected (SRA)')
    axes[0, 0].set_ylabel('Acceleration')
    axes[0, 0].set_title('SRA Method')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[1, 0].plot(t, veloc_sra)
    axes[1, 0].set_ylabel('Velocity')
    axes[1, 0].grid(True)
    
    axes[2, 0].plot(t, displ_sra)
    axes[2, 0].set_ylabel('Displacement')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].grid(True)
    
    # ZFV method
    axes[0, 1].plot(t, accel_noisy, 'r-', alpha=0.5, label='Noisy')
    axes[0, 1].plot(t, accel_zfv, 'b-', label='Corrected (ZFV)')
    axes[0, 1].set_ylabel('Acceleration')
    axes[0, 1].set_title('ZFV Method')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 1].plot(t, veloc_zfv)
    axes[1, 1].set_ylabel('Velocity')
    axes[1, 1].grid(True)
    
    axes[2, 1].plot(t, displ_zfv)
    axes[2, 1].set_ylabel('Displacement')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nSRA Method - Final values:")
    print(f"  Velocity: {veloc_sra[-1]:.6f}")
    print(f"  Displacement: {displ_sra[-1]:.6f}")
    
    print(f"\nZFV Method - Final values:")
    print(f"  Velocity: {veloc_zfv[-1]:.6f}")
    print(f"  Displacement: {displ_zfv[-1]:.6f}")
