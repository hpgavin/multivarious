import numpy as np
import matplotlib.pyplot as plt

def pz_plot(p, z, figno=1, dt=None):
    """
    Plot the poles and zeros of a MIMO system in the Laplace or Z domain.
    
    Parameters:
        p     : array of poles (complex numbers)
        z     : array of zeros (complex numbers)
        figno : figure number for plotting (default: 1)
        dt    : sample time (if provided, plots discrete-time poles/zeros
                with respect to unit circle; default: None for continuous-time)
    
    For continuous-time systems (dt=None):
        - Plots in s-plane (Laplace domain)
        - Stable poles are in left half-plane (Re < 0)
    
    For discrete-time systems (dt provided):
        - Plots in z-plane
        - Stable poles are inside unit circle (|z| < 1)
        - Shows unit circle boundary
    
    Author: H.P. Gavin, Duke Univ., 2017-10-07
    """
    
    p = np.asarray(p).flatten()
    z = np.asarray(z).flatten()
    
    ff = 1.20
    
    if dt is not None:
        # Discrete-time: plot with unit circle
        ax = [-1.1, 1.1, -1.1, 1.1]
    else:
        # Continuous-time: auto-scale based on pole/zero locations
        all_vals = np.concatenate([p, z]) if z.size > 0 else p
        
        if all_vals.size > 0:
            xr = (np.max(np.real(all_vals)) - np.min(np.real(all_vals))) / 2
            yr = (np.max(np.imag(all_vals)) - np.min(np.imag(all_vals))) / 2
            xa = (np.max(np.real(all_vals)) + np.min(np.real(all_vals))) / 2
            ya = (np.max(np.imag(all_vals)) + np.min(np.imag(all_vals))) / 2
            
            ax = [xa - ff*xr, max(xa + ff*xr, 0.1*xr), 
                  ya - ff*yr, ya + ff*yr]
        else:
            ax = [-1, 1, -1, 1]
    
    # Create figure
    plt.figure(figno)
    plt.clf()
    
    # Plot poles (red x) and zeros (blue o)
    if p.size > 0:
        plt.plot(np.real(p), np.imag(p), 'xr', markersize=10, 
                markeredgewidth=2, label='Poles')
    if z.size > 0:
        plt.plot(np.real(z), np.imag(z), 'ob', markersize=8, 
                markerfacecolor='none', markeredgewidth=2, label='Zeros')
    
    # Set axis limits and make square
    plt.axis(ax)
    plt.axis('square')
    
    # Draw axes through origin
    plt.plot([ax[0], ax[1]], [0, 0], '-k', linewidth=1)
    plt.plot([0, 0], [ax[2], ax[3]], '-k', linewidth=1)
    
    # Draw unit circle for discrete-time
    if dt is not None:
        theta = np.linspace(0, 2*np.pi, 361)
        plt.plot(np.cos(theta), np.sin(theta), '-k', linewidth=1)
    
    # Formatting
    ax_obj = plt.gca()
    ax_obj.spines['left'].set_position('zero')
    ax_obj.spines['bottom'].set_position('zero')
    ax_obj.spines['right'].set_color('none')
    ax_obj.spines['top'].set_color('none')
    ax_obj.spines['left'].set_linewidth(2)
    ax_obj.spines['bottom'].set_linewidth(2)
    
    # Labels
    if dt is not None:
        plt.text(0.7, -0.1, r'Re($\lambda_i$)', fontsize=12)
        plt.text(0.1, 0.7, r'Im($\lambda_i$)', fontsize=12, rotation=90)
        plt.title('Pole-Zero Map (Discrete-Time)', fontsize=14, pad=20)
    else:
        plt.text(0.8*ax[0], 0.1*ax[3], r'Re($\lambda_i$) = $\zeta_i \omega_{ni}$', fontsize=11)
        plt.text(0.5*ax[1], 0.5*ax[3], r'Im($\lambda_i$) = $\omega_{di}$', 
                fontsize=11, rotation=90)
        plt.title('Pole-Zero Map (Continuous-Time)', fontsize=14, pad=20)
    
    if p.size > 0 or z.size > 0:
        plt.legend(loc='upper right')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


# --------------------------------------------------------------------------
# H.P. Gavin, Duke Univ., 2017-10-07

# Example usage and test
if __name__ == "__main__":
    print("Testing pz_plot: Pole-Zero Plotting\n")
    print("="*60)
    
    # Example 1: Continuous-time system
    print("\nExample 1: Continuous-time system (damped oscillator)")
    print("-"*60)
    
    # Second-order system: (s^2 + 2*zeta*wn*s + wn^2)
    # Poles: s = -zeta*wn ± j*wn*sqrt(1-zeta^2)
    wn = 10    # natural frequency
    zeta = 0.1 # damping ratio
    
    wd = wn * np.sqrt(1 - zeta**2)  # damped frequency
    p_cont = np.array([-zeta*wn + 1j*wd, -zeta*wn - 1j*wd])
    z_cont = np.array([])  # no zeros
    
    print(f"Natural frequency: {wn} rad/s")
    print(f"Damping ratio: {zeta}")
    print(f"Poles: {p_cont}\n")
    
    pz_plot(p_cont, z_cont, figno=1)
    
    # Example 2: Continuous-time with zeros
    print("\n" + "="*60)
    print("\nExample 2: Continuous-time with zeros")
    print("-"*60)
    
    # Transfer function: (s+2) / ((s+1)(s+3))
    p_cont2 = np.array([-1, -3])
    z_cont2 = np.array([-2])
    
    print(f"Poles: {p_cont2}")
    print(f"Zeros: {z_cont2}\n")
    
    pz_plot(p_cont2, z_cont2, figno=2)
    
    # Example 3: Discrete-time system
    print("\n" + "="*60)
    print("\nExample 3: Discrete-time system")
    print("-"*60)
    
    # Discrete poles inside unit circle (stable)
    dt = 0.1
    p_disc = np.array([0.8 + 0.2j, 0.8 - 0.2j, 0.5])
    z_disc = np.array([0.3 + 0.1j, 0.3 - 0.1j])
    
    print(f"Sample time: {dt} s")
    print(f"Poles: {p_disc}")
    print(f"Pole magnitudes: {np.abs(p_disc)}")
    print(f"Zeros: {z_disc}\n")
    
    if np.all(np.abs(p_disc) < 1):
        print("✓ System is stable (all poles inside unit circle)")
    else:
        print("✗ System is unstable (poles outside unit circle)")
    
    pz_plot(p_disc, z_disc, figno=3, dt=dt)
    
    # Example 4: Unstable discrete-time system
    print("\n" + "="*60)
    print("\nExample 4: Unstable discrete-time system")
    print("-"*60)
    
    p_unstable = np.array([1.2 + 0.3j, 1.2 - 0.3j, -0.5])
    z_unstable = np.array([0.6])
    
    print(f"Poles: {p_unstable}")
    print(f"Pole magnitudes: {np.abs(p_unstable)}")
    
    if np.all(np.abs(p_unstable) < 1):
        print("✓ System is stable")
    else:
        print("✗ System is unstable (poles outside unit circle)")
    
    pz_plot(p_unstable, z_unstable, figno=4, dt=0.1)
    
    # Example 5: High-order system
    print("\n" + "="*60)
    print("\nExample 5: High-order continuous-time system")
    print("-"*60)
    
    # Multiple complex conjugate pole pairs
    p_high = np.array([
        -1 + 5j, -1 - 5j,
        -2 + 3j, -2 - 3j,
        -3 + 1j, -3 - 1j,
        -5
    ])
    z_high = np.array([-0.5 + 2j, -0.5 - 2j])
    
    print(f"Number of poles: {len(p_high)}")
    print(f"Number of zeros: {len(z_high)}")
    
    pz_plot(p_high, z_high, figno=5)
    
    plt.show()
    
    print("\n" + "="*60)
    print("\nAll pole-zero plots generated successfully!")
    print("Close the plot windows to continue...")
    print("="*60)
