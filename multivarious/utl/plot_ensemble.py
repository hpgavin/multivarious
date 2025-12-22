"""
Ensemble Time Series Plotting

This module provides visualization for multiple time series stacked vertically
with vertical offsets, useful for comparing multiple realizations or channels
of data side-by-side.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union


def plot_ensemble(t: np.ndarray,
                  y1: Optional[np.ndarray] = None,
                  y2: Optional[np.ndarray] = None, 
                  y3: Optional[np.ndarray] = None,
                  fig: Optional[int] = None,
                  t_min: Optional[float] = None,
                  t_max: Optional[float] = None,
                  lbl: str = 'y',
                  ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Plot an ensemble of time series with vertical offsets for easy comparison.
    
    Each time series (row) is plotted with a vertical offset to prevent overlap.
    Up to three different datasets can be overlaid with different colors and
    line widths. Min/max values are marked for the primary dataset (y2).
    
    Parameters
    ----------
    t : ndarray
        Time vector (1D array of length N)
    y1 : ndarray, optional
        First dataset, shape (m1, N) where m1 is number of channels
        Plotted in blue, thin line
    y2 : ndarray, optional
        Second dataset, shape (m2, N) where m2 is number of channels
        Plotted in green, thick line (primary dataset)
    y3 : ndarray, optional
        Third dataset, shape (m3, N) where m3 is number of channels
        Plotted in purple, medium line
    fig : int, optional
        Figure number. If None, creates new figure
    t_min : float, optional
        Minimum time for x-axis. Default: t[0]
    t_max : float, optional
        Maximum time for x-axis. Default: t[-1]
    lbl : str, optional
        Label prefix for each channel. Default: 'y'
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes or creates new figure
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot
    
    Notes
    -----
    - Time series are stacked vertically with automatic spacing
    - Horizontal reference lines (dashed black) mark zero for each channel
    - Y-axis shows relative values around each channel's baseline
    - Min/max values are marked with circles and text for y2 dataset
    
    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 1000)
    >>> # Three channels of data
    >>> y1 = np.sin(2*np.pi*np.outer([1, 2, 3], t))
    >>> y2 = y1 + 0.2*np.random.randn(3, len(t))
    >>> fig = plot_ensemble(t, y1, y2, lbl='channel')
    """
    # Handle empty arrays
    if y1 is None:
        y1 = np.array([])
    if y2 is None:
        y2 = np.array([])
    if y3 is None:
        y3 = np.array([])
    
    # Ensure arrays are 2D
    if y1.size > 0 and y1.ndim == 1:
        y1 = y1.reshape(1, -1)
    if y2.size > 0 and y2.ndim == 1:
        y2 = y2.reshape(1, -1)
    if y3.size > 0 and y3.ndim == 1:
        y3 = y3.reshape(1, -1)
    
    # Determine number of channels
    m1 = y1.shape[0] if y1.size > 0 else 0
    m2 = y2.shape[0] if y2.size > 0 else 0
    m3 = y3.shape[0] if y3.size > 0 else 0
    m = max(m1, m2, m3)
    
    if m == 0:
        raise ValueError("At least one of y1, y2, or y3 must be provided")
    
    # Time parameters
    N = len(t)
    dt = t[1] - t[0]
    
    if t_min is None:
        t_min = t[0]
    if t_max is None:
        t_max = t[-1]
    
    # Find global min/max for scaling
    max_vals = []
    min_vals = []
    if y1.size > 0:
        max_vals.append(np.max(y1))
        min_vals.append(np.min(y1))
    if y2.size > 0:
        max_vals.append(np.max(y2))
        min_vals.append(np.min(y2))
    if y3.size > 0:
        max_vals.append(np.max(y3))
        min_vals.append(np.min(y3))
    
    yMax = max(max_vals)
    yMin = min(min_vals)
    y_range = 0.7 * abs(yMax - yMin)
    ly = round(y_range)
    
    # Create or get figure and axes
    if ax is None:
        if fig is not None:
            fig_obj = plt.figure(fig)
            plt.clf()
        else:
            fig_obj = plt.figure(figsize=(12, 2*m + 2))
        ax = fig_obj.gca()
    else:
        fig_obj = ax.figure
    
    # Initialize tick positions and labels
    yTickVal = []
    yTickLbl = []
    
    # Plot each channel with vertical offset
    for ii in range(m):
        dy = -y_range * ii  # Vertical offset
        
        # Plot horizontal reference line at zero
        ax.plot([t[0], t[-1]], [dy, dy], '--k', linewidth=1.5, zorder=1)
        
        # Plot y1 (blue, thin)
        if y1.size > 0 and ii < m1:
            ax.plot(t, y1[ii, :] + dy, '-', color=[0, 0.2, 0.8], 
                   linewidth=1.0, zorder=2)
        
        # Plot y2 (green, thick) - primary dataset
        if y2.size > 0 and ii < m2:
            ax.plot(t, y2[ii, :] + dy, '-', color=[0, 0.8, 0.2], 
                   linewidth=2.5, zorder=3)
        
        # Plot y3 (purple, medium)
        if y3.size > 0 and ii < m3:
            ax.plot(t, y3[ii, :] + dy, '-', color=[0.8, 0.2, 0.8], 
                   linewidth=2.0, zorder=4)
        
        # Add channel label
        ax.text(t_min, dy + y_range/7, f'{lbl}_{ii+1}', 
               fontsize=14, fontweight='bold', zorder=5)
        
        # Set up y-axis ticks for this channel
        yTickVal.extend([dy - ly/3, dy, dy + ly/3])
        yTickLbl.extend([f'{-ly/3:4.1f}', f'{0:3.0f}', f'{ly/3:4.1f}'])
        
        # Mark min/max for y2 (primary dataset)
        if y2.size > 0 and ii < m2:
            imin = np.argmin(y2[ii, :])
            imax = np.argmax(y2[ii, :])
            ymin = y2[ii, imin]
            ymax = y2[ii, imax]
            
            # Plot markers
            ax.plot(imin * dt, ymin + dy, 'ok', markersize=6, zorder=6)
            ax.plot(imax * dt, ymax + dy, 'ok', markersize=6, zorder=6)
            
            # Add text labels
            ax.text(imin * dt, ymin + dy, f'{ymin:+5.0f}', 
                   fontsize=10, ha='center', va='bottom', zorder=6)
            ax.text(imax * dt, ymax + dy, f'{ymax:+5.0f}', 
                   fontsize=10, ha='center', va='top', zorder=6)
    
    # Set axis properties
    ax.set_yticks(yTickVal)
    ax.set_yticklabels(yTickLbl)
    ax.set_xlim([t_min, t_max])
    ax.set_ylim([-(m - 0.5) * y_range, 0.5 * y_range])
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Disable clipping to allow labels outside plot area
    ax.set_clip_on(False)
    
    plt.tight_layout()
    
    return fig_obj


# ============================================================================
# Example usage
# ============================================================================
if __name__ == "__main__":
    """
    Demonstration of the plot_ensemble function with synthetic data.
    """
    # Enable interactive mode
    plt.ion()
    plt.rcParams['text.usetex'] = False  # Set to True if LaTeX is installed
    
    # ========================================================================
    # Generate synthetic ensemble data
    # ========================================================================
    
    # Time vector
    t = np.linspace(0, 10, 2000)
    dt = t[1] - t[0]
    
    # Number of channels
    m = 3
    
    # Base signal: combination of sinusoids
    frequencies = [1.0, 2.5, 0.5]  # Different frequency for each channel
    y_base = np.zeros((m, len(t)))
    for i in range(m):
        y_base[i, :] = (10 * np.sin(2 * np.pi * frequencies[i] * t) + 
                        5 * np.sin(2 * np.pi * frequencies[i] * 2 * t))
    
    # Dataset 1: Clean signals (theory/model)
    y1 = y_base.copy()
    
    # Dataset 2: Signals with moderate noise (experimental data)
    np.random.seed(42)
    y2 = y_base + 2.0 * np.random.randn(m, len(t))
    
    # Dataset 3: Signals with different characteristics (filtered)
    y3 = np.zeros_like(y_base)
    for i in range(m):
        # Simple moving average filter
        window = 20
        kernel = np.ones(window) / window
        y3[i, :] = np.convolve(y2[i, :], kernel, mode='same')
    
    # ========================================================================
    # Example 1: Plot all three datasets
    # ========================================================================
    fig1 = plot_ensemble(t, y1, y2, y3, 
                        fig=1,
                        t_min=0, 
                        t_max=10,
                        lbl='channel')
    fig1.suptitle('Ensemble Plot: Three Datasets Comparison', 
                  fontsize=16, fontweight='bold')
    
    # Add legend
    ax = fig1.gca()
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=[0, 0.2, 0.8], lw=1.0, label='Theory (y1)'),
        Line2D([0], [0], color=[0, 0.8, 0.2], lw=2.5, label='Experiment (y2)'),
        Line2D([0], [0], color=[0.8, 0.2, 0.8], lw=2.0, label='Filtered (y3)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # ========================================================================
    # Example 2: Plot single dataset with zoom
    # ========================================================================
    fig2 = plot_ensemble(t, y1=None, y2=y2, y3=None,
                        fig=2,
                        t_min=2.0,
                        t_max=4.0,
                        lbl='signal')
    fig2.suptitle('Ensemble Plot: Zoomed View (2-4 seconds)', 
                  fontsize=16, fontweight='bold')
    
    # ========================================================================
    # Example 3: Many channels (stress test)
    # ========================================================================
    t_long = np.linspace(0, 5, 1000)
    m_many = 8
    y_many = np.zeros((m_many, len(t_long)))
    for i in range(m_many):
        freq = 0.5 + i * 0.3
        y_many[i, :] = 15 * np.sin(2 * np.pi * freq * t_long) * np.exp(-0.2 * t_long)
        y_many[i, :] += 3 * np.random.randn(len(t_long))
    
    fig3 = plot_ensemble(t_long, y1=None, y2=y_many, y3=None,
                        fig=3,
                        lbl='response')
    fig3.suptitle('Ensemble Plot: 8 Channels with Decay', 
                  fontsize=16, fontweight='bold')
    fig3.set_size_inches(12, 12)
    
    plt.show()
    
    # ========================================================================
    # Print usage information
    # ========================================================================
    print("\n" + "="*70)
    print("PLOT_ENSEMBLE DEMONSTRATION")
    print("="*70)
    print(f"Generated {m} channels of synthetic data")
    print(f"Time vector: {len(t)} points, dt = {dt:.6f} s")
    print("\nPlot descriptions:")
    print("  Figure 1: Three datasets (theory, experiment, filtered)")
    print("  Figure 2: Single dataset with time zoom (2-4 seconds)")
    print("  Figure 3: Many channels (8) with exponential decay")
    print("\nFeatures demonstrated:")
    print("  - Vertical stacking with automatic spacing")
    print("  - Multiple dataset overlay with different colors/widths")
    print("  - Min/max value markers (circles and text)")
    print("  - Channel labels and custom y-axis ticks")
    print("  - Horizontal reference lines at zero")
    print("="*70 + "\n")
    
    # Keep plots open
    input("Press Enter to close plots and exit...")
    plt.close('all')
