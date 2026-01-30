#! /usr/bin/env -S python3 -i
"""
Demonstration of linear elastic response spectrum calculation.

This script demonstrates the use of lers_2d() to compute response spectra
for earthquake ground motion records.
"""

import numpy as np
import matplotlib.pyplot as plt
from multivarious.dsp import lers_2d
from multivarious.dsp import eqgm_1d
from multivarious.utl import format_plot

# Set of natural periods to evaluate
Tn = np.array([0.01, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])

# Damping ratio
zz = 0.05  # 5% damping

# Gravitational acceleration
g = 9.81  # m/s²

# Example: Simple sinusoidal ground motion
# In practice, you would load actual earthquake records or use
# synthetic ground motion generators like eqgm_1D()

T  = 25.0  # duration (s)
dt = 0.01  # time step (s)
n  = int(T / dt)
t  = np.arange(n) * dt

# Create simple example ground motion (replace with actual data)

ax, _, _, _, _, _, _ = eqgm_1d(PGA=3.5, fg=1.5, zg=0.9, aa=4.0, ta=2.0, t=None, fig_no=1, seed=None)
ay, _, _, _, _, _, _ = eqgm_1d(PGA=3.5, fg=1.5, zg=0.9, aa=4.0, ta=2.0, t=None, fig_no=1, seed=None)

# Compute response spectrum using square root of the sum of the squares (SRSS) method
PSA_srss, SD_srss = lers_2d(ax, ay, t, g, Tn, zz, method='SRSS', fig_no=10, save_plot=False)

print("Response Spectrum Results (SRSS method):")
print(f"{'Period (s)':<12} {'PSA (g)':<12} {'SD (m)':<12}")
print("-" * 36)
for i, Tn_val in enumerate(Tn):
    print(f"{Tn_val:<12.2f} {PSA_srss[i]:<12.4f} {SD_srss[i]:<12.4f}")

# Compute response spectrum using geometric mean (GM) method
PSA_gm, SD_gm = lers_2d(ax, ay, t, g, Tn, zz, method='GM', fig_no=20, save_plot=False)

print("\nResponse Spectrum Results (GM method):")
print(f"{'Period (s)':<12} {'PSA (g)':<12} {'SD (m)':<12}")
print("-" * 36)
for i, Tn_val in enumerate(Tn):
    print(f"{Tn_val:<12.2f} {PSA_gm[i]:<12.4f} {SD_gm[i]:<12.4f}")

format_plot(font_size=15, line_width=2, marker_size=7)

# Compare methods
plt.figure(5, figsize=(6,6))
plt.clf()
plt.subplot(2, 1, 1)
plt.plot(Tn, PSA_srss, '-o', label='SRSS', linewidth=2)
plt.plot(Tn, PSA_gm, '-s', label='GM', linewidth=2)
plt.ylabel(r'$S_A$, g')
plt.xlabel(r'$T_n$, s')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Comparison of SRSS and GM Methods')

plt.subplot(2, 1, 2)
plt.plot(Tn, SD_srss, '-o', label='SRSS', linewidth=2)
plt.plot(Tn, SD_gm, '-s', label='GM', linewidth=2)
plt.ylabel(r'$S_D$, m')
plt.xlabel(r'$T_n$, s')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

