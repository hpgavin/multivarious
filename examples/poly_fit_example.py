#! /usr/bin/env -S python3 -i
"""
poly_fit_example.py - example for the use of poly_fit.py
2025-12-02
"""

import numpy as np
import matplotlib.pyplot as plt
from multivarious.fit import poly_fit
import time

"""
Test poly_fit with example data
"""
print("\n" + "="*70)
print("Testing poly_fit.py")
print("="*70)
    
# Generate test data
rng = np.random.default_rng(42)                        #  the same seed each time 
rng = np.random.default_rng((int)(time.time()*100000)) # different seed each time
x_l =  -1  # Lower end of the fit domain
x_h =   1  # Upper end of the fit domain
Nd  =  40  # Number of data points
    
x = np.linspace(x_l, x_h, Nd)  # the independent data points from x_l to x_h
    
measurement_error = 0.20   # the level of simulated measurement error 
# simulate the measured data with a latent model plus the measurement error 
y = -np.cos(4*x) + x**3 * np.exp(-x/3) + measurement_error * rng.standard_normal(Nd)
    
print(f"\nGenerated {Nd} data points")
print(f"x range: [{x_l}, {x_h}]")
print(f"Measurement error (RMS): {measurement_error}")
    
# Test 1: Full polynomial basis
print("\n" + "-"*70)
print("Test 1: Powers [0, 1, 2, 3, 4]")
print("-"*70)
    
p1 = [ 0, 1, 2, 3, 4 ]  # the powers of the polynomial terms 
# ... run poly_fit() to do the fit! ...
c1, x_fit1, y_fit1, Sc1, Sy_fit1, Rc1, R2_1, Vr1, AIC1, BIC1, cond1 = \
        poly_fit(x, y, p1, fig_no=10)
    
# Test 2: Reduced polynomial basis (no linear term)
print("\n" + "-"*70)
print("Test 2: Powers [0, 2, 3, 4]")
print("-"*70)
    
p2 = [ 0, 2, 3, 4 ]  # the powers of the polynomial terms without an "x" term
c2, x_fit2, y_fit2, Sc2, Sy_fit2, Rc2, R2_2, Vr2, AIC2, BIC2, cond2 = \
        poly_fit(x, y, p2, fig_no=20)
    
# Comparison
print("\n" + "="*70)
print("Comparison of Models")
print("="*70)
print(f"{'Metric':<28} {'Model 1 (5 terms)':<20} {'Model 2 (4 terms)':<20}")
print("-"*70)
print(f"{'Cond. Number':<28} {cond1:<20.1f} {cond2:<20.1f}")
print(f"{'Residual Standard Deviation':<28} {np.sqrt(Vr1):<20.4f} {np.sqrt(Vr2):<20.4f}")
print(f"{'R²':<28} {R2_1:<20.4f} {R2_2:<20.4f}")
print(f"{'AIC':<28} {AIC1:<20.2f} {AIC2:<20.2f}")
print(f"{'BIC':<28} {BIC1:<20.2f} {BIC2:<20.2f}")
print("="*70)

if AIC2 < AIC1:
    print("\n✓ Model 2 is preferred (lower AIC)")
else:
    print("\n✓ Model 1 is preferred (lower AIC)")
    
plt.show()
    
print("\n" + "="*70)
