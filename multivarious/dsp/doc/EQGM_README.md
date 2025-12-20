# Synthetic Earthquake Ground Motion Generator

## Overview

This package generates realistic artificial earthquake ground motion records using digital signal processing and linear systems theory. The approach models ground motion as the response of a linear time-invariant system driven by filtered white noise with a prescribed temporal envelope.

## Physical Basis

### Ground Motion Model

The acceleration time history is generated as the output of a **second-order linear system**:

```
ẍ + 4πζ_g f_g ẋ + (2πf_g)² x = a_bar · w(t)
```

where:

- `x` = ground displacement
- `ẋ` = ground velocity  
- `ẍ` = ground acceleration (output)
- `f_g` = characteristic ground frequency (Hz)
- `ζ_g` = ground damping ratio
- `w(t)` = band-limited white noise input (0-25 Hz)
- `a_bar` = scaling factor to achieve target RMS acceleration

This models the ground as a **single-degree-of-freedom oscillator** representing the dominant soil response characteristics.

### State-Space Representation

In state-space form with states `[x, ẋ]ᵀ`:

```python
A = [    0          1     ]
    [ -ω_g²    -2ζ_g ω_g  ]

B = [  0   ]
    [ a_bar ]

C = [ 0,  2ζ_g ω_g ]
```

where `ω_g = 2πf_g` is the angular frequency.

The **output is velocity**, which when differentiated gives acceleration. However, the implementation directly outputs acceleration through the choice of the LTI system output %   X = LIAP(A,B,C) solves the general form of the Liapunov matrix
%   equation (also called Sylvester equation):
%   (A and B must be square and C must have the rows of A and columns of B)
%
%           A*X + X*B + C = 0
%
%   Q = LIAP(A,B) solves the Right Liapunov matrix equation:
%       (A and B have the same number of rows.)
%
%           A*Q + Q*A' + B*B' = 0
%
%   P = LIAP(A',C) solves the Left Liapunov matrix equation:
%       (A and C have the same number of columns.)
%
%           A'*P + P*A + C'*C = 0matrix  C`.

---

## Temporal Envelope

The envelope function shapes the intensity over time to match real earthquake characteristics:

```
env(t) = ((t-t₀)/(aa·ta))^aa · exp(aa - (t-t₀)/ta) · taper(t)
```

**Three phases:**

1. **Buildup phase** (controlled by `aa`):
   
   - Small `aa` (1-2): Rapid onset (near-field pulses)
   - Large `aa` (4-5): Gradual buildup (far-field motion)

2. **Peak intensity** (at `t ≈ t₀ + aa·ta`):
   
   - Maximum amplitude occurs naturally from envelope shape

3. **Decay phase** (controlled by `ta`):
   
   - Exponential decay with time constant `ta`
   - Typical values: `ta = 1-3 seconds`

**Cosine tapers** at start and end prevent discontinuities.

### Envelope Parameters

The total **significant duration** is approximately:

```
T ≈ t₀ + 5.74 · aa^0.42 · ta
```

For `aa=4, ta=2`: `T ≈ 1 + 5.74·4^0.42·2 ≈ 20 seconds`

---

## Peak-to-RMS Ratio Model

The **peak-to-RMS ratio** (`P2R`) is predicted empirically from ground motion parameters:

```python
P2R = f(T_pk·f_g, aa, log₁₀(ζ_g))
```

where `T_pk = aa·ta` is the time to peak envelope.

This is a **regression model** fit to actual earthquake data, accounting for:

- Frequency content (`f_g`)
- Envelope shape (`aa`, `ta`)
- Damping (`ζ_g`)

The model includes interaction terms and is accurate for typical parameter ranges.

### Scaling Logic

Given target `PGA`:

```
RMS_a = PGA / P2R
a_bar = RMS_a / √(2π ζ_g f_g)
```

The last equation comes from the **steady-state variance** of the ground motion model, computed via the **Lyapunov equation**:

```
A·Σ + Σ·Aᵀ + B·Bᵀ = 0
```

where `Σ` is the state covariance matrix.

---

## Signal Processing Chain

### 1. White Noise Generation

```python
w(t) = randn() / √dt
```

- Unit Gaussian white noise
- Power spectral density: `PSD = 1.0`
- Variance: `σ² = 1/dt`

### 2. Band-Limiting (0-25 Hz)

A **9th-order Butterworth low-pass filter** at 25 Hz removes unrealistic high-frequency content:

```python
A_filt, B_filt, C_filt, D_filt = butter_synth_ss(N=9, fc=25, fs=1/dt)
```

**Forward-backward filtering** ensures:

- Zero phase distortion
- Effective filter order: 18
- Sharp cutoff at Nyquist/2

### 3. Envelope Application

```python
w_env(t) = env(t) · w_filt(t)
```

### 4. Ground Motion Model

```python
accel = lsym(Ag, Bg, Cg, 0, w_env, t, method='foh')
```

Uses **first-order hold** (FOH) for accurate time-stepping.

### 5. Baseline Correction

```python
accel_corrected = accel2displ(accel, t, method='SRA')
```

**Subtract Running Average (SRA)** removes:

- DC bias
- Low-frequency drift

This mimics real strong-motion processing where:

- Sensor drift must be removed
- Final displacement should return to zero

### 6. Integration to Velocity and Displacement

```python
veloc = cumulative_trapezoid(accel_corrected) * dt
displ = cumulative_trapezoid(veloc) * dt
```

---

## Calibrated Parameter Sets

Based on **ATC-63** ground motion characterization:

### Far-Field (FF)

```python
PGA = 3.5 m/s²
PGV = 0.33 m/s
fg = 1.5 Hz
zg = 0.9
aa = 4.0
ta = 2.0 s
P2R ≈ 2.7
```

**Characteristics:**

- Gradual onset
- Moderate frequency content
- Typical of distant earthquakes

### Near-Field, No Pulse (NFNP)

```python
PGA = 5.0 m/s²
PGV = 0.52 m/s
fg = 1.3 Hz
zg = 1.1
aa = 3.0
ta = 2.0 s
P2R ≈ 2.7
```

**Characteristics:**

- Stronger shaking
- Slightly lower frequency
- Higher damping (more inelastic ground response)

### Near-Field, Pulse (NFP)

```python
PGA = 4.7 m/s²
PGV = 0.80 m/s
fg = 0.5 Hz
zg = 1.8
aa = 1.0
ta = 2.0 s
P2R ≈ 2.5
```

**Characteristics:**

- **Very low frequency** (forward directivity pulse)
- **Rapid onset** (small `aa`)
- **High velocity** (V/A ratio ≈ 0.17)
- Heavy damping (strong nonlinear effects)
- Characteristic of strike-slip faults

---

## Mathematical Details

### Lyapunov Equation for Covariance

For the stochastic system:

```
dx = A·x dt + B·dW
```

where `dW` is Wiener process (continuous-time white noise), the **steady-state covariance** satisfies:

```
A·Σ + Σ·Aᵀ + B·Bᵀ = 0
```

For our ground motion model:

```
Σ = liap(A, B)  # Solves the Lyapunov equation
RMS_accel = √(C·Σ·Cᵀ)
```

This ensures the **statistical properties** match the target RMS acceleration.

### Discrete-Time Conversion

The continuous-time system is converted to discrete-time using the **matrix exponential**:

```python
M = [A, B]
    [0, 0]

exp(M·dt) = [A_d, B_d]
            [0,   I  ]
```

For **first-order hold** (FOH):

```python
M = [A, B, 0]
    [0, 0, I]
    [0, 0, 0]
```

This provides more accurate representation of smooth inputs compared to zero-order hold.

---

## Usage Examples

### Basic Usage

```python
from eqgm_1d import eqgm_1d
import numpy as np

# Far-field ground motion
accel, veloc, displ, scale, Ag, Bg, Cg = eqgm_1d(
    PGA=3.5,    # Peak ground acceleration (m/s²)
    fg=1.5,     # Ground frequency (Hz)
    zg=0.9,     # Damping ratio
    aa=4.0,     # Envelope rise parameter
    ta=2.0,     # Envelope decay time (s)
    fig_no=1,   # Plot results
    seed=42     # Reproducible results
)

print(f"PGA: {np.max(np.abs(accel)):.3f} m/s²")
print(f"PGV: {np.max(np.abs(veloc)):.3f} m/s")
print(f"PGD: {np.max(np.abs(displ)):.3f} m")
```

### Custom Time Vector

```python
# 60-second record with 0.005 s time step (200 Hz sampling)
t = np.arange(1, 12001) * 0.005

accel, veloc, displ, _, _, _, _ = eqgm_1d(
    PGA=5.0, fg=1.3, zg=1.1, aa=3.0, ta=2.0,
    t=t, fig_no=2
)
```

### Near-Field Pulse

```python
# Low-frequency pulse-like motion
accel, veloc, displ, _, _, _, _ = eqgm_1d(
    PGA=4.7, fg=0.5, zg=1.8, aa=1.0, ta=2.0,
    fig_no=3, seed=123
)
```

### Ensemble of Ground Motions

```python
# Generate 10 realizations with different random seeds
n_realizations = 10
PGA_values = np.zeros(n_realizations)

for i in range(n_realizations):
    accel, _, _, _, _, _, _ = eqgm_1d(
        PGA=3.5, fg=1.5, zg=0.9, aa=4.0, ta=2.0,
        seed=i, fig_no=0  # No plots
    )
    PGA_values[i] = np.max(np.abs(accel))

print(f"Mean PGA: {np.mean(PGA_values):.3f} m/s²")
print(f"Std PGA: {np.std(PGA_values):.3f} m/s²")
```

---

## Output Files and Plots

### Figure 1: Envelope Visualization

Shows:

- Acceleration time history
- RMS envelope (red)
- Target PGA (green dashed)
- Peak location (green star)

### Figure 2: Three-Panel Display

1. **Acceleration** with envelope bounds
2. **Velocity** time history
3. **Displacement** time history

Standard format for earthquake engineering applications.

---

## Interpretation Guidelines

### Frequency Content

The ground frequency `fg` represents the **dominant frequency** of the motion:

- **Low `fg` (0.3-0.8 Hz)**: Soft soil sites, basins
- **Medium `fg` (1.0-2.0 Hz)**: Typical rock sites
- **High `fg` (2.5-5.0 Hz)**: Hard rock, shallow earthquakes

### Damping Ratio

The damping `zg` represents **effective damping** including:

- Material damping of soil/rock
- Radiation damping
- Nonlinear effects (captured phenomenologically)

Values: `zg = 0.5-2.0` (higher for softer soils and stronger shaking)

### Velocity-to-Acceleration Ratio

The ratio `PGV/PGA` indicates:

- **Low ratio (0.05-0.15 s)**: Short-period content, hard sites
- **Medium ratio (0.15-0.30 s)**: Mixed frequencies
- **High ratio (0.30-0.60 s)**: Long-period pulses, soft sites

### Significant Duration

Approximately `T ≈ 5.74 · aa^0.42 · ta`:

- Larger `aa` → longer duration
- Larger `ta` → longer duration
- Typical: 10-30 seconds for M 6-7 earthquakes

---

## Validation

The synthetic ground motions match statistical properties of real earthquakes:

### Spectral Characteristics

- **Frequency content** determined by `fg` and `zg`
- **Bandwidth** shaped by filter and damping
- **Peak spectral acceleration** approximately at `fg`

### Temporal Properties

- **Buildup phase** matches P-wave arrivals
- **Peak intensity** represents S-wave arrivals
- **Decay** represents coda waves and site effects

### Intensity Measures

- **PGA** calibrated exactly (by design)
- **PGV** approximately matches typical V/A ratios
- **Arias intensity** scales with `aa` and `ta`
- **Significant duration** matches empirical models

---

## Dependencies

```python
numpy                # Array operations
scipy                # Integration, Lyapunov solver
matplotlib           # Plotting

# Local modules:
lsym                 # Continuous-time simulation
dlsym                # Discrete-time simulation  
butter_synth_ss      # Butterworth filter synthesis
accel2displ          # Baseline correction
liap                 # Lyapunov equation solver
```

---

## Applications

### Structural Dynamics

Use synthetic ground motions for:

- **Time-history analysis** of structures
- **Nonlinear response** studies
- **Monte Carlo** simulations
- **Fragility analysis**

### Seismic Hazard

Generate suites of motions with:

- Consistent spectral shape
- Varying intensity
- Statistical scatter (different seeds)

### Educational

Demonstrates:

- **Stochastic process** generation
- **Linear systems** theory
- **Digital filtering**
- **Signal processing**

---

## References

1. **ATC-63** (2008), "Quantification of Building Seismic Performance Factors",
   Applied Technology Council

2. **Kanai-Tajimi Model**, basis for ground motion model:
   
   - Kanai, K. (1957), "Semi-empirical formula for seismic characteristics"
   - Tajimi, H. (1960), "A statistical method for determining the maximum response"

3. **Envelope Functions**:
   
   - Jennings et al. (1969), "Simulated earthquake motions for design purposes"
   - Saragoni & Hart (1974), "Simulation of artificial earthquakes"

4. **Baseline Correction**:
   
   - Boore (2001), "Effect of baseline corrections on displacements and response spectra for several recordings of the 1999 Chi-Chi, Taiwan, earthquake"

---

## Future Enhancements

Potential improvements:

- **Multi-axis** ground motion (3-component)
- **Coherent** ground motion (spatially varying)
- **Frequency-dependent** envelope
- **Soil amplification** models
- **Response spectra** targeting
- **Duration** targeting (Arias intensity)

---
