# butter_synth_ss.py - Quick Reference

# 

---

## 🎯 Quick Start

```python
from butter_synth_ss import butter_synth_ss
import numpy as np

# Continuous-time low-pass filter
A, B, C, D, poles = butter_synth_ss(N=4, fc=10)

# Discrete-time low-pass filter
A, B, C, D, poles = butter_synth_ss(N=4, fc=10, fs=100)

# High-pass filter
A, B, C, D, poles = butter_synth_ss(N=4, fc=10, filter_type='high')
```

---

## 📐 Function Signature

```python
def butter_synth_ss(N, fc, fs=None, filter_type='low'):
    """
    Parameters
    ----------
    N : int
        Filter order (number of poles)
    fc : float
        -3 dB cutoff frequency (Hz)
    fs : float, optional
        Sampling frequency (Hz)
        - None: Continuous-time output
        - float: Discrete-time via FOH
    filter_type : str, optional
        'low' or 'high' (default: 'low')

    Returns
    -------
    A : ndarray (N, N)
        State matrix (companion form)
    B : ndarray (N, 1)
        Input matrix
    C : ndarray (1, N)
        Output matrix
    D : ndarray (1, 1)
        Feedthrough matrix
    p : ndarray (N,)
        Continuous-time poles (s-plane)
    """
```

---

## ⭐ Key Features

### **Exact Pole Placement**

- Butterworth poles computed analytically
- N poles equally spaced on semicircle in s-plane
- Radius = 2π·fc (cutoff frequency in rad/s)

### **Matrix Exponential Discretization**

- Uses `con2dis()` with first-order hold (FOH)
- More accurate than bilinear or impulse invariance
- Exact discretization via matrix exponential
- Preserves frequency response better

### **Companion Form State-Space**

- Minimal realization (N states for Nth-order filter)
- Easy to implement
- Compatible with control system tools

---

## 🔬 Butterworth Filter Properties

### **What Makes It Special**

1. **Maximally Flat** - No ripple in passband
2. **Monotonic** - Response always decreases in stopband
3. **Smooth** - No sharp transitions
4. **Predictable** - Rolloff = -20N dB/decade

### **Frequency Response**

```
|H(jω)|² = 1 / (1 + (ω/ωc)^(2N))

where:
  ωc = 2π·fc  (cutoff frequency in rad/s)
  N = filter order
```

At cutoff: |H(jωc)| = 1/√2 ≈ -3 dB

### **Pole Locations (s-plane)**

For low-pass filter:

```
p_k = ωc · exp(jπ(0.5 + (2k-1)/(2N)))  for k = 1, 2, ..., N
```

All poles:

- In left half-plane (stable)
- On semicircle of radius ωc
- Equally spaced angularly

---

## 📊 State-Space Realization

### **Companion (Controller Canonical) Form**

**Low-Pass Filter:**

```
A = [0      1      0    ...  0  ]
    [0      0      1    ...  0  ]
    [⋮      ⋮      ⋮    ⋱   ⋮  ]
    [0      0      0    ...  1  ]
    [-aₙ   -aₙ₋₁  -aₙ₋₂ ... -a₁]

B = [0]      C = [aₙ  0  ...  0]      D = 0
    [0]
    [⋮]
    [1]
```

**High-Pass Filter:**

```
A = same

B = [0]      C = [-aₙ  -aₙ₋₁  ...  -a₁]      D = 1
    [0]
    [⋮]
    [1]
```

where [1, a₁, a₂, ..., aₙ] are coefficients of:

```
∏(s - p_k) = sᴺ + a₁sᴺ⁻¹ + ... + aₙ
```

---

## 💡 Use Cases

### **1. Digital Filtering**

```python
# Design discrete-time filter
fs = 1000  # Hz
fc = 50    # Hz
N = 6

A, B, C, D, _ = butter_synth_ss(N, fc, fs, 'low')

# Filter signal
x = np.random.randn(10000)  # Input signal
y = np.zeros_like(x)
state = np.zeros(N)

for k in range(len(x)):
    y[k] = C @ state + D * x[k]
    state = A @ state + B.flatten() * x[k]
```

### **2. Control Systems**

```python
# Design anti-aliasing filter for sensor
A, B, C, D, poles = butter_synth_ss(N=4, fc=100, fs=1000)

# Check stability
print(f"Poles: {np.linalg.eigvals(A)}")
print(f"All inside unit circle? {np.all(np.abs(np.linalg.eigvals(A)) < 1)}")
```

### **3. Frequency Response Analysis**

```python
from scipy.signal import dlti, dfreqresp
import matplotlib.pyplot as plt

# Design filter
A, B, C, D, _ = butter_synth_ss(N=8, fc=20, fs=200)

# Compute frequency response
sys = dlti(A, B, C, D, dt=1/200)
w = np.logspace(-1, 2, 1000)
_, h = dfreqresp(sys, w=2*np.pi*w/200)

# Plot
plt.semilogx(w, 20*np.log10(np.abs(h.flatten())))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('8th-Order Butterworth Filter')
plt.grid(True)
plt.show()
```

### **4. Cascaded Filters**

```python
# Design two stages
A1, B1, C1, D1, _ = butter_synth_ss(N=4, fc=10, fs=100)
A2, B2, C2, D2, _ = butter_synth_ss(N=4, fc=50, fs=100)

# Cascade: sys2(sys1(u))
# State-space: [A1  0 ] [B1]
#              [B2C1 A2] [B2D1]
```

---

## 🎓 Technical Details

### **Why State-Space Form?**

Advantages over transfer function (b, a coefficients):

1. **Numerical Stability** - No polynomial rootfinding
2. **Easy Cascading** - Block diagram algebra
3. **Time-Domain Implementation** - Direct state updates
4. **Extensible** - Easy to add observers, controllers
5. **Exact Poles** - Computed analytically, not from polynomial

### **Why Companion Form?**

1. **Minimal** - N states for Nth-order system
2. **Standard** - Widely recognized form
3. **Observable** - Single output observes all states
4. **Controllable** - Single input controls all states

### **Discretization Methods**

| Method       | Accuracy | Use Case                  |
| ------------ | -------- | ------------------------- |
| **FOH**      | High     | Smooth signals (default)  |
| ZOH          | Medium   | Piecewise constant inputs |
| Bilinear     | Good     | Frequency warping OK      |
| Impulse Inv. | Poor     | Not recommended           |

**FOH (First-Order Hold):**

- Input varies linearly between samples
- More accurate than ZOH for smooth signals
- Used by `butter_synth_ss` when `fs` is provided

---

## 🔬 Examples

### **Example 1: Compare Filter Orders**

```python
import matplotlib.pyplot as plt

fc = 10  # Hz
fs = 200  # Hz
orders = [2, 4, 6, 8]

fig, ax = plt.subplots()
for N in orders:
    A, B, C, D, _ = butter_synth_ss(N, fc, fs)
    sys = dlti(A, B, C, D, dt=1/fs)
    w = np.logspace(-1, 2, 1000)
    _, h = dfreqresp(sys, w=2*np.pi*w/fs)
    ax.semilogx(w, 20*np.log10(np.abs(h.flatten())), 
                label=f'N={N}')

ax.axvline(fc, color='k', linestyle='--', alpha=0.5)
ax.axhline(-3, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Magnitude (dB)')
ax.set_title('Butterworth Filters: Effect of Order')
ax.legend()
ax.grid(True)
plt.show()
```

### **Example 2: Low-Pass vs High-Pass**

```python
fc = 20  # Hz
fs = 200  # Hz
N = 6

# Low-pass
A_lp, B_lp, C_lp, D_lp, _ = butter_synth_ss(N, fc, fs, 'low')

# High-pass
A_hp, B_hp, C_hp, D_hp, _ = butter_synth_ss(N, fc, fs, 'high')

# Compare
w = np.logspace(0, 2, 1000)

sys_lp = dlti(A_lp, B_lp, C_lp, D_lp, dt=1/fs)
_, h_lp = dfreqresp(sys_lp, w=2*np.pi*w/fs)

sys_hp = dlti(A_hp, B_hp, C_hp, D_hp, dt=1/fs)
_, h_hp = dfreqresp(sys_hp, w=2*np.pi*w/fs)

fig, ax = plt.subplots()
ax.semilogx(w, 20*np.log10(np.abs(h_lp.flatten())), label='Low-pass')
ax.semilogx(w, 20*np.log10(np.abs(h_hp.flatten())), label='High-pass')
ax.axvline(fc, color='k', linestyle='--', alpha=0.5, label=f'fc={fc} Hz')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Magnitude (dB)')
ax.set_title('Low-Pass vs High-Pass Butterworth')
ax.legend()
ax.grid(True)
plt.show()
```

### **Example 3: Pole-Zero Map**

```python
# Continuous-time
A_cont, B_cont, C_cont, D_cont, poles_cont = butter_synth_ss(N=6, fc=10)

# Discrete-time
A_disc, B_disc, C_disc, D_disc, _ = butter_synth_ss(N=6, fc=10, fs=100)
poles_disc = np.linalg.eigvals(A_disc)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Continuous-time poles (s-plane)
ax1.plot(np.real(poles_cont), np.imag(poles_cont), 'rx', 
         markersize=12, markeredgewidth=2, label='Poles')
theta = np.linspace(0, np.pi, 100)
wc = 2 * np.pi * 10
ax1.plot(wc*np.cos(theta), wc*np.sin(theta), 'k--', 
         alpha=0.3, label=f'|s|={wc:.1f}')
ax1.set_xlabel('Real')
ax1.set_ylabel('Imaginary')
ax1.set_title('Continuous-Time (s-plane)')
ax1.legend()
ax1.grid(True)
ax1.axis('equal')

# Discrete-time poles (z-plane)
ax2.plot(np.real(poles_disc), np.imag(poles_disc), 'bx',
         markersize=12, markeredgewidth=2, label='Poles')
theta = np.linspace(0, 2*np.pi, 100)
ax2.plot(np.cos(theta), np.sin(theta), 'k--', 
         alpha=0.3, label='|z|=1')
ax2.set_xlabel('Real')
ax2.set_ylabel('Imaginary')
ax2.set_title('Discrete-Time (z-plane)')
ax2.legend()
ax2.grid(True)
ax2.axis('equal')

plt.tight_layout()
plt.show()
```

---

## 📊 Comparison with SciPy

### **scipy.signal.butter**

```python
from scipy.signal import butter

# SciPy returns transfer function coefficients
b, a = butter(N=4, Wn=10, btype='low', fs=100)

# Then convert to state-space
from scipy.signal import tf2ss
A, B, C, D = tf2ss(b, a)
```

### **butter_synth_ss**

```python
# Direct state-space synthesis
A, B, C, D, poles = butter_synth_ss(N=4, fc=10, fs=100)

# Advantages:
# ✓ Direct state-space (no tf2ss conversion)
# ✓ Returns continuous-time poles
# ✓ Uses matrix exponential (more accurate)
# ✓ FOH discretization (better for smooth signals)
# ✓ Companion form (minimal, standard)
```

---

## ⚠️ Important Notes

### **1. Nyquist Constraint**

```python
# ❌ WRONG
butter_synth_ss(N=4, fc=60, fs=100)  # fc >= fs/2!

# ✅ CORRECT
butter_synth_ss(N=4, fc=40, fs=100)  # fc < fs/2
```

### **2. Order Selection**

Higher order = steeper rolloff, but:

- More computation
- More states to store
- Potential numerical issues for very high N

**Rule of thumb**: N = 4-8 is typical

### **3. Continuous vs Discrete**

```python
# Continuous-time (for analysis)
A_cont, B_cont, C_cont, D_cont, poles = butter_synth_ss(N, fc)

# Discrete-time (for implementation)
A_disc, B_disc, C_disc, D_disc, poles = butter_synth_ss(N, fc, fs)
```

---

## 🔧 Implementation Tips

### **Real-Time Filtering**

```python
class ButterworthFilter:
    def __init__(self, N, fc, fs, filter_type='low'):
        self.A, self.B, self.C, self.D, _ = \
            butter_synth_ss(N, fc, fs, filter_type)
        self.state = np.zeros(N)

    def filter(self, u):
        """Filter single sample"""
        y = (self.C @ self.state + self.D * u)[0, 0]
        self.state = self.A @ self.state + self.B.flatten() * u
        return y

    def filter_batch(self, u):
        """Filter array of samples"""
        y = np.zeros_like(u)
        for k, uk in enumerate(u):
            y[k] = self.filter(uk)
        return y

# Usage
filt = ButterworthFilter(N=6, fc=10, fs=100)
filtered_signal = filt.filter_batch(noisy_signal)
```

---

## ✅ Validation

The implementation has been validated against:

1. ✓ Analytical Butterworth pole locations
2. ✓ SciPy's `butter()` function
3. ✓ Frequency response at cutoff (-3 dB)
4. ✓ Stability (poles in LHP for s-plane, inside unit circle for z-plane)
5. ✓ Rolloff rate (-20N dB/decade)

---

## 📚 References

1. **DSP Related**: https://www.dsprelated.com/showarticle/1119.php
2. **Original MATLAB**: H.P. Gavin, Duke University, 2021
3. **Oppenheim & Schafer**: "Discrete-Time Signal Processing"
4. **Wikipedia**: Butterworth filter

---

## 📦 Dependencies

```python
import numpy as np
from con2dis import con2dis  # Continuous-to-discrete conversion
```

**Status**: ✅ Tested and validated

---

**Perfect for control systems, digital filtering, and signal processing applications!** 🚀
