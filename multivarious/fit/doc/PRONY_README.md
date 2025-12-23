# Prony Series Fitting with L1 Regularization

## Overview

This package implements Prony series fitting for frequency-domain viscoelastic material characterization using **L1 regularization** and an **active set method** for constrained optimization.

The Prony series is a widely used model in viscoelasticity to represent the time- or frequency-dependent behavior of materials like polymers, biological tissues, and composite materials.

---

## The Prony Series Model

### Time Domain

In the time domain, the relaxation modulus of a viscoelastic material is expressed as:

```
G(t) = k₀ + Σ kₖ exp(-t/τₖ)
       k=1 to N
```

where:

- `k₀` = equilibrium (long-time) modulus
- `kₖ` = Prony series coefficients (stiffness contributions)
- `τₖ` = relaxation times
- `N` = number of terms in the series

### Frequency Domain

In the frequency domain (which is what we work with here), the complex modulus is:

```
G(ω) = k₀ + Σ (iω τₖ kₖ) / (iω τₖ + 1)
       k=1 to N
```

where:

- `ω = 2πf` is the angular frequency (rad/s)
- `i = √(-1)` is the imaginary unit
- `G(ω) = G'(ω) + iG"(ω)` is the complex modulus
  - `G'(ω)` = storage modulus (elastic component)
  - `G"(ω)` = loss modulus (viscous component)

### Physical Interpretation

Each term in the Prony series represents a **Maxwell element** (a spring and dashpot in series):

- The **storage modulus** `G'(ω)` represents the elastic energy stored and recovered
- The **loss modulus** `G"(ω)` represents the energy dissipated as heat
- The **loss tangent** `tan(δ) = G"(ω)/G'(ω)` is a measure of damping

---

## The Identification Problem

Given experimental measurements of the complex modulus `G_dat(ωᵢ)` at frequencies `ωᵢ`, we want to identify:

1. The equilibrium modulus `k₀`
2. The Prony coefficients `kₖ`

We **specify** the relaxation times `τₖ` a priori (typically logarithmically spaced).

### Why This is Challenging

1. **Overcomplete basis**: We often use many more relaxation times than are truly needed (e.g., 50-100 terms) to ensure we capture the behavior across all frequencies.

2. **Ill-conditioning**: The problem is ill-posed because:
   
   - Many different combinations of `kₖ` can produce similar `G(ω)`
   - The basis functions overlap significantly in frequency space
   - Small changes in data can lead to large changes in parameters

3. **Physical constraints**: The coefficients `kₖ` must be non-negative for physical realizability (materials can't have negative stiffness).

---

## L1 Regularization: The Solution

### Standard Least Squares Problem

Without regularization, we would solve:

```
minimize: ||G_dat - G_model(k)||²
```

This often produces:

- Many small non-zero coefficients (overfitting)
- Negative coefficients (unphysical)
- High sensitivity to noise

### L1 Regularized Problem

We instead solve:

```
minimize: ||G_dat - G_model(k)||² + α||k||₁
subject to: kₖ ≥ 0 for all k
```

where:

- `||k||₁ = Σ|kₖ|` is the L1 norm (sum of absolute values)
- `α > 0` is the regularization parameter

### Why L1 Regularization Works

The L1 penalty has a special property called **sparsity promotion**:

1. **Sparsity**: L1 regularization drives many coefficients exactly to zero, effectively performing automatic model selection. This identifies which relaxation times are truly important.

2. **Stability**: The regularization term prevents overfitting and reduces sensitivity to noise.

3. **Interpretability**: A sparse solution with few non-zero terms is easier to interpret physically.

**Mathematical Insight**: The L1 norm has "corners" at the axes where many components can be exactly zero, unlike the L2 norm (ridge regression) which only shrinks coefficients toward zero.

### Choosing α

The regularization parameter `α` controls the trade-off:

- **Small α**: More terms retained, better fit to data, risk of overfitting
- **Large α**: Fewer terms (sparser), smoother solution, risk of underfitting

Typical values: `α ∈ [0.1, 10]`

---

## The Optimization Method: Active Set Algorithm

### Mathematical Formulation

The optimization problem can be written as:

```
minimize: f(k) = ½k^T(T^H T)k - k^T(T^H G_dat) + α·1^T k
subject to: kₖ ≥ 0
```

where:

- `T` is the design matrix (basis functions)
- `T^H` denotes the Hermitian (conjugate) transpose
- `1` is a vector of ones

### KKT Conditions

At the optimum, the **Karush-Kuhn-Tucker (KKT)** conditions must hold:

```
∇f(k) + Σ λₖ ∇gₖ(k) = 0     (stationarity)
kₖ ≥ 0                       (primal feasibility)
λₖ ≥ 0                       (dual feasibility)
λₖ · kₖ = 0                  (complementarity)
```

where:

- `λₖ` are the Lagrange multipliers
- `gₖ(k) = -kₖ` are the inequality constraints

### Active Set Method

The algorithm maintains and updates the **active set** `A` = {indices where `kₖ = 0`}:

**Algorithm Steps:**

1. **Identify active constraints**: Find indices where `kₖ ≈ 0`

2. **Form KKT system**: For step `h` and multipliers `λ`:
   
   ```
   [ T^H T    I_A^T ] [ h ]   = [ -(T^H T)k + T^H G_dat - α ]
   [ I_A        0    ] [ λ ]     [          -k_A              ]
   ```
   
   where `I_A` selects rows corresponding to active constraints

3. **Solve for step direction** `h` and multipliers `λ`

4. **Line search**: Find maximum step size `dh` such that `k + dh·h ≥ 0`

5. **Update**: `k ← k + dh·h`

6. **Check convergence**: Stop if `||h|| < ε||k||` and `min(k) ≥ 0`

### Why This Works

- The active set method efficiently handles the inequality constraints
- By working with equality constraints on the active set, we can solve a linear system
- The method naturally enforces non-negativity without requiring projection or barrier functions
- Convergence is typically achieved in 10-20 iterations

---

## Using the Code

### Basic Usage

```python
from prony_fit import prony_fit
import numpy as np

# Your experimental data
G_dat = ...  # Complex modulus (M points)
f_dat = ...  # Frequencies in Hz (M points)

# Specify relaxation times (logarithmically spaced)
tau = np.logspace(-4, 2, 50)  # 50 relaxation times

# L1 regularization parameter
alpha = 0.5

# Perform the fit
ko, k, cvg_hst = prony_fit(G_dat, f_dat, tau, alpha)

# Results:
# ko - equilibrium modulus
# k  - Prony coefficients (length 50, but many will be zero)
# cvg_hst - convergence history
```

### Running the Test Example

```bash
python prony_fit_example.py
```

This will:

1. Generate synthetic noisy data with 6 true relaxation times
2. Fit using 97 candidate relaxation times
3. Show that L1 regularization correctly identifies only the relevant times
4. Display storage modulus, loss modulus, tan(δ), and convergence

---

## Interpreting the Results

### Figures Generated

**Figure 101**: Basis Functions

- Shows the real and imaginary parts of each basis function
- Each function is associated with one relaxation time
- Functions are smoother at low frequencies, sharper at high frequencies

**Figure 102**: Data vs Fit (updates during iteration)

- Top: Storage modulus G'(ω) - elastic response
- Bottom: Loss modulus G"(ω) - viscous response
- Shows how well the model captures the data

**Figure 1** (in test): Complete Comparison

- Storage modulus, loss modulus, and tan(δ)
- Demonstrates quality of fit across all frequencies

**Figure 2** (in test): Relaxation Spectrum

- Shows identified coefficients vs relaxation time
- Sparse solution: most coefficients are zero
- Non-zero coefficients indicate dominant relaxation processes

**Figure 3** (in test): Convergence History

- Parameters converge smoothly to final values
- Lagrange multipliers show which constraints are active
- Typically converges in 10-20 iterations

### Physical Interpretation

A sparse relaxation spectrum tells you:

- **Which time scales** are important for the material's response
- **How much stiffness** is contributed at each time scale
- **Transition frequencies** where behavior changes from elastic to viscous

For example:

- A single peak → material has one dominant relaxation process
- Multiple peaks → material has several distinct relaxation mechanisms
- Broad distribution → material has a wide range of relaxation times

---

## Mathematical Details

### Design Matrix Construction

The design matrix `T` is constructed as:

```python
T = [1, iω₁τ₁/(iωτ₁+1), ..., iω₁τₙ/(iω₁τₙ+1)]
    [1, iω₂τ₁/(iωτ₁+1), ..., iω₂τₙ/(iω₂τₙ+1)]
    [⋮        ⋮                    ⋮        ]
    [1, iωₘτ₁/(iωτ₁+1), ..., iωₘτₙ/(iωₘτₙ+1)]
```

Each column is a basis function evaluated at all measurement frequencies.

### Normal Equations

Since we're working with complex data, the normal equations become:

```
(T^H T) k = T^H G_dat
```

We take the real part (which is equivalent to minimizing over both real and imaginary parts):

```
2·Re(T^H T) k = 2·Re(T^H G_dat)
```

### Regularization Effect

Without regularization:

- Solution: `k = (T^H T)^(-1) T^H G_dat`
- Problem: `T^H T` may be nearly singular → unstable solution

With L1 regularization:

- The gradient includes the penalty: `∇f = (T^H T)k - T^H G_dat + α·sign(k)`
- This "pushes" small coefficients to exactly zero
- The active set method efficiently handles the resulting non-smooth optimization

---

## Advanced Topics

### Selecting the Regularization Parameter

**Cross-validation approach:**

1. Split data into training and validation sets
2. Try multiple values of α
3. Choose α that minimizes validation error

**L-curve method:**

1. Plot ||G_dat - G_model|| vs ||k||₁ for various α
2. The "corner" of the L-curve often indicates good regularization

**Physical insight:**

- Start with α ≈ 0.1 to 1.0
- Increase α if you get too many non-zero terms
- Decrease α if the fit quality is poor

### Choosing Relaxation Times

**Logarithmic spacing** is standard:

```python
tau = np.logspace(log10(tau_min), log10(tau_max), N)
```

Guidelines:

- `tau_min ≈ 1/(10·ω_max)` - at least 10× faster than highest frequency
- `tau_max ≈ 10/ω_min` - at least 10× slower than lowest frequency
- `N = 30-100` - enough to capture all relevant time scales

### Alternative: Time Domain Fitting

The same L1 regularization approach works in the time domain with stress relaxation data:

```
G(t) = k₀ + Σ kₖ exp(-t/τₖ)
```

The basis functions become:

```python
T = [1, exp(-t₁/τ₁), ..., exp(-t₁/τₙ)]
    [1, exp(-t₂/τ₁), ..., exp(-t₂/τₙ)]
    [⋮       ⋮                ⋮      ]
```

---

## References

1. **Active Set Methods**:
   
   - Nocedal & Wright, "Numerical Optimization", 2nd ed., Springer, 2006

2. **L1 Regularization**:
   
   - Tibshirani, R., "Regression Shrinkage and Selection via the Lasso", JRSS-B, 1996
   - Hastie, Tibshirani & Friedman, "The Elements of Statistical Learning", 2009

3. **Prony Series in Viscoelasticity**:
   
   - Park & Schapery, "Methods of Interconversion Between Linear Viscoelastic Material Functions", Int. J. Solids Structures, 1999
   - Baumgaertel & Winter, "Determination of discrete relaxation and retardation time spectra from dynamic mechanical data", Rheologica Acta, 1989

4. **Applications**:
   
   - Lakes, R.S., "Viscoelastic Materials", Cambridge University Press, 2009
   - Christensen, R.M., "Theory of Viscoelasticity", 2nd ed., Dover, 2003

---

## Application Notes

This code is particularly well-suited for:

1. **Regularization concepts**: Shows why and how regularization prevents overfitting

2. **Constrained optimization**: Demonstrates KKT conditions and active set methods

3. **Ill-posed problems**: Illustrates how to handle underdetermined or ill-conditioned systems

4. **Physical constraints**: Shows how to incorporate domain knowledge (non-negativity)

5. **Complex optimization**: Demonstrates optimization with complex-valued objectives

6. **Sparsity**: Visualizes how L1 regularization promotes sparse solutions

---

# 
