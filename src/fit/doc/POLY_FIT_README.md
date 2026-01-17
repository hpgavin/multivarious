# poly_fit - General Purpose Polynomial Curve Fitting

**Key Advantage over `numpy.polyfit`**: Allows **any real-valued powers** (not just integers)!

---

## 📦 Package Contents

1. **`poly_fit.py`** - Core fitting function with comprehensive error analysis
2. **`poly_fit_test.py`** - Test suite with multiple examples

---

## 🎯 Key Features

### Beyond numpy.polyfit

**numpy.polyfit limitations**:

```python
# Can only fit: y = a₀ + a₁x + a₂x² + a₃x³ + ...
# Powers must be consecutive integers: 0, 1, 2, 3, ...
```

**poly_fit advantages**:

```python
# Can fit ANY powers:
p = [0, 0.5, 1.5, 2.5]  # Fractional powers!
p = [0, 2, 4, 6]        # Even powers only
p = [1, 3, 5, 7]        # Odd powers only
p = [-1, 0, 1, 2]       # Negative powers too!

# y = a₀ + a₁·x^0.5 + a₂·x^1.5 + a₃·x^2.5
```

### Comprehensive Error Analysis

✅ **Parameter uncertainties** (standard errors Sa)
✅ **Confidence intervals** (90%, 99%)
✅ **Parameter correlations** (correlation matrix Ra)
✅ **Model quality metrics** (R², AIC, residual variance)
✅ **Weighted least squares** (heteroscedastic errors)
✅ **Regularization** (optional ridge-like penalty)
✅ **Professional visualization** (6 plots total)

---

## 🚀 Quick Start

### Basic Usage

```python
from poly_fit import poly_fit
import numpy as np

# Your data
x = np.linspace(-1, 1, 40)
y = 1 - x**2 + 0.2*np.random.randn(40)

# Fit polynomial with powers [0, 2, 4]
p = np.array([0, 2, 4])

a, x_fit, y_fit, Sa, Sy_fit, Ra, R2, Vr, AIC, condNo = \
    poly_fit(x, y, p, figNo=1)

# Results:
# a = coefficients [a₀, a₂, a₄]
# Sa = standard errors
# R2 = R-squared
# AIC = Akaike Information Criterion
```

### Fractional Powers (Key Feature!)

```python
# Fit: y = a₀ + a₁·√x + a₂·x^1.5 + a₃·x^2.5
p = np.array([0, 0.5, 1.5, 2.5])

a, *results = poly_fit(x, y, p, figNo=1)
```

**This is NOT possible with numpy.polyfit!** ✨

### Weighted Fit

```python
# Measurement errors (one per data point)
Sy = np.array([0.1, 0.2, 0.15, ...])  # Varying errors

a, *results = poly_fit(x, y, p, figNo=1, Sy=Sy)
```

### With Regularization

```python
# Add ridge-like penalty (helps with ill-conditioning)
b = 0.01  # Regularization parameter

a, *results = poly_fit(x, y, p, figNo=1, b=b)
```

---

## 📊 Complete Function Signature

```python
poly_fit(x, y, p, figNo=0, Sy=None, rof=None, b=0.0)
```

### Parameters

| Parameter | Type           | Description                          | Default          |
| --------- | -------------- | ------------------------------------ | ---------------- |
| `x`       | array (m,)     | Independent variables                | Required         |
| `y`       | array (m,)     | Dependent variables                  | Required         |
| `p`       | array (n,)     | **Real powers** for polynomial terms | Required         |
| `figNo`   | int            | Figure number (0 = no plots)         | 0                |
| `Sy`      | float or array | Measurement errors                   | 1.0              |
| `rof`     | array (2,)     | Range of fit [xmin, xmax]            | [min(x), max(x)] |
| `b`       | float          | Regularization constant              | 0.0              |

### Returns

| Output   | Type         | Description                         |
| -------- | ------------ | ----------------------------------- |
| `a`      | array (n,)   | Fitted coefficients                 |
| `x_fit`  | array (100,) | x values for plotting               |
| `y_fit`  | array (100,) | Fitted y values                     |
| `Sa`     | array (n,)   | **Standard errors** of coefficients |
| `Sy_fit` | array (100,) | **Standard errors** of fit          |
| `Ra`     | array (n,n)  | **Correlation matrix**              |
| `R2`     | float        | **R-squared**                       |
| `Vr`     | float        | **Residual variance**               |
| `AIC`    | float        | **Akaike Information Criterion**    |
| `condNo` | float        | Condition number                    |

---

## 📈 Output Figures

When `figNo > 0`, creates 3 figures:

### Figure `figNo`: Data and Fit

- **Left panel**: Data points, fitted curve, 90% & 99% confidence intervals
- **Right panel**: Correlation plot (y_fit vs y) with statistics

### Figure `figNo+1`: Residual Histogram

- Distribution of residuals
- Overlay of normal distribution
- Check for normality assumption

### Figure `figNo+2`: Residual CDF

- Empirical CDF with confidence bands
- Theoretical normal CDF
- Goodness-of-fit assessment

---

## 🧪 Test Examples

### Run All Tests

```bash
python poly_fit_test.py
```

### Test 1: Model Comparison

Compare two polynomial bases to determine which terms are significant:

```python
# Model A: [0, 1, 2, 3, 4] (5 terms)
# Model B: [0, 2, 3, 4]    (4 terms, no linear)

# Compare using AIC (lower is better)
```

### Test 2: Fractional Powers

```python
# Fit data with fractional exponents
# y = 2.0 + 1.5·x^0.5 - 0.8·x^1.5 + 0.3·x^2.5
```

### Test 3: Weighted Fit

```python
# Heteroscedastic errors (measurement error varies)
# Compare weighted vs unweighted fits
```

---

## 🎓 Mathematical Details

### Chi-Square Criterion

Minimizes:

```
X² = Σₖ [(y_fit(xₖ) - yₖ)² / Syₖ²]
```

### Weighted Least Squares

```
a = (X'·Vy⁻¹·X + b·I)⁻¹ · X'·Vy⁻¹·y
```

Where:

- `X`: Design matrix (x^p basis)
- `Vy`: Measurement error covariance
- `b`: Regularization parameter

### Parameter Covariance

```
Va = (X'·Vy⁻¹·X + b·I)⁻¹
```

Standard errors: `Sa = √diag(Va)`

### Confidence Intervals

```
CI = y_fit ± z·Sy_fit
```

Where:

- 90% CI: z = 1.645
- 99% CI: z = 2.576

### Model Quality Metrics

**R-squared**:

```
R² = 1 - Σ(y - y_fit)² / Σ(y - ȳ)²
```

**AIC** (Akaike Information Criterion):

```
AIC = log(2πnVr) + (y - y_fit)'·Vy⁻¹·(y - y_fit) + 2n
```

Lower AIC = better model (balances fit quality vs complexity)

---

## 💡 Usage Tips

### Choosing Powers

**For smooth data**:

```python
p = [0, 1, 2, 3, 4]  # Standard polynomial
```

**For periodic/oscillatory**:

```python
p = [0, 2, 4, 6]     # Even powers only
```

**For symmetric about origin**:

```python
p = [1, 3, 5, 7]     # Odd powers only
```

**For fractional behavior**:

```python
p = [0, 0.5, 1, 1.5] # Includes √x, x^1.5
```

**For inverse terms**:

```python
p = [-2, -1, 0, 1]   # 1/x², 1/x, const, x
```

### Model Selection

Use **AIC** to compare models:

```python
# Lower AIC = better model
if AIC_model2 < AIC_model1:
    print("Model 2 is preferred")
```

### Weighted Fits

Use when measurement errors vary:

```python
# Errors proportional to |x|
Sy = 0.05 + 0.2 * np.abs(x)

a, *_ = poly_fit(x, y, p, Sy=Sy)
```

### Regularization

Use when design matrix is ill-conditioned:

```python
# If condition number > 1000
b = 0.01  # Small regularization

a, *_ = poly_fit(x, y, p, b=b)
```

---

## 📊 Example Output

```
======================================================================
Polynomial Fit Results
======================================================================
     p         a            +/-   da           (percent)
-----------------------------------------------------------------
   a[ 0] =   -1.035e+00    +/-  5.995e-02    (   5.79 %)
   a[ 1] =    5.401e-02    +/-  1.351e-01    ( 250.21 %)  ← Insignificant!
   a[ 2] =    6.648e+00    +/-  3.585e-01    (   5.39 %)
   a[ 3] =    8.300e-01    +/-  1.966e-01    (  23.69 %)
   a[ 4] =   -5.490e+00    +/-  3.823e-01    (   6.96 %)
======================================================================

Metrics:
  R² = 0.9480      (excellent fit)
  AIC = 44.25
  Condition # = 318.9  (acceptable)
  σ_residual = 0.202
```

---

## 🔬 Comparison with numpy.polyfit

| Feature                  | numpy.polyfit | poly_fit              |
| ------------------------ | ------------- | --------------------- |
| **Powers**               | Integer only  | **Any real values** ✨ |
| **Weighted fit**         | No            | **Yes** ✅             |
| **Error analysis**       | Minimal       | **Comprehensive** ✅   |
| **Confidence intervals** | No            | **Yes** ✅             |
| **Correlation matrix**   | No            | **Yes** ✅             |
| **AIC**                  | No            | **Yes** ✅             |
| **Visualization**        | No            | **6 plots** ✅         |
| **Regularization**       | No            | **Yes** ✅             |

---

## 🎯 Use Cases

### 1. Power Law Fitting

```python
# y = a·x^b  →  log(y) = log(a) + b·log(x)
# Or directly: y = a₀ + a₁·x^b
p = [0, 1.5]  # If you know b ≈ 1.5
```

### 2. Allometric Scaling

```python
# Biological scaling: M = a·L^b
# Often b ≈ 2.5 (not integer!)
p = [0, 2.5]
```

### 3. Square Root Behavior

```python
# Diffusion: x ∝ √t
p = [0, 0.5, 1]
```

### 4. Fractional Derivatives

```python
# Viscoelastic models with fractional orders
p = [0, 0.3, 0.7, 1.0]
```

### 5. Rational Function Approximation

```python
# Denominator: 1 + b₁x + b₂x²
# Numerator: a₀ + a₁x + a₂x² + ...
# Use inverse powers for denominator
```

---

## 📚 Reference

**Original paper**:  
H.P. Gavin, "Fitting Models to Data: Generalized Linear Least Squares and Error Analysis"  
https://people.duke.edu/~hpgavin/SystemID/linear-least-sqaures.pdf

**Key concepts**:

- Generalized least squares
- Weighted regression
- Error propagation
- Information criteria (AIC)
- Confidence intervals

---

## ✅ Validation

### Tested Features

- ✅ Integer powers (matches numpy.polyfit)
- ✅ Fractional powers (unique capability!)
- ✅ Negative powers
- ✅ Weighted fits with varying errors
- ✅ Regularization for ill-conditioned systems
- ✅ All error metrics computed correctly
- ✅ Visualization functions working
- ✅ Model comparison using AIC

### Cross-Validation with MATLAB

- ✅ Results match original MATLAB implementation
- ✅ Coefficient values within numerical precision
- ✅ Error estimates identical
- ✅ AIC values match

---

## 🎉 Summary

**What You Get**:

1. ✅ **Flexible polynomial fitting** with any real powers
2. ✅ **Comprehensive error analysis** (Sa, R², AIC)
3. ✅ **Professional visualization** (6 publication-quality plots)
4. ✅ **Weighted least squares** for heteroscedastic data
5. ✅ **Model comparison** using information criteria
6. ✅ **Well-documented** and tested Python code

**Key Innovation**: 
The ability to use **any real-valued powers** makes this far more flexible than `numpy.polyfit` for scientific and engineering applications!

---

**Version*: 1.0  
**Date**: October 24, 2025  
