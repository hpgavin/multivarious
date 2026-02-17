# The Levenberg-Marquardt algorithm for nonlinear least-squares curve fitting problems

Implementatiom of the Levenberg-Marquardt algorithm for nonlinear curve fitting.  Source code with transparent structure for pedagogically-oriented applications.  This implementation offers resources to investigate and interpret how algorithmic options affect model statistics:

1. **Confidence Intervals** - Chi-squared based uncertainty quantification for both coefficients and fitted curve
2. **Iteration Visualization** - Watch the fit converge in real-time (set `print_level=3`)
3. **Update Strategies Options** - Compare Levenberg-Marquardt, Quadratic, and Nielsen λ adaptation
4. **Broyden Jacobian Updates** - Efficient rank-1 approximation when appropriate
5. **Hybrid Jacobian Strategy** - Automatic switching between finite differences and Broyden updating based on convergence history
6. **Explicit Algorithm Structure** - See exactly how damping parameter λ adapts

### **Pedagogical Features**

- **Transparent Code**: To reveal every algorithmic stepx without hidden abstractions.
- **Convergence Diagnostics**: To show λ adaptation in action
- **Examples**: To demonstrate algorithm behavior
- **Sensitivity Analysis**: To explore initial guess dependence
- **Documented**: To explain the purpose of the methods

### **Production Quality**

- **Fit in multi-dimensional domains**: fit ($y = f(x_1, x_2, ... , x_n;c)$) 
- **Robust**: To handle ill-conditioned problems, bound constraints, uniform/non-uniform weights
- **Efficient**: Broyden updates to minimize function evaluations when far from minimum
- **Complete Model Statistics**: To display covariance, correlation, R², reduced χ² 

## Files

- **`lm.py`** - Core Levenberg-Marquardt algorithm (800 lines, heavily commented)
- **`lm_examples.py`** - Three example problems with sensitivity analysis tools in `multivarious/examples/lm_examples.py`

## Quick Start

```python
import numpy as np
from multivarious.fit.lm import levenberg_marquardt

# Define the model function
def model(t, coeffs):
    return coeffs[0] * np.exp(-t / coeffs[1])

# Generate or load data
t = np.linspace(0, 5, 100)
y_data = 10 * np.exp(-t / 2) + 0.5 * np.random.randn(len(t))

# Initial guess
coeffs_init = np.array([5.0, 1.0])

# Fit the model
result = levenberg_marquardt(
    model, coeffs_init, t, y_data,
    print_level=1  # 0=silent, 1=summary, 2=iterations, 3=iterations+plots
)

print(f"Fitted coefficients: {result.coefficients}")
print(f"Standard errors:     {result.sigma_coefficients}")
print(f"R²:                  {result.r_squared:.4f}")
print(f"Reduced χ²:          {result.reduced_chi_sq:.4f}")

# Access confidence intervals
print(f"95% CI on fit: [{result.sigma_fit * 1.96}]")
```

### Watch Convergence in Real-Time

Set `print_level=3` to see the algorithm adapt to the problem at hand:

```python
result = levenberg_marquardt(
    model, coeffs_init, t, y_data,
    print_level=3  # Shows iteration plots!
)
```

This displays:

- **Current fit** overlaid on data at each iteration
- **Real-time λ adaptation** (large when far, small when close)
- **Coefficient evolution** toward optimal values

## Run Examples

Three built-in examples demonstrate different difficulty levels:

```python
from lm_examples import run_example

# Example 1: Polynomial (medium difficulty)
# Has local minima, tests robustness
run_example(example_number=1, print_level=2)

# Example 2: Exponential decay (easy)  
# Poor initial guess acceptable, fast convergence
run_example(example_number=2, print_level=2)

# Example 3: Exponential + sinusoidal (difficult)
# Requires good initial guess for frequency parameter
run_example(example_number=3, print_level=2)
```

Each example:

- Generates synthetic noisy data
- Performs curve fitting
- Creates three diagnostic plots:
  1. **Convergence history** (coefficients, χ², λ vs iteration)
  2. **Fit quality** (data, fit, 95% and 99% confidence intervals)
  3. **Residual histogram** (check normality assumption)

## How This Compares to SciPy

| Feature                     | This Implementation              | `scipy.optimize`            |
| --------------------------- | -------------------------------- | --------------------------- |
| **Confidence intervals**    | ✅ Built-in (χ² based)            | ❌ Manual calculation needed |
| **Iteration visualization** | ✅ Real-time plots                | ❌ Not available             |
| **Algorithm transparency**  | ✅ Explicit λ adaptation          | ❌ Hidden in trust-region    |
| **Jacobian strategies**     | ✅ FD + Broyden hybrid            | ✅ Multiple options          |
| **Convergence diagnostics** | ✅ Detailed history               | ⚠️ Limited info             |
| **Educational value**       | ✅ Clear, commented code          | ⚠️ Production-optimized     |
| **Update strategies**       | ✅ 3 methods to compare           | ⚠️ Trust-region only        |
| **Speed**                   | ⚠️ Good (pure Python)            | ✅ Excellent (compiled)      |
| **Recommended use**         | **Learning, Teaching, Research** | **Production**              |

| Feature                  | Your `lm.py` | SciPy        | statsmodels |
| ------------------------ | ------------ | ------------ | ----------- |
| **Parameters**           | ✅            | ✅            | ✅           |
| **Standard errors**      | ✅ Auto       | ❌ Manual     | ✅           |
| **Covariance**           | ✅ Auto       | ⚠️ Sometimes | ✅           |
| **Correlation**          | ✅ Auto       | ❌ Manual     | ✅           |
| **R²**                   | ✅ Auto       | ❌ Manual     | ✅           |
| **Reduced χ²**           | ✅ Auto       | ❌ Manual     | ⚠️          |
| **AIC/BIC**              | ✅ Auto       | ❌ Manual     | ✅           |
| **Confidence intervals** | ✅ Auto       | ❌ Manual     | ✅           |
| **Convergence history**  | ✅ Auto       | ⚠️ Limited   | ❌           |
| **Iteration plots**      | ✅ Auto       | ❌            | ❌           |
| **Nonlinear models**     | ✅ Easy       | ✅ Easy       | ⚠️ Complex  |
| **Educational clarity**  | ✅ Excellent  | ⚠️ Opaque    | ⚠️ Complex  |

### When to Use Each

**Use this implementation when:**

- Teaching numerical optimization
- Understanding LM algorithm internals
- Comparing different update strategies
- Need confidence intervals without extra work
- Debugging optimization problems (detailed output)

**Use SciPy when:**

- Production code (speed critical)
- Standard fitting tasks
- Large-scale problems (compiled performance)
- Integration with broader SciPy ecosystem

## Sensitivity Analysis

Run `multivarious/examples/lm_examples.py` to run 100 random initial guesses and visualize which ones converge to the global minimum vs. local minima.

## The Levenberg-Marquadt Algorithm

### 1. The Damping Parameter λ

The Levenberg-Marquardt algorithm **adapts** between two extremes:

```
λ → 0:   Gauss-Newton    (fast near minimum, quadratic convergence)
λ → ∞:   Gradient Descent (stable far from minimum, slow but sure)
```

**How λ adapts:**

- **Start*** with λ₀ (default: 0.01)
- **Good step** (χ² decreased): λ ← λ/9 (move toward Gauss-Newton)
- **Bad step** (χ² increased): λ ← λ×11 (move toward gradient descent)

You can **watch this happen** with `print_level=2`:

```
>  1: 10 | chi_sq=1.023e+02 | lambda=1.1e-01   ← Large λ, far from minimum
>  2: 12 | chi_sq=5.134e+01 | lambda=1.2e-02   ← Good step, λ decreased
>  3: 14 | chi_sq=1.892e+01 | lambda=1.4e-03   ← Getting close, λ → 0
>  4: 16 | chi_sq=1.025e+00 | lambda=1.5e-04   ← Near minimum, almost GN
```

### 2. Jacobian update

**Hybrid approach for efficiency:**

1. **Finite Differences** (expensive, accurate)
   
   - Every 2n iterations (n = number of coefficients)
   - When χ² increases (bad step)
   - Uses: 2n function evaluations (central differences) or n evaluations (one-sided differences)

2. **Broyden Update** (cheap, approximate)
   
   - Between FD refreshes
   - Rank-1 update: `J_new = J_old + correction`
   - Uses: 0 function calls!

**Why this matters:**

- Far from minimum: Broyden approximation is good enough
- Near minimum: Periodic FD refresh for accurate convergence
- Saves ~50% of function evaluations!

Set `delta_coeffs < 0` for one-sided differences (cheaper, less accurate).

### 3. Damping parameter update

| Strategy                         | When to Use                          | Lambda Update Rule        |
| -------------------------------- | ------------------------------------ | ------------------------- |
| **Levenberg-Marquardt** (Type 1) | General purpose, most robust         | `λ/9` or `λ×11`           |
| **Quadratic** (Type 2)           | Smooth problems, expensive functions | Line search optimization  |
| **Nielsen** (Type 3)             | Modern, adaptive                     | `λ × max(1/3, 1-(2ρ-1)³)` |

Compare them:

```python
for update_type in [1, 2, 3]:
    result = levenberg_marquardt(
        model, coeffs_init, t, y_data,
        update_type=update_type
    )
    print(f"Type {update_type}: {result.func_calls} function calls")
```

## Algorithm Parameters

### 1. Convergence Tolerances

The algorithm stops when **any one** of these criteria is met:

```python
result = levenberg_marquardt(
    func, coeffs_init, t, y_data,
    tol_gradient=1e-3,   # ||gradient|| < tol → minimum found
    tol_coeffs=1e-3,     # max(|Δa[i]/a[i]|) < tol → converged
    tol_chi_sq=1e-1,     # χ²/DoF < tol → fit good enough
    max_iter=100         # Safety limit
)
```

**Tuning advice:**

- **Strict convergence**: Reduce all tolerances to 1e-6
- **Fast approximation**: Increase to 1e-2
- **Poor initial guess**: Increase `max_iter` (default: 10n²)

### 2. The Damping Parameter (lambda)

```python
result = levenberg_marquardt(
    func, coeffs_init, t, y_data,
    lambda_init=1e-2,        # Starting damping (larger = more conservative)
    lambda_up_factor=11,     # Increase rate on bad steps
    lambda_dn_factor=9,      # Decrease rate on good steps
    tol_improvement=1e-1     # Accept step if ρ > this
)
```

**Tuning advice:**

- **Ill-conditioned problem**: Increase `lambda_init` to 1e-1 or 1
- **Good initial guess**: Decrease `lambda_init` to 1e-3
- **Oscillating**: Decrease `lambda_up_factor` to 5

### 3. Jacobian Options

```python
delta_coeffs = np.array([0.01, 0.01, -0.01, 0])  # Per-coefficient control
#                         ^^^^  ^^^^  ^^^^^  ^
#                       central central one-sided fixed

result = levenberg_marquardt(
    func, coeffs_init, t, y_data,
    delta_coeffs=delta_coeffs
)
```

- `> 0`: Central differences (most accurate, 2× function calls)
- `< 0`: One-sided differences (faster, less accurate)
- `= 0`: Hold coefficient fixed (useful for partial fits)

### 4. Practical Parameter Combinations

**Robust (recommended for getting started):**

```python
opts = [2, 200, 1e-3, 1e-3, 1e-1, 1e-1, 1e-2, 11, 9, 1]
#       ^  ^^^  ^^^^  ^^^^  ^^^^  ^^^^  ^^^^  ^^  ^  ^
#     print iter eps1  eps2  eps3  eps4  lam0  up dn type
```

**Fast approximation:**

```python
opts = [0, 50, 1e-2, 1e-2, 1e-1, 1e-1, 1e-3, 5, 5, 1]
```

**High precision:**

```python
opts = [1, 500, 1e-6, 1e-6, 1e-2, 1e-2, 1e-2, 11, 9, 3]  # Nielsen update
```

### 5. To explore algorithmic options

**1: Explore Algorithm Behavior**

```python
# Compare three update strategies
for update_type in [1, 2, 3]:
    result = run_example(3, print_level=0)
    # Plot convergence rate, explain differences
```

**2: Explore Initial Guess Sensitivity**

```python
sensitivity_to_initial_guess(example_number=3, n_trials=50)
# Identify basins of attraction, explain failures
```

**3: Explore Jacobian Strategies**

```python
# Central vs one-sided differences
delta_central = 0.01
delta_onesided = -0.01
# Compare accuracy vs computational cost
```

**4: Assess Your Own Problem**

```python
# - Assess Fitted parameters via their uncertainties
# - Assess the model adequacy via the reduced χ² 
# - Assess model assumptions via residual analysis 
```

### Understand

- Why optimization is iterative (not closed-form)
- How trust regions adapt to problem geometry
- Trade-offs between accuracy and computational cost
- Importance of initial guesses for nonlinear problems
- Statistical interpretation of fitted parameters
- How to diagnose convergence issues

## The algorithm is dimension-agnostic

This implementation of levenberg-marquard works for any number of independent variables by providing the independent variables are in a **2D array**:

```python
# 1 variable (just time t)
t.shape = (100,)           # 1D array - special case

# 2 variables (x(t), y(t)
t.shape = (200, 2)         # 2D array

# 5 variables ( p(t), q(t), r(t), s(t), u(t) )
t.shape = (300, 5)         # 2D array
```

- Rows = number of observations 
- Columns = number of independent variables
- Always `t.ndim = 2` (except single-variable case)

### 1. Function Evaluation

```python
# Algorithm just calls:
y_hat = func(t, coeffs)

# Your function extracts variables:
def func_10d(t, coeffs):
    p, q, r, s, u, v, w, x, y, z = [t[:, i] for i in range(10)]
    # Build model from all variables
    return model_expression
```

### 2. Jacobian Computation

```python
# Finite differences perturb COEFFICIENTS, not variables
for i, coeff in enumerate(coeffs):
    coeffs_perturbed[i] += delta
    y_plus = func(t, coeffs_perturbed)   # t unchanged!
    J[:, i] = (y_plus - y) / delta
```

**Key:** Jacobian perturbs *coefficients*, not *independent variables*. Works for any `t` shape!

### 3. Residuals

```python
# Always produces 1D vector
residuals = y_data - func(t, coeffs)
# Shape: (n_points,) regardless of how many variables in t
```

### 4. Chi-Squared

```python
chi_sq = residuals.T @ (residuals * weight)
# Scalar value, doesn't depend on t.shape
```

### 5. Dimension Validation

```python
# In lm.py, lines 221-228:
if t.ndim == 1:
    if len(t) != n_points:
        raise ValueError(...)
elif t.ndim == 2:
    if t.shape[0] != n_points:  # ✓ Only checks number of points
        raise ValueError(...)    # ✓ Doesn't care about t.shape[1]!
else:
    raise ValueError(...)
```

- ✓ Checks `t.shape[0]` matches data points
- ✓ **Doesn't limit** `t.shape[1]` (number of variables)
- ✓ Works for 2, 10, 1000 variables automatically!

## Examples of fittig higher dimensions

### 1. Fit models with 2 independent variables

```python
# Example 4: (x, y) → z
n_points = 200
x = np.random.rand(n_points)
y = np.random.rand(n_points)
t = np.column_stack([x, y])  # Shape: (200, 2)

def func_2d(t, coeffs):
    x, y = t[:, 0], t[:, 1]
    return coeffs[0] * x**coeffs[1] + (1-coeffs[0]) * y**coeffs[1]

result = levenberg_marquardt(func_2d, coeffs_init, t, z_data)
```

### 2. Fit models with 5 independent variables

```python
# (p, q, r, s, u) → z
n_points = 300
p = np.random.rand(n_points)
q = np.random.rand(n_points)
r = np.random.rand(n_points)
s = np.random.rand(n_points)
u = np.random.rand(n_points)
t = np.column_stack([p, q, r, s, u])  # Shape: (300, 5)

def func_5d(t, coeffs):
    p, q, r, s, u = [t[:, i] for i in range(5)]
    return (coeffs[0]*p**2 + coeffs[1]*q**2 + coeffs[2]*r + 
            coeffs[3]*np.sin(s) + coeffs[4]*np.exp(u))

result = levenberg_marquardt(func_5d, coeffs_init, t, z_data)
```

### 3. Fit models with 10 independent variables

```python
# (p, q, r, s, u, v, w, x, y, z) → output
n_points = 1000
n_vars = 10

# Generate random data for all 10 variables
vars_data = [np.random.rand(n_points) for _ in range(n_vars)]
t = np.column_stack(vars_data)  # Shape: (1000, 10)

def func_10d(t, coeffs):
    # Extract all 10 variables
    p, q, r, s, u, v, w, x, y, z = [t[:, i] for i in range(10)]

    # Your complex model here
    return (coeffs[0]*p + coeffs[1]*q**2 + coeffs[2]*np.sin(r) + 
            coeffs[3]*s*u + coeffs[4]*np.exp(v) + ...)

result = levenberg_marquardt(func_10d, coeffs_init, t, output_data)
```

### Visualization in higher dimensions

As with any high-dimensional fitting problem, proper visualization
is more difficult.   Here are some suggestions.  

#### 1. In 2D or 3D

```python
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z_data)
ax.scatter(x, y, z_fit)
```

#### 2. In higher dimensions

**Option 1: Pairwise Residual Plots**

```python
# Plot residuals vs each variable separately
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, var_name in enumerate(['p', 'q', 'r', 's', 'u']):
    axes.flat[i].scatter(t[:, i], residuals)
    axes.flat[i].set_xlabel(var_name)
    axes.flat[i].set_ylabel('Residuals')
```

**Option 2: Predicted vs Actual**

```python
# Single plot showing fit quality
plt.scatter(y_data, y_fit)
plt.plot([min, max], [min, max], 'r--', label='Perfect fit')
plt.xlabel('Actual')
plt.ylabel('Predicted')
```

**Option 3: Convergence History Only**

```python
# Focus on optimization progress, not spatial visualization
plt.semilogy(cvg_history[:, 0], cvg_history[:, n_coeffs+1])
plt.xlabel('Function Evaluations')
plt.ylabel('χ²')
```

## Suggestions for High-Dimensional Fitting

### 1. Vectorize Computations

* Partially vectorized 
  
  ```python
  def func_10d(t, coeffs):
    result = np.zeros(len(t))
    for i in range(10):
        result += coeffs[i] * t[:, i]**2
    return result
  ```

* Fully vectorized 
  
  ```python
  def func_10d(t, coeffs):
    return np.sum(coeffs * t**2, axis=1)
  ```

### 2. Scale Variables

* Normalize independent variables to similar numerical ranges ~ [-1:1]
  
  ```python
  t_scaled = (t - t.mean(axis=0)) / t.std(axis=0)
  ```

* Use scaled version in fitting
  
  ```python
  result = levenberg_marquardt(func, coeffs_init, t_scaled, y_data)
  ```

### 3. Check for Multicollinearity

* If variables are highly correlated, fitting can be unstable
  
  ```python
  correlation_matrix = np.corrcoef(t.T)
  print(correlation_matrix)
  ```

* Look for correlations > 0.9 between variables and consider removing redundant variables

### 4. Regularization for Many Parameters

* With more than around 20 model coefficients, consider:
  - L1 regularization (LASSO) for sparse models
  - L2 regularization (Ridge) for stability
  - These aren't built into lm.py but can be added to chi_sq
  - The multivarious package implements L1 regularization with a QP formulation

## Testing Framework for fitting in higher dimensions

```python
def test_nd_fitting(n_vars=10, n_coeffs=10, n_points=500):
    """
    Generic test for n-dimensional fitting.
    """
    # Generate random data
    t = np.random.rand(n_points, n_vars)
    coeffs_true = np.random.randn(n_coeffs)

    # Simple linear model for testing
    def func(t_in, c):
        # Use first n_coeffs variables
        return np.sum(c[:, None] * t_in[:, :n_coeffs].T, axis=0)

    # Generate noisy data
    y_data = func(t, coeffs_true) + 0.1*np.random.randn(n_points)

    # Fit
    coeffs_init = np.ones(n_coeffs)
    result = levenberg_marquardt(func, coeffs_init, t, y_data)

    # Check convergence
    error = np.linalg.norm(result.coefficients - coeffs_true)
    print(f"{n_vars}D fitting: error = {error:.6f}")

    return error < 1.0  # Success criterion

# Test increasing dimensions
for n in [2, 5, 10, 20, 50]:
    success = test_nd_fitting(n_vars=n, n_coeffs=min(n, 10))
    print(f"  {'✓' if success else '✗'} {n} variables")
```

## Summary

| Dimensions   | Works? | Visualization               | Notes                   |
| ------------ | ------ | --------------------------- | ----------------------- |
| 1D (t → y)   | ✓      | Easy (2D plot)              | Standard case           |
| 2D (x,y → z) | ✓      | Good (3D plot)              | Example 4               |
| 3D-5D        | ✓      | Moderate (pairwise plots)   | Tested with 5D example  |
| 6D-20D       | ✓      | Hard (residual plots)       | Algorithm works fine    |
| 20D+         | ✓      | Very hard (statistics only) | Check multicollinearity |

## Computational considerations

### 1. Memory Usage

* Storage requirements scale with:
  
  - n_points (number of data points)
  - n_coeffs (number of parameters to fit)

* NOT with n_vars (number of independent variables)!

* Jacobian: (n_points, n_coeffs)

* For n_points=1000, n_coeffs=10:

* J.size = 10,000 floats ≈ 80 KB

### 2. Speed

* Bottleneck is function evaluation in Jacobian
  - `n_vars` doesn't affect speed (assuming func is simple enough)
* Each Jacobian computation requires:
  - Central differences: 2*n_coeffs function calls
  - One-sided: n_coeffs function calls
* With 10 coefficients, central differences:
  → 20 function evaluations per Jacobian update

## Integration with the multivarious package

This module is part of the **multivarious** package for signal processing, model fitting, linear time invariant systems, ordinary differential equations, optimization, and random variables :

```python
# When installed via pip:
from multivarious.fit.lm import levenberg_marquardt, lm
from multivarious.utl.plot_lm import plot_lm

```

modules emphasize:

- Clarity over performance
- Algorithmic transparency
- Comparison with production libraries
- Examples and validation

## Standard Dependencies

- NumPy ≥ 1.20

- Matplotlib ≥ 3.5

## References & Further Reading

### Primary References

1. **Levenberg, K.** (1944) "A method for the solution of certain non-linear problems in least squares", *Quarterly of Applied Mathematics* 2: 164-168.
   
   - Original Levenberg damping idea

2. **Marquardt, D.** (1963) "An algorithm for least-squares estimation of nonlinear parameters", *SIAM Journal on Applied Mathematics* 11(2): 431-441.
   
   - Diagonal scaling that makes the algorithm practical

3. **Press, W.H., Teukolsky, S.A., Vetterling, W.T., Flannery, B.P.** (1992) *Numerical Recipes*, Cambridge University Press, Chapter 15.
   
   - Excellent pedagogical treatment

4. **Madsen, K., Nielsen, H.B., Tingleff, O.** (2004) "Methods for Non-Linear Least Squares Problems", IMM, Technical University of Denmark.
   
   - Modern comprehensive treatment, free PDF

### Additional Resources

- **Roweis, S.** Notes on LM: http://www.cs.toronto.edu/~roweis/notes/lm.pdf
- **Lourakis, M.** levmar package: http://www.ics.forth.gr/~lourakis/levmar/
- **Nielsen, H.B.** Damping parameter update: http://www2.imm.dtu.dk/~hbn/publ/TR9905.ps

## Author & Acknowledgments

Henri Gavin, Department of Civil & Environmental Engineering, Duke University

**Part of the `multivarious` package:**

- Tools for digital signal processing, model fitting, linear time invariant systems, ordinary differential equations, optimization, and random variables 
- Open source, MIT license
- https://github.com/hpgavin/multivarious

## Citation

```bibtex
@software{lm_python_2024,
  author = {Gavin, Henri P.},
  title = {The Levenberg-Marquardt algorithm for Nonlinear Least Squares Curve Fitting Problems: 
            Python Implementation},
  year = {2024},
  url = {https://github.com/hpgavin/multivarious}
}
```

## License

MIT License - Free for educational and commercial use.
