# The Levenberg-Marquardt algorithm for nonlinear least-squares curve fitting problems

Implementatiom of the Levenberg-Marquardt algorithm for nonlinear curve fitting.  Source code with transparent structure for pedagogically-oriented applications.

## Why This Implementation?

This implementation fills a gap between production libraries (SciPy) and educational needs:

### **Unique Features Not in SciPy**

1. **Confidence Intervals** - Chi-squared based uncertainty quantification for both coefficients and fitted curve
2. **Three Update Strategies** - Compare Levenberg-Marquardt, Quadratic, and Nielsen λ adaptation
3. **Iteration Visualization** - Watch the fit converge in real-time (set `print_level=3`)
4. **Explicit Algorithm Structure** - See exactly how damping parameter λ adapts
5. **Broyden Jacobian Updates** - Efficient rank-1 approximation when appropriate
6. **Hybrid Jacobian Strategy** - Automatic switching between finite differences and Broyden updating based on convergence history

### **Pedagogical Advantages**

- **Transparent Code**: No hidden abstractions - students see every algorithmic step
- **Well-Documented**: Extensive comments explain the "why" not just the "what"  
- **Multiple Examples**: Three problems of increasing difficulty demonstrate algorithm behavior
- **Convergence Diagnostics**: Detailed iteration output shows λ adaptation in action
- **Sensitivity Analysis**: Tools to explore initial guess dependence

### **Production Quality**

- **Robust**: Handles ill-conditioned problems, bound constraints, uniform/non-uniform weights
- **Efficient**: Broyden updates minimize function evaluations when far from minimum
- **Complete Statistics**: Covariance, correlation, R², reduced χ² - everything you need

## Files

- **`lm.py`** - Core Levenberg-Marquardt algorithm (800 lines, heavily commented)
- **`lm_examples.py`** - Three example problems with sensitivity analysis tools

## Quick Start

```python
import numpy as np
from lm import levenberg_marquardt

# Define your model function
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

Demonstrate how initial guess affects convergence:

```python
from lm_examples import sensitivity_to_initial_guess

sensitivity_to_initial_guess(example_number=3, n_trials=100)
```

This runs 100 random initial guesses and visualizes which ones converge to the global minimum vs. local minima.

## Understanding the Algorithm

### The Damping Parameter λ

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

### Three Update Strategies

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

### Jacobian Computation Strategy

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

- Far from minimum: Broyden approximation good enough
- Near minimum: Periodic FD refresh for accurate convergence
- Saves ~50% of function evaluations!

Set `delta_coeffs < 0` for one-sided differences (cheaper, less accurate).

## Algorithm Parameters (Advanced)

### Convergence Tolerances

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

### Lambda Control

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

### Jacobian Options

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

### Practical Parameter Combinations

**Robust (recommended for teaching):**

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

## For Exploration of Algorithmic Options

### Suggested Assignments

**Assignment 1: Algorithm Behavior**

```python
# Compare three update strategies
for update_type in [1, 2, 3]:
    result = run_example(3, print_level=0)
    # Students: Plot convergence rate, explain differences
```

**Assignment 2: Initial Guess Sensitivity**

```python
sensitivity_to_initial_guess(example_number=3, n_trials=50)
# Students: Identify basins of attraction, explain failures
```

**Assignment 3: Jacobian Strategies**

```python
# Central vs one-sided differences
delta_central = 0.01
delta_onesided = -0.01
# Students: Compare accuracy vs computational cost
```

**Assignment 4: Your Own Problem**

```python
# Fit data from lab experiment, report:
# - Fitted parameters with uncertainties
# - Reduced χ² (is model adequate?)
# - Residual analysis (are assumptions met?)
```

### Learning Outcomes

After using this implementation, students will understand:

- Why optimization is iterative (not closed-form)
- How trust regions adapt to problem geometry
- Trade-offs between accuracy and computational cost
- Importance of initial guesses for nonlinear problems
- Statistical interpretation of fitted parameters
- How to diagnose convergence issues

## Integration with `multivarious` Package

This module is part of the **multivarious** educational signal processing and optimization package:

```python
# When installed via pip:
from multivarious.lm import levenberg_marquardt
```

All modules emphasize:

- Pedagogical clarity over performance
- Complete algorithmic transparency
- Comparison with production libraries
- Real-world examples and validation

## Dependencies

**Required:**

- NumPy ≥ 1.20

**Optional (for examples and plotting):**

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

- Tools for digital signal processing,  linear systems, optimization,ordinary differential equations, and random variables
- Open source, MIT license
- https://github.com/hpgavin/multivarious

## Citation

If you use this in academic work:

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
