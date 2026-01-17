# Multi-Dimensional Fitting: Quick Reference

## YES! Works for 2, 5, 10, 100+ Variables ✓

No code changes needed beyond the one validation fix already applied!

## Data Format

```python
# Single variable (special case)
t = np.array([1, 2, 3, ...])     # Shape: (n_points,)

# Multiple variables (general case)
t = np.column_stack([var1, var2, var3, ...])  # Shape: (n_points, n_vars)
```

## Examples

### 2D: (x, y) → z
```python
x = np.random.rand(200)
y = np.random.rand(200)
t = np.column_stack([x, y])  # Shape: (200, 2)

def model(t, coeffs):
    x, y = t[:, 0], t[:, 1]
    return coeffs[0]*x + coeffs[1]*y

result = levenberg_marquardt(model, [1, 1], t, z_data)
```

### 5D: (p, q, r, s, u) → z
```python
p, q, r, s, u = [np.random.rand(300) for _ in range(5)]
t = np.column_stack([p, q, r, s, u])  # Shape: (300, 5)

def model(t, coeffs):
    p, q, r, s, u = [t[:, i] for i in range(5)]
    return coeffs[0]*p**2 + coeffs[1]*q + coeffs[2]*np.sin(r) + ...

result = levenberg_marquardt(model, initial_guess, t, z_data)
```

### 10D: Any Number of Variables
```python
# Generate n_vars independent variables
n_vars = 10
n_points = 500
vars_data = [np.random.rand(n_points) for _ in range(n_vars)]
t = np.column_stack(vars_data)  # Shape: (500, 10)

def model(t, coeffs):
    # Extract all variables
    variables = [t[:, i] for i in range(n_vars)]
    # Build your model
    return your_complex_function(*variables, coeffs)

result = levenberg_marquardt(model, initial_guess, t, output_data)
```

## Visualization Strategies

| Dimensions | Best Approach |
|------------|---------------|
| 1D | `plt.plot(t, y)` |
| 2D | `ax.scatter(x, y, z)` (3D plot) |
| 3D+ | Residuals vs each variable separately |
| Any | Predicted vs Actual scatter plot |

## Run the Examples

```bash
# Test 2D fitting (Example 4)
python -c "from lm_examples import run_example; run_example(4)"

# Test 5D fitting
python example_5d.py
```

## Key Points

✓ **Algorithm:** Dimension-agnostic, works for any number of variables  
✓ **Validation:** Already correct (checks t.shape[0] only)  
✓ **Performance:** Scales with n_points and n_coeffs, not n_vars  
✓ **Memory:** Minimal - Jacobian is (n_points, n_coeffs)  
⚠ **Visualization:** Gets harder with more dimensions (use residual plots)  

## What Gets Fitted

The LM algorithm fits **COEFFICIENTS**, not variables!

```
Variables (p, q, r, ...) → INPUT (you provide)
Coefficients (a, b, c, ...) → OUTPUT (LM finds optimal values)
```

Number of variables can be huge - doesn't affect optimization difficulty.  
Number of coefficients should be modest (<50 typically) for stability.
