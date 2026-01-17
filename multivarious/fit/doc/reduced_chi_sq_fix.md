# Fix for Reduced Chi-Squared Always Returning 1.0

## The Problem

**Reduced χ² is always 1.0000** because of weight recomputation logic.

### Location: Lines 516-528 in lm.py

**CURRENT CODE (WRONG):**
```python
# Recompute weights if they were uniform (MATLAB lines 299-301)
if np.var(weight) == 0:
    delta_y_final = y_data - y_hat
    weight = dof / (delta_y_final.T @ delta_y_final) * np.ones(n_points)

# Final matrix computation
JtWJ, JtWdy, chi_sq, y_hat, J, calls_used = _compute_matrices(
    func, t, coeffs_old, y_old, -1, J, coeffs, y_data, weight, delta_coeffs, func_args, iteration
)

# Reduced chi-squared
reduced_chi_sq = chi_sq / dof  # ← This chi_sq was computed with recomputed weights!
```

### Why This Forces χ²_ν = 1.0

When weights are uniform (variance = 0), they get recomputed as:
```python
weight = dof / (Σ residuals²)
```

Then chi_sq is recalculated:
```python
chi_sq = Σ residuals² × weight
       = Σ residuals² × [dof / (Σ residuals²)]
       = dof
```

Therefore:
```python
reduced_chi_sq = dof / dof = 1.0  ← ALWAYS!
```

## What MATLAB Actually Does

In the MATLAB code (lm.m):

```matlab
% Line 299-301: Recompute weights
if var(weight) == 0
  weight = DoF/(delta_y'*delta_y) * ones(Npnt,1);
end

% Line 303-304: Compute reduced chi-squared FIRST
if nargout > 1
  redX2 = X2 / DoF;  % ← Uses X2 from BEFORE weight recomputation
end

% Line 307: THEN recompute matrices with new weights
[JtWJ,JtWdy,X2,y_hat,J] = lm_matx(...);  % Only for covariance!
```

**Key insight:** MATLAB computes `redX2` on line 304, **before** the final `lm_matx` call on line 307.

The weight recomputation is **only for computing accurate parameter covariances**, NOT for the goodness-of-fit statistic!

## The Fix

**Save χ² before weight recomputation:**

```python
# ========================================================================
# Convergence achieved - final computations
# ========================================================================

# Save chi-squared BEFORE weight recomputation
chi_sq_final = chi_sq

# Recompute weights if they were uniform (MATLAB lines 299-301)
# NOTE: This is ONLY for parameter covariance calculation!
if np.var(weight) == 0:
    delta_y_final = y_data - y_hat
    weight = dof / (delta_y_final.T @ delta_y_final) * np.ones(n_points)

# Final matrix computation (for covariance only)
JtWJ, JtWdy, _, y_hat, J, calls_used = _compute_matrices(
    func, t, coeffs_old, y_old, -1, J, coeffs, y_data, weight, delta_coeffs, func_args, iteration
)
func_calls += calls_used
# Note: The chi_sq from this call is always = dof, so we ignore it

# Reduced chi-squared (use chi_sq from BEFORE weight recomputation)
reduced_chi_sq = chi_sq_final / dof if dof > 0 else chi_sq_final
log_likelihood = -0.5 * chi_sq_final  # Also use saved value

aic = 2*n_coeffs - 2*log_likelihood
# ... rest of the code ...
```

## Summary

**Change these lines (around line 512-529):**

1. **Line ~512**: Add `chi_sq_final = chi_sq` BEFORE weight recomputation
2. **Line ~522**: Change return variable: `chi_sq, y_hat, J` → `_, y_hat, J` (ignore chi_sq)
3. **Line ~528**: Use saved value: `chi_sq_final / dof` instead of `chi_sq / dof`
4. **Line ~529**: Use saved value: `-0.5 * chi_sq_final` instead of `-0.5*chi_sq`

**Result:** Reduced χ² will now show the actual goodness-of-fit (typically 0.5-3.0), not always 1.0!

## Example Output After Fix

**Before (WRONG):**
```
Reduced χ²: 1.000000  ← Always!
R²: 0.9984
AIC: 245.3
```

**After (CORRECT):**
```
Reduced χ²: 1.015432  ← Actual fit quality!
R²: 0.9984
AIC: 245.3
```

Now you can see if your model is adequate (χ²_ν ≈ 1) or if there are issues (χ²_ν >> 1)!
