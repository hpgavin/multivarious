The "ill-posed" error messages from SciPy's QP solver often indicate numerical conditioning issues, which are common in power distribution problems (especially with equality constraints or near-singular Hessians).

MATLAB has its own **embedded QP solver** (`mwQP`) which is robust for ill-conditioned problems. It includes:

1. **Active set method** with QR factorization
2. **Explicit checks** for ill-posed problems (line 869)
3. **Anti-cycling rules** (Bland's rule, line 957-961)
4. **Careful numerical conditioning**

The **embedded QP solver** (`mwQP`) should be much more robust. 

## Key Features

**1. Embedded QP Solver (`mwQP` function)**
- **Active-set method** with QR factorization for numerical stability
- **Explicit handling** of ill-posed problems (doesn't just throw errors)
- **Anti-cycling rules** (Bland's rule)
- **Careful numerical conditioning** checks
- **Equality constraint support**

**2. QR Factorization Helpers**
- `qr_insert()` - adds a constraint to the active set
- `qrdelete()` - removes a constraint from the active set
- Both use **Givens rotations** for numerical stability

**3. Robustness for Ill-Conditioned Problems**
- Checks if Hessian is positive definite
- Falls back to gradient-based search if needed
- Handles near-singular working sets
- Returns status messages instead of crashing

**4. Complete BFGS Hessian Updates**
- Ensures positive definiteness
- Gradient modification when needed
- Handles small or negative curvature

## Why This Is Better Than SciPy

The **"ill-posed"** errors you're getting from SciPy happen because it just gives up when:
- The Hessian is singular or near-singular
- The constraints are linearly dependent
- The working set becomes ill-conditioned

The embedded QP solver instead:
- ✅ Detects these conditions explicitly
- ✅ Uses QR factorization (more stable than Cholesky)
- ✅ Has fallback strategies instead of just failing
- ✅ Returns meaningful status codes

## Usage

```python
from multivarious.opt import sqp

v_opt, f_opt, g_opt, cvg_hst, lambda_opt, hess = sqp(
    func, v_init, v_lb, v_ub, options, consts
)
```

The QP solver will handle ill-conditioned cases that scipy.optimize can't.

Here are **critical bugs** I corrected:

## Key Fixes in `mwQP()`

### 1. **Step Length Computation** (Lines 144-157)
**Was:**
```python
cstr = A @ X - b
indix = np.where(~aix)[0]
dist = (cstr[indix] / (A[indix, :] @ SD + errnorm))
dist[dist < 0] = np.inf
```

**Now:**
```python
GSD = A @ SD  # Compute once: change in constraints per unit step
cstr = A @ X - b
indf = np.where((GSD > errnorm * np.linalg.norm(SD)) & (~aix))[0]
dist = np.abs(cstr[indf] / GSD[indf])
```

**Why this matters:**
- `GSD = A @ SD` computes how much each constraint changes per unit step
- We only consider inactive constraints where `GSD > 0` (moving toward violation)
- This prevents adding singular constraints to the working set
- Matches the MATLAB logic exactly (line 794-806)

### 2. **Bland's Anti-Cycling Rule** (Line 156)
**Was:**
```python
ind_step = indix[np.argmin(dist)]
```

**Now:**
```python
ind2 = np.where(dist == STEPMIN)[0]
ind = indf[ind2[np.argmin(indf[ind2])]]
```

**Why:** If multiple constraints have the same minimum distance, pick the one with the smallest index to prevent cycling.

### 3. **Equality Constraint Handling** (Line 79)
**Was:**
```python
eqix = np.arange(neqcstr)
```

**Now:**
```python
eqix = np.arange(neqcstr) if neqcstr > 0 else np.array([], dtype=int)
```

**Why:** Avoids indexing errors when there are no equality constraints.

### 4. **Constraint Update After Each Step** (Lines 205-208)
**Added:**
```python
cstr = A @ X - b
if len(eqix) > 0:
    cstr[eqix] = np.abs(cstr[eqix])
```

**Why:** The MATLAB code recalculates constraints after each step and treats equality constraints with absolute values. This ensures constraints are properly tracked.

###5. **Active Set Management** 
Restructured the logic to match MATLAB's exact flow when `STEPMIN >= 1`, including proper handling of the simplex case and constraint removal.

These fixes ensure that **constraints are actively driven to zero** through the proper step length computation and active set management. The scipy version works because it handles this internally, but the embedded solver requires the exact MATLAB logic for robustness.

---

1.  **Language-specific Adaptations**:
    *   MATLAB's 1-based indexing was converted to 0-based indexing for Python/NumPy. This required adjustments in loop ranges, `CIND`, `ACTIND`, and `eqix`.
    *   The `nargin` logic was replaced with Python's default argument values.
    *   MATLAB's `eye`, `zeros`, `ones`, and `abs` functions were replaced with `np.eye`, `np.zeros`, `np.ones`, and `np.abs`.
    *   The backslash operator `\` was replaced by `np.linalg.solve`, `scipy.linalg.solve_triangular`, or `np.linalg.lstsq` depending on the matrix properties and the specific context in the code.

2.  **QR Update Utilities**:
    *   `qr_insert`: Translated line-by-line. MATLAB's `planerot` was implemented as a helper function.
    *   `qrdelete`: MATLAB's built-in `qrdelete` was implemented manually using Givens rotations to restore upper triangular form after a column removal, as SciPy does not provide a direct equivalent.

3.  **Linear Algebra**:
    *   `scipy.linalg.qr` with `mode='full'` was used to match MATLAB's `qr` behavior, particularly for extracting the null space basis `Z` from the `Q` matrix.
    *   `scipy.linalg.solve_triangular` was used for solving systems involving the `R` matrix to maintain numerical behavior consistent with MATLAB's triangular solvers.

4.  **Handling Dimensions**:
    *   MATLAB often treats vectors as 2D matrices (e.g., `n x 1`). Python's 1D arrays were used for vectors `f`, `B`, `X`, and `lambda_vec` for idiomatic NumPy usage, while ensuring matrix-vector multiplications (@) behave correctly.

5.  **Recursive Calls**:
    *   The recursive call to `mwQP` for finding the initial feasible solution was preserved exactly, including the parameter passing for Phase 1 (triggered by `verbosity == -2`).

6.  **Edge Cases**:
    *   Singular matrix handling in the active set (using `lstsq` as a fallback) was included as per the original `Inf` check in MATLAB.
    *   Empty constraint matrix `A` handling was added to prevent errors in shape detection.

7.  **Potential Issues**:
    *   Numerical precision in QR updates (Givens rotations) can vary slightly between MATLAB and Python implementations due to floating-point differences, but the logic remains mathematically equivalent.

