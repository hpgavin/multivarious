# Theory of mimoSHORSA
## Multi-Input Multi-Output Stochastic High Order Response Surface Algorithm

*A Comprehensive Theoretical Overview*

---

## Table of Contents
1. Introduction and Motivation
2. Mathematical Foundation
3. Hermite Polynomial Basis
4. Model Construction Process
5. Iterative Model Reduction
6. Statistical Evaluation Metrics
7. Scaling and Preprocessing
8. Computational Considerations
9. Applications and Use Cases

---

## 1. Introduction and Motivation

### 1.1 The Response Surface Problem

In many engineering and scientific applications, we have:
- **Input variables** X = [X₁, X₂, ..., Xₙ] ∈ ℝⁿ
- **Output variables** Y = [Y₁, Y₂, ..., Yₘ] ∈ ℝᵐ
- A complex, possibly nonlinear relationship Y = f(X)

The goal is to approximate this relationship with a computationally efficient model that:
1. Captures nonlinear behavior
2. Handles high-dimensional inputs
3. Provides uncertainty quantification
4. Remains computationally tractable

### 1.2 Why Response Surfaces?

Traditional approaches face challenges:
- **Direct simulation**: Computationally expensive (finite element, CFD, etc.)
- **Low-order polynomials**: Inadequate for complex nonlinear behavior
- **Neural networks**: Black-box nature, difficult uncertainty quantification
- **Kriging/Gaussian processes**: Computational scaling issues for large datasets

**Response surface methods** provide a middle ground: explicit polynomial models with:
- Analytical derivatives
- Uncertainty quantification
- Interpretable coefficients
- Efficient evaluation

---

## 2. Mathematical Foundation

### 2.1 General Polynomial Representation

mimoSHORSA approximates the input-output relationship as:

```
Y(X) = Σ cⱼ Ψⱼ(X)
       j=1
```

Where:
- **cⱼ**: Model coefficients to be determined
- **Ψⱼ(X)**: Basis functions (products of Hermite functions)
- **nTerm**: Total number of terms in the expansion

### 2.2 Detailed Polynomial Structure

The full expansion includes:

1. **Constant term**: c₀

2. **Pure power terms**: 
   ```
   Σ  Σ  bᵢⱼ Ψⱼ(Xᵢ)
   i=1 j=1
   ```
   where kᵢ is the maximum order for variable Xᵢ

3. **Mixed terms** (cross-products):
   ```
   Σ cq Π Ψₚᵢq(Xᵢ)
   q=1  i=1
   ```

### 2.3 Multi-Output Extension

For multiple outputs, mimoSHORSA fits **separate models** for each output:

```
Yₖ(X) = Σ cₖⱼ Ψₖⱼ(X),  k = 1, 2, ..., m
        j=1
```

Each output can have:
- Different numbers of terms (nTermₖ)
- Different coefficient values
- Different retained terms (after culling)

---

## 3. Hermite Polynomial Basis

### 3.1 Why Hermite Functions?

Standard power polynomials (1, x, x², x³, ...) have problems:
- **Numerical instability** for high orders
- **Poor conditioning** of basis matrices
- **Lack of orthogonality**

Hermite functions provide:
- **Orthogonality** with respect to Gaussian weight
- **Numerical stability** for higher orders
- **Natural scaling** for standardized variables

### 3.2 Hermite Function Definition

The Hermite functions are defined as:

```
ψₙ(z) = (1/√(2ⁿ n! √π)) Hₙ(z) exp(-z²/2)
```

Where:
- **Hₙ(z)**: Hermite polynomial of order n (physicist's version)
- **exp(-z²/2)**: Gaussian weight function
- Normalization ensures orthonormality

### 3.3 Low-Order Hermite Functions

```
ψ₀(z) = π^(-1/4) exp(-z²/2)

ψ₁(z) = √2 π^(-1/4) z exp(-z²/2)

ψ₂(z) = (1/√2) π^(-1/4) (2z² - 1) exp(-z²/2)

ψ₃(z) = (1/√3) π^(-1/4) (2z³ - 3z) exp(-z²/2)

ψ₄(z) = (1/(2√6)) π^(-1/4) (4z⁴ - 12z² + 3) exp(-z²/2)

ψ₅(z) = (1/(2√15)) π^(-1/4) (4z⁵ - 20z³ + 15z) exp(-z²/2)
```

### 3.4 Orthogonality Property

```
∫₋∞^∞ ψₘ(z) ψₙ(z) dz = δₘₙ
```

This orthonormality is key to numerical stability.

### 3.5 Multi-Dimensional Basis Functions

For multi-dimensional inputs, basis functions are **products** of 1D Hermite functions:

```
Ψ(Z) = ψₚ₁(Z₁) × ψₚ₂(Z₂) × ... × ψₚₙ(Zₙ)
```

Where **p** = [p₁, p₂, ..., pₙ] is the **order vector** specifying the polynomial order in each dimension.

---

## 4. Model Construction Process

### 4.1 Three-Stage Algorithm

mimoSHORSA follows a three-stage process:

#### Stage 1: Polynomial Order Determination (Optional)
- Determine optimal maximum order kᵢ for each variable
- Uses 1D Chebyshev sampling and curve fitting
- Currently disabled in practice (uses uniform maximum order)

#### Stage 2: Mixed Term Enumeration
- Generate all possible term combinations
- Filter based on total order constraint
- Create order matrix specifying each term

#### Stage 3: Model Fitting and Reduction
- Fit full model using least squares
- Iteratively remove uncertain terms
- Continue until tolerance criteria met

### 4.2 Term Generation Algorithm

The algorithm generates all combinations where:

```
p₁ ≤ maxOrder₁
p₂ ≤ maxOrder₂
...
pₙ ≤ maxOrderₙ

AND

p₁ + p₂ + ... + pₙ ≤ max(maxOrder)
```

This creates a structured polynomial space with controlled complexity.

**Example** (n=2, maxOrder=[2,2]):
```
[0,0] → ψ₀(Z₁)ψ₀(Z₂) = constant
[1,0] → ψ₁(Z₁)ψ₀(Z₂) = linear in Z₁
[0,1] → ψ₀(Z₁)ψ₁(Z₂) = linear in Z₂
[2,0] → ψ₂(Z₁)ψ₀(Z₂) = quadratic in Z₁
[1,1] → ψ₁(Z₁)ψ₁(Z₂) = bilinear term
[0,2] → ψ₀(Z₁)ψ₂(Z₂) = quadratic in Z₂
```

### 4.3 Least Squares Fitting

Given:
- **Z**: Scaled input data (nInp × mData)
- **Y**: Output data (nOut × mData)
- **B**: Basis matrix (mData × nTerm)

The coefficients are found by solving:

```
min ||B·c - Y||²
 c
```

Solution (via SVD for numerical stability):
```
c = (B^T B)^(-1) B^T Y
```

In practice, singular value decomposition handles ill-conditioned cases.

### 4.4 Design Matrix Construction

The basis matrix **B** has structure:

```
B = [Ψ₁(Z¹) Ψ₂(Z¹) ... ΨₙTₑᵣₘ(Z¹)]
    [Ψ₁(Z²) Ψ₂(Z²) ... ΨₙTₑᵣₘ(Z²)]
    [  ...    ...   ...    ...     ]
    [Ψ₁(Zᵐ) Ψ₂(Zᵐ) ... ΨₙTₑᵣₘ(Zᵐ)]
```

Where:
- Rows correspond to data points
- Columns correspond to terms
- Each entry is the basis function evaluated at that data point

---

## 5. Iterative Model Reduction

### 5.1 Motivation for Reduction

Initial full model often has problems:
- **Overfitting**: Too many parameters relative to data
- **Uncertain coefficients**: High variance estimates
- **Poor generalization**: Fits noise rather than signal
- **Computational cost**: Unnecessary terms

### 5.2 Coefficient of Variation (COV)

For each coefficient, compute:

```
COV(cⱼ) = σ(cⱼ) / |cⱼ|
```

Where the standard error is:

```
σ(cⱼ) = √[(RSS/(m - nTerm)) × (B^T B)⁻¹ⱼⱼ]
```

Components:
- **RSS**: Residual sum of squares = ||Y - Ŷ||²
- **m**: Number of data points
- **nTerm**: Number of model terms
- **(B^T B)⁻¹ⱼⱼ**: Diagonal element of inverse Gram matrix

### 5.3 Culling Strategy

At each iteration:
1. Identify term with **largest COV** (most uncertain)
2. Remove that term from model
3. Refit remaining coefficients
4. Re-evaluate COV for all remaining terms
5. Repeat until: COV < tolerance OR max iterations reached

**Why largest COV?**
- High COV indicates coefficient is poorly determined
- Removal has minimal impact on model quality
- Improves model parsimony and generalization

### 5.4 Stopping Criteria

Iteration stops when:
```
max(COV) < tolerance  AND  ρ_test > 0
```

Where:
- **tolerance**: User-specified (typically 0.05 - 0.15)
- **ρ_test**: Model-data correlation on test set (must be positive)

This ensures both:
- Coefficient certainty (low COV)
- Predictive capability (positive correlation)

---

## 6. Statistical Evaluation Metrics

### 6.1 R-Squared (R²)

Measures explained variance:

```
R² = 1 - (RSS / TSS)
```

Where:
- **RSS** = Σ(Yᵢ - Ŷᵢ)² (residual sum of squares)
- **TSS** = Σ(Yᵢ - Ȳ)² (total sum of squares)

Interpretation:
- R² = 1: Perfect fit
- R² = 0: Model no better than mean
- R² < 0: Model worse than mean (on test data)

### 6.2 Adjusted R-Squared (R²_adj)

Penalizes model complexity:

```
R²_adj = ((m-1)·R² - nTerm) / (m - nTerm)
```

Why adjust?
- Raw R² always increases with more terms
- R²_adj accounts for degrees of freedom
- Prevents overfitting through complexity penalty

### 6.3 Model-Data Correlation (ρ)

Pearson correlation between predictions and observations:

```
ρ = Cov(Y, Ŷ) / (σ_Y · σ_Ŷ)
```

Advantages over R²:
- Scale-invariant
- More interpretable for practitioners
- Robust to offset errors

### 6.4 Coefficient of Variation (COV)

For each coefficient:

```
COV(cⱼ) = SE(cⱼ) / |cⱼ|
```

Interpretation:
- COV < 0.10: Well-determined coefficient
- 0.10 < COV < 0.30: Moderate uncertainty
- COV > 0.30: Highly uncertain, candidate for removal

### 6.5 Condition Number

Measures numerical stability:

```
κ(B) = ||B|| · ||B⁻¹||
```

Interpretation:
- κ < 100: Well-conditioned
- 100 < κ < 1000: Moderate conditioning
- κ > 1000: Ill-conditioned, numerical issues possible

---

## 7. Scaling and Preprocessing

### 7.1 Why Scale?

Raw data problems:
- **Different units**: Variables on vastly different scales
- **Numerical instability**: Extreme values cause conditioning issues
- **Poor Hermite basis fit**: Hermite functions optimal for ~N(0,1) data

### 7.2 Scaling Options

#### Option 0: No Scaling
```
Z = X
```
Use when data already normalized.

#### Option 1: Standardization
```
Z = (X - μ) / σ
```
Centers and scales to unit variance.

#### Option 2: Decorrelation (Whitening)
```
Z = V Λ^(-1/2) V^T (X - μ)
```
Where V·Λ·V^T = Cov(X)

Removes linear correlations between variables.

#### Option 3: Log-Standardization
```
Z = (log₁₀(X) - μ_log) / σ_log
```
For data with multiplicative structure or log-normal distributions.

#### Option 4: Log-Decorrelation
```
Z = V Λ^(-1/2) V^T (log₁₀(X) - μ_log)
```
Combines logarithmic and linear decorrelation.

### 7.3 Transformation Matrices

The transformations are stored as:
```
Z = T⁻¹ (X - μ)
```

Where:
- **μ**: Mean vector
- **T**: Transformation matrix

**Inverse transformation** (for predictions):
```
X = T·Z + μ
```

For log transformations:
```
X = 10^(T·Z + μ)
```

### 7.4 Outlier Removal

After scaling, remove data points where:
```
|Zᵢⱼ| > threshold  (typically 4)
```

Rationale:
- Hermite functions designed for ~N(0,1)
- Extreme values degrade approximation
- Removes potential data errors

---

## 8. Computational Considerations

### 8.1 Computational Complexity

**Term generation**: O(k^n) where k = maxOrder, n = nInp
- Combinatorial explosion for high dimensions
- Filtering reduces to manageable size

**Basis construction**: O(m·nTerm·n) where m = mData
- Linear in number of data points
- Dominates for large datasets

**Least squares solve**: O(m·nTerm² + nTerm³)
- SVD used for numerical stability
- Can be expensive for many terms

**Per-iteration cost**: Dominated by least squares
- Typically 10-50 iterations
- Each iteration removes one term

### 8.2 Conditioning and Stability

**Sources of ill-conditioning**:
1. Highly correlated input variables
2. Insufficient data (m < nTerm)
3. Extreme polynomial orders

**Mitigation strategies**:
1. **Hermite basis**: Better conditioned than power basis
2. **Decorrelation** (scaling option 2 or 4)
3. **SVD-based solve**: Handles near-singular systems
4. **Model reduction**: Removes problematic terms

### 8.3 Train-Test Split

Critical for validation:
- **Training set**: Used for coefficient estimation
- **Test set**: Used for performance evaluation

Typical split: 50-80% training, 20-50% testing

**Why separate?**
- Training metrics overestimate performance
- Test metrics indicate generalization
- Prevents overfitting bias

### 8.4 Memory Requirements

**Storage needs**:
- **B matrix**: O(m·nTerm) - largest structure
- **Data**: O(m·(n+p)) 
- **Coefficients**: O(nTerm·p)

For large problems (m > 10⁶), consider:
- Batch processing
- Out-of-core algorithms
- Reduced precision storage

---

## 9. Applications and Use Cases

### 9.1 Structural Reliability Analysis

**Original motivation** (Gavin & Yau, 2005):
- Approximate limit state functions
- Compute failure probabilities
- Sensitivity analysis

Advantages:
- Explicit failure surface
- Analytical gradients for importance sampling
- Handles high-dimensional random variables

### 9.2 Uncertainty Quantification

Applications:
- **Forward propagation**: Input uncertainty → output uncertainty
- **Sensitivity analysis**: Which inputs matter most?
- **Reliability**: Probability of exceeding thresholds

mimoSHORSA provides:
- Coefficient uncertainties (COV)
- Analytical variance propagation
- Importance measures via coefficients

### 9.3 Design Optimization

Use response surface as surrogate:
```
minimize   Y(X)
subject to constraints on X
```

Benefits:
- Cheap function evaluations (vs. simulation)
- Analytical gradients available
- Global optimization feasible

### 9.4 Model Reduction for Complex Simulations

When expensive simulations (FEA, CFD) exist:
1. Run limited design of experiments
2. Fit mimoSHORSA surrogate
3. Use surrogate for:
   - Optimization
   - Monte Carlo analysis
   - Real-time prediction

### 9.5 Multi-Physics Problems

Natural fit for coupled systems:
- Multiple outputs from single input
- Each output modeled independently
- Maintains physical intuition

Examples:
- Thermal-structural coupling
- Fluid-structure interaction
- Electro-mechanical systems

### 9.6 Practical Considerations

**When mimoSHORSA excels**:
- Smooth, continuous responses
- Moderate dimensions (n < 20)
- Sufficient data (m > 5·nTerm)
- Need for interpretability

**When to use alternatives**:
- Discontinuous responses → Classification methods
- Very high dimensions → Dimension reduction first
- Sparse data → Bayesian approaches
- Black-box OK → Neural networks, tree methods

---

## 10. Mathematical Appendix

### 10.1 Hermite Polynomial Recurrence

Higher-order Hermite polynomials via recurrence:

```
H_{n+1}(z) = 2z H_n(z) - 2n H_{n-1}(z)
```

Starting values:
```
H_0(z) = 1
H_1(z) = 2z
```

### 10.2 Standard Error Derivation

For linear model Y = Bc + ε where ε ~ N(0, σ²I):

```
Var(c) = σ² (B^T B)^(-1)
```

Estimate σ² from residuals:
```
σ̂² = RSS / (m - nTerm)
```

Therefore:
```
SE(cⱼ) = σ̂ √[(B^T B)^(-1)]ⱼⱼ
```

### 10.3 R² Relationship to Correlation

For centered data:
```
R² = ρ²
```

But for general case (with intercept):
```
R² ≠ ρ²
```

R²_adj provides better comparison across models.

---

## 11. References and Further Reading

### Primary Reference
Gavin, H.P. and Yau, S.C., "High order limit state functions in the 
response surface method for structural reliability analysis,"
*Structural Safety*, 2008, Vol. 30, pp. 162-179.

### Theoretical Background

**Response Surface Methods**:
- Box, G.E.P. and Draper, N.R., "Response Surfaces, Mixtures, and Ridge Analyses," 2007
- Myers, R.H. et al., "Response Surface Methodology," 2016

**Hermite Polynomials**:
- Szegö, G., "Orthogonal Polynomials," 1939
- Abramowitz, M. and Stegun, I., "Handbook of Mathematical Functions," 1964

**Uncertainty Quantification**:
- Sudret, B., "Global sensitivity analysis using polynomial chaos expansions," 2008
- Xiu, D. and Karniadakis, G.E., "The Wiener-Askey polynomial chaos," 2002

**Model Selection**:
- Akaike, H., "A new look at the statistical model identification," 1974
- Burnham, K.P. and Anderson, D.R., "Model Selection and Multimodel Inference," 2002

---

## 12. Summary

mimoSHORSA provides a powerful framework for approximating complex input-output relationships through:

1. **Hermite function basis** for numerical stability
2. **Systematic term generation** for comprehensive coverage
3. **Iterative model reduction** for parsimony
4. **Statistical validation** for confidence
5. **Multiple scaling options** for diverse data types

The method excels in applications requiring:
- Explicit functional forms
- Uncertainty quantification  
- Computational efficiency
- Physical interpretability

By combining classical polynomial approximation theory with modern statistical model selection, mimoSHORSA achieves both accuracy and practicality for real-world engineering and scientific problems.

---

*Document prepared October 23, 2025*
*Based on translation and analysis of mimoSHORSA MATLAB implementation*
*Duke University, Department of Civil and Environmental Engineering*
