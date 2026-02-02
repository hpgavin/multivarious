# Multi-Input Multi-Output Response Surface - Theory

## Table of Contents

1. Introduction and Motivation
2. Mathematical Foundation
3. Hermite Function or Legendre Polynomial Bases
4. Model Construction Process
5. Statistical Evaluation MetricsScaling and PreprocessingScaling and PreprocessingScaling and Preprocessing
6. Computational Considerations
7. Applications and Use Cases
8. Mathematical Appendix
9. References
10. Summary

---

## 1. Introduction and Motivation

### 1.1 The Response Surface Problem

In many engineering and scientific applications, we have:

- **Input variables** $\mathbf{X} = [ X_1, X_2, \cdots , X_n] \in {\mathbb R}^n$ 
- **Output variables**  $\mathbf{Y} = [Y_1, Y_2, \cdots, Y_m] \in {\mathbb R}^m$ 
- A complex, possibly nonlinear relationship  $Y = f (X)$ 

The goal is to to make use a sample of $N$ measured observations of each input value and each corresponding output value in order to approximate this relationship with a computationally efficient model that:

1. Captures nonlinear behavior
2. Handles high-dimensional inputs  ($n > 20$)
3. Provides uncertainty quantification
4. Remains computationally tractable 

### 1.2 Why Response Surfaces?

Traditional approaches face challenges:

- **Direct simulation**: Computationally expensive (finite element, CFD, etc.)
- **Low-order polynomials**: Inadequate for complex nonlinear behavior
- **Neural networks**: Black-box nature, difficult uncertainty quantification
- **Kriging/Gaussian processes**: Computational scaling issues for large datasets

**Response surface methods** provide a middle ground: an explicit polynomial models with:

- Analytical derivatives
- Uncertainty quantification
- Interpretable coefficients
- Efficient evaluation

---

## 2. Mathematical Foundation

### 2.1 General Polynomial Representation

**mimo_rs** approximates the input-output relationship for each output variable $y_i$ as: 

$\displaystyle \hat y_i(\mathbf{x}) = \sum_{k=0}^{p-1} c_{i,k} \ \prod_{j=1}^r \psi_{O_{k,j}}(z_j(\mathbf{x}))$

Where:

- $z_j(\mathbf{x})$ : the $j$-component of $\mathbf{Z} = [z_1, \cdots, z_r] \in \mathbb{R}^r$ which is a decorrelated sample of $\mathbf{X}$, $\mathbf{x} = (x_1, \cdots , x_n)$ . Decorrelation is described below.  
- $r$: the rank of $\mathbf{X}$ and the dimension of $\mathbf{z}$, depending on the selected standardization method
- $c_{i,k}$ : the model coefficients, which minimize the $\chi^2$ criterion with $L_1$ (LASSO) regularization
- $\Pi \psi_O$: a multivariate polynomial basis function of order $O$, which is a unique product of power polynomial functions of each standardized input variable $(Z_1, \cdots , Z_r)$ with corresponding orders $O_{k,:} = (O_{k,1}, \cdots , O_{k,r})$
  - $p$: Total number of unique terms in the expansion

### 2.2 The Structure of the Polynomial-Products

The $k$-th term in the sum above (each column in the model basis) isetsets a product of polynomials $\psi_O(z)$ of order $O$
$\displaystyle \prod_{j=1}^r \psi_{O_{k,j}}( z_j( \mathbf{x} ) )$

where the $j$-th factor in the polynomial product is a function of the $j$-th standardized input variable $z_j(\mathbf{x})$, and has an order $O_{k,j}$ 

As an example, for a quadratic model in three input variables, $(X_1, X_2, X_3)$, in which $\bf X$ has full rank ($r=n=3$), the model  would have ten terms $(k=0, \cdots , 9)$ with polynomial orders given in the $(10 \times 3)$ *order matrix*, $O$ :

| $k$ | $O_{k,1}$ | $O_{k,2}$ | $O_{k,3}$ |
| --- | --------- | --------- | --------- |
| 0   | 0         | 0         | 0         |
| 1   | 1         | 0         | 0         |
| 2   | 0         | 1         | 0         |
| 3   | 0         | 0         | 1         |
| 4   | 1         | 1         | 0         |
| 5   | 1         | 0         | 1         |
| 6   | 0         | 1         | 1         |
| 7   | 2         | 0         | 0         |
| 8   | 0         | 2         | 0         |
| 9   | 0         | 0         | 2         |

so that the polynomial-product expansion is: 

$\hat y_i(\mathbf{x}) = c_{i,0} \ \phi_0(z_1) \phi_0(z_2) \phi_0(z_3) +  c_{i,1} \ \phi_1(z_1) \phi_0(z_2) \phi_0(z_3) +  c_{i,2} \ \phi_0(z_1) \phi_1(z_2) \phi_0(z_3) + \cdots + c_{i,9} \ \phi_0(z_1) \phi_0(z_2) \phi_2(z_3)$ 

---

## 3. Hermite Function Basis

### 3.1 Why Hermite Functions?

Standard power polynomials $(1, x, x^2, x^3, \cdots )$ have problems:

- **Numerical instability** for high orders
- **Poor conditioning** of basis matrices
- **Lack of orthogonality**

Hermite functions provide:

- **Orthogonality** with respect to a Gaussian weight
- **Numerical stability** for higher orders
- **Natural scaling** for standardized variables

### 3.2 Hermite Function Definition

The Hermite  *functions* are defined as: 
$\displaystyle \psi(z) = \frac{1}{\sqrt{2^n \ n! \ \sqrt{\pi} } } \ H_O(z) \ \exp\left(\frac{-z^2}{2} \right)  $

* $H_O(z)$ is the Hermite *polynomial* of order $O$
- $\exp(-z^2/2)$ is Gaussian weight function
- Normalization ensures orthonormality (for a single input variable ($n=1$))

### 3.3 Low-Order Hermite Functions

```
ψ₀(z) = π^(-1/4) exp(-z²/2)

ψ₁(z) = √2 π^(-1/4) z exp(-z²/2)

ψ₂(z) = (1/√2) π^(-1/4) (2z² - 1) exp(-z²/2)

ψ₃(z) = (1/√3) π^(-1/4) (2z³ - 3z) exp(-z²/2)

ψ₄(z) = (1/(2√6)) π^(-1/4) (4z⁴ - 12z² + 3) exp(-z²/2)

ψ₅(z) = (1/(2√15)) π^(-1/4) (4z⁵ - 20z³ + 15z) exp(-z²/2)
```

### 3.4 Orthogonality Property of Hermite functions

Hermite functions are orthogonal with respect to a unit weight over the range of all real values.  

$\displaystyle \int_{-\infty}^\infty \psi_m(z) \psi_n(z) \ dz = \delta_{m.n}$ over the range of all real values

This orthonormality leads to diagonalization of the basis in fitting in one dimension with uniformly-spaced independnet variabes.   And it supports numerical stability for fits in higher dimensions.  

---

## 4. Model Construction Process

### Synopsis

**mimo_rs** follows a three-stage process:

#### Stage 1: Scaling and Preprocessing

- data are optionally log-transformed, scaled, standardized or decorrelated
- input data and output data are scaled and decorrelated separately
- columns of data with one or more outliers are removed 

#### Stage 2: Build the order matrix

- Generate all possible term combinations
- Filter based on total order constraint and uniqueness constraint
- Create order matrix specifying the orders in the polynomial products within each term

#### Stage 3: Model Fitting and Reduction

- Randomly split the data sets into a training set and a testing set.  
- Fit the model using least squares with $L_1$ regularization and a specified level of the regularization penalty factor, $\alpha$ .
- Formulate the $L_1$ regularized problem as a KKT matrix equation with $2q$ inequality constraint.
- Solve the $L_1$ problem to minimize the quadratic objective subject to $2q$ equality constraints, and therby set a significant number of coefficients $c_{i,k}$ to (nearly) zero.  

### 4.1 Scaling and Preprocessing

#### Why Scale?

Raw data problems:

- **Different units**: Variables can be on vastly different numerical scales
- **Numerical instability**: Extreme values cause conditioning issues
- **Poor Hermite basis fit**: Hermite functions are intended for $\sim \cal N(0,1)$ data

#### Scaling Options

##### Option 0: No Scaling

Does nothing.  Use when data is already normalized.

$\mathbf{Z} = \mathbf{X}$

##### Option 1: Standardization

Centers the data and scales the data to unit variance.

$\mathbf{Z} = (\mathbf{X} - {\sf avg}(\mathbf{X}) ) \ / \ {\sf sdv}(\mathbf{X})$

##### Option 2: Linear Decorrelation (whitening)

Removes linear correlations between variables.

$\mathbf{Z} = \mathbf{T}^+ (\mathbf{X} - {\sf avg}(\mathbf{X}))$

where $\mathbf{T}^+$ is the psudo inverse of the model correlation matrix $\bf T$,  which relates an i.i.d. sample,  $\mathbf{Z} \in \mathbb{R}^{(r \times N)}$, $\mathbf{Z} \sim \cal{N}(0,1)$ to $\mathbf{X}$ . 

 $\mathbf{X} = \mathbf{TZ} + {\sf avg}(\mathbf{X})
  $.   

$\mathbf{T}
$  is a full-rank spectral factor of the data covariance $\mathbf{C}(\mathbf{X}) = \mathbf{X} \mathbf{X}^{\sf T}/N$ which has an eigenvalue decomposition $\mathbf{C}(\mathbf{X}) = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^{\sf T}$. It is constructed from the $r$ "non-zero" eigenvalues $\mathbf{\Lambda}_r$ of $\mathbf{C}(\mathbf{X})$ and their corresponding eigenvectors $\mathbf{V}_r$.  

$\min({\sf diag}(\mathbf{\Lambda}_r)) > \epsilon \max({\sf diag}(\mathbf{\Lambda}_r))$

$\mathbf{T} = \mathbf{V}_r \mathbf{\Lambda}_r^{1/2} , (\mathbf{T} \in \mathbb{R}^{n \times r})$.

Eliminating infinitessimal  eigenvalues from the eigenvalue decompostion of the covariance improves numerical conditioning.   The (rectangular) psuedo-inverse of $\mathbf{T}$ is 

$\mathbf{T}^+ = \mathbf{\Lambda}_r^{-1/2} \mathbf{V}_r^{\sf T}$ , $\mathbf{T}^+ \in \mathbb{R}^{(r \times n)}$ .

##### Option 3: Log-Standardization

Positive-valued data with multiplicative structure or log-normal distributions may be log-transformed before standardization or decorrelation.   

$\mathbf{Z} = (\log_{10}(\mathbf{X}) - {\sf avg}(\log_{10}(\mathbf{X})))  \ / \ {\sf sdv}(\log_{10}(\mathbf{X}))  $

##### Option 4: Log Decorrelation

Combines log transformation and linear decorrelation.

$\mathbf{Z} = \mathbf{T}^+ (\log_{10}(\mathbf{X}) - {\sf avg}(\log_{10}(\mathbf{X})))$

where here $\mathbf{T}^+$ is the psudo inverse of the model correlation matrix for log-transformed data $\mathbf{T}$ , which relates an i.i.d sample $\mathbf{Z} \in \mathbb{R}^{(r \times N)}$, $\mathbf{Z} \sim \cal{N}(0,1)$ to $\log_{10}(\mathbf{X})$

$\log_{10}(\mathbf{X}) = \mathbf{TZ} + {\sf avg}(\log_{10}(\mathbf{X}))$.  

$\bf T$ is a  full-rank spectral factor of the data covariance of log-transfomred data $\mathbf{C}({\log_{10}(\mathbf{X}))}$, which has an eigen decomposition $\mathbf{C}({\log_{10}(\mathbf{X}))} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^{\sf T}$.  It is constructed from the $r$ "non-zero" eigenvalues $\mathbf{\Lambda}_r$ of $\mathbf{C}(\mathbf{X})$ and their corresponding eigenvectors $\mathbf{V}_r$.

$\min({\sf diag}(\mathbf{\Lambda}_r)) > \epsilon \max({\sf diag}(\mathbf{\Lambda}_r))$

 $\mathbf{T} = \mathbf{V}_r \mathbf{\Lambda}_r^{1/2}$. 

The (rectangular) psuedo-inverse of $\mathbf{T}$  is

 $\mathbf{T}^+ = \mathbf{\Lambda}_r^{-1/2} \mathbf{V}_r^{\sf T}$ ,   ($\mathbf{T}^+ \in \mathbb{R}^{r \times n}$) .  

#### Outlier Removal using Chauvenet's criterion

After scaling, **mimo_rs** removes data outliers in which a standardized data value $z$  exceeds criteria limits.  
$|z| > 0.8 + 0.4 \log(N)$ 

Here, $0.8 + 0.4 \log(N)$  is a a simple approximiation to Chauvent's criterion. 

Rationale:

- Hermite functions designed for $\mathbf{Z} \sim \cal N(0,1)$
- Extreme values degrade approximation
- Removes potential data errors

### 4.2 Order Matrix  Algorithm

The algorithm generates the unique rows of the order matrix $O$ for a model order of $\hat O$ such that 
$0 \leq O_{k,j} \leq \hat O$  
and 
$\displaystyle \sum_{j=0}^r O_{k,j} \leq \hat O$

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

mimo_rs### 4.3 Construction of the Matrix of model basis vectors

The model basis $\bf B$ has structure:

$$
\displaystyle \mathbf{B} = \begin{bmatrix} 
\prod_{j=1}^r \psi_{O_{0,j}}(z_j(\mathbf{x}_1)) & \ldots & \prod_{j=1}^r \psi_{O_{p-1,j}}(z_j(\mathbf{x}_1)) \\ 
\prod_{j=1}^r \psi_{O_{0,j}}(z_j(\mathbf{x}_2)) & \ldots & \prod_{j=1}^r \psi_{O_{p-1,j}}(z_j(\mathbf{x}_2)) \\ 
\vdots & \cdots & \vdots \\ 
\prod_{j=1}^r \psi_{O_{0,j}}(z_j(\mathbf{x}_N)) & \ldots & \prod_{j=1}^r \psi_{O_{p-1,j}}(z_j(\mathbf{x}_N)) 
\end{bmatrix}
$$

Where:

- Rows correspond to data points $(x_1 , \cdots , x_N)$ 
- Columns correspond to terms with index $k$ and coefficient $c_{i,k}$,   $\prod_j \psi_{O_{k,j}}$
- Each entry is the product of basis functions with given orders, evaluated at that data point

---

### 4.3 Least Squares Fitting with $L_1$ Regularization

Given:

- $\bf Z$: Scaled input data (r × N)
- $\bf Y$: Output data (m × N)
- $\bf B$: Basis matrix (N × p)

The coefficients are found by minimizing the L1 regularized objective:
$\displaystyle \min_\mathbf{c} || \mathbf{B} \mathbf{c} - \mathbf{y} ||_2 + \alpha || \mathbf{c}||_1$

combinations inSolution (of the KKT system via SVD for numerical stability):

## 5. Statistical Evaluation Metrics

### 5.1 R-Squared $R^2$

Measures explained variance:

$R^2 = 1 - ( {\sf RSS} / {\sf TSS} )$

Where:

- ${\sf RSS} = \sum ( y_i - \hat y_i )^2$  (residual sum of squares) 
- ${\sf TSS} = \sum ( y_i - {\sf avg}(\mathbf{y}) )^2$  (total sum of squares) 

Interpretation:

- $R^2 = 1$: Perfect fit
- $R^2 = 0$: Model no better than mean
- $R^2 < 0$: Model worse than mean (on test data)

### 5.2 Adjusted R-Squared $R^2_{\sf adj}$

Penalizes model complexity:

$R^2_{\sf adj} = ((N-1) R^2 - p) ) / (N - p)  )$

where $p$ is the number of coefficients in the model and $N$ is the sample size of measured observations.  

  Why adjust?

- Raw $R^2$ always increases with more terms
- $R^2_{\sf adj}$  accounts for degrees of freedom
- Prevents overfitting through complexity penalty

### 5.3 Model-Data Correlation (ρ)

Pearson correlation between predictions and observations:

$\rho = C_{Y, \hat{Y}} / \left( \sqrt{C_{{Y}, {Y}}} \sqrt{C_{\hat {Y}, \hat {Y}}} \right)$

Advantages over R²:

- Scale-invariant
- More interpretable for practitioners
- Robust to offset errors

### 5.4 Coefficient of Variation (COV) of each coefficient

For each coefficient:

${\sf COV}(c_{i,k}) = {\sf ASE}(c_{i,k}) / |\hat c_{i,k}|$ 

Where

${\sf ASE}(c_{i,k})$ is the asymptotic standard error of the coefficient 

Interpretation:

- COV < 0.10: Well-determined coefficient
- 0.10 < COV < 0.30: Moderate uncertainty
- COV > 0.30: Highly uncertain, candidate for removal

### 5.5 Condition Number

Measures numerical stability:
$\kappa(B) = || \mathbf{B} || \cdot || \mathbf{B}^{-1} ||$

Interpretation:

- $\kappa < 100$: Well-conditioned
- $100 < \kappa < 1000$: Moderate conditioning
- $\kappa > 1000$: Ill-conditioned, numerical issues possible

---

### 5.6 Akaike Infomation Criterion (AIC)

${\sf AIC} = \log( 2 \pi \cdot p \sigma^2_{\sf r} ) + p \sigma_{\sf r}^2 + 2 N$

where $p$ is the number of coefficients in the model and  $\sigma^2_{\sf r}$ is the variance of the residuals

Measures over-fitting.   

Interpretation:

- models with an AIC that increases with an increase in N are usually conisdered to be over-fit. 
- models with an AIC that decreases with an increase in N are are under-fit. 

---

## 6. Computational Considerations

### 6.1 Computational Complexity

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

### 6.2 Conditioning and Stability

**Sources of ill-conditioning**:

1. Highly correlated input variables
2. Insufficient data (m < nTerm)
3. Extreme polynomial orders

**Mitigation strategies**:

1. **Hermite basis**: Better conditioned than power basis
2. **Decorrelation** (scaling option 2 or 4)
3. **SVD-based solve**: Handles near-singular systems
4. **Model reduction**: Removes problematic terms

### 6.3 Train-Test Split

Critical for validation:

- **Training set**: Used for coefficient estimation
- **Test set**: Used for performance evaluation

Typical split: 50-80% training, 20-50% testing

**Why separate?**

- Training metrics overestimate performance
- Test metrics indicate generalization
- Prevents overfitting bias

### 6.4 Memory Requirements

**Storage needs**:

- **B matrix**: O(m·nTerm) - largest structure
- **Data**: O(m·(n+p)) 
- **Coefficients**: O(nTerm·p)

For large problems (m > 10⁶), consider:

- Batch processing
- Out-of-core algorithms
- Reduced precision storage

---

## 7. Applications and Use Cases

### 7.1 Structural Reliability Analysis

**Original motivation** (Gavin & Yau, 2005):

- Approximate limit state functions
- Compute failure probabilities
- Sensitivity analysis

Advantages:

- Explicit failure surface
- Analytical gradients for importance sampling
- Handles high-dimensional random variables

### 7.2 Uncertainty Quantification

Applications:

- **Forward propagation**: Input uncertainty → output uncertainty
- **Sensitivity analysis**: Which inputs matter most?
- **Reliability**: Probability of exceeding thresholds

mimo_rs provides:

- Coefficient uncertainties (COV)
- Analytical variance propagation
- Importance measures via coefficients

### 7.3 Design Optimization

Use response surface as surrogate:

```
minimize   Y(X)
subject to constraints on X
```

Benefits:

- Cheap function evaluations (vs. simulation)
- Analytical gradients available
- Global optimization feasible

### 7.4 Model Reduction for Complex Simulations

When expensive simulations (FEA, CFD) exist:

1. Run limited design of experiments
2. Fit mimo_rs surrogate
3. Use surrogate for:
   - Optimization
   - Monte Carlo analysis
   - Real-time prediction

### 7.5 Multi-Physics Problems

Natural fit for coupled systems:

- Multiple outputs from single input
- Each output modeled independently
- Maintains physical intuition

Examples:

- Thermal-structural coupling
- Fluid-structure interaction
- Electro-mechanical systems

### 7.6 Practical Considerations

**When mimo_rs excels**:

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

## 8. Appendix

### 8.1 Hermite Polynomial Recurrence

Higher-order Hermite polynomials via recurrence:

```
H_{n+1}(z) = 2z H_n(z) - 2n H_{n-1}(z)
```

Starting values:

```
H_0(z) = 1
H_1(z) = 2z
```

### 8.2 Standard Error Derivation

For linear model Y = Bc + ε where ε ~ N(0, σ²I):

${\sf VAR}(\mathbf{c}) = \sigma^2 (\mathbf{B}^T \mathbf{B})^{-1}$

Estimate $\sigma^2$ from residuals:

$\sigma^2 = {\sf RSS} / ( {\sf length}(\mathbf{X}) - {\sf length}(\mathbf{c})$

Therefore:

${\sf ASE}(c_{i,k}) = \sigma^2 \sqrt{ [ [ \mathbf{B}^{\sf T} \mathbf{B} ]^{-1} ]_{k,k} }$

### 8.3 $R^2$ Relationship to Correlation

For centered data:

$R^2 = \rho^2$

But for general case (with intercept):

$R^2 \neq \rho^2$

 $R^2_{\sf adj}$ provides better comparison across models.

---

## 9. References and Further Reading

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

## 10. Summary

mimo_rs provides a framework for approximating complex input-output relationships through:

1. **Hermite or Legendre function basis** for numerical stability

2. **Systematic term generation** for comprehensive coverage

3. **L_1 regularization**
   
   min ||B·c - Y|_2 + a ||c||_1 

4. **Statistical validation** for confidence*L_1 re**gularization

5. **Multiple scaling options** for diverse data types

The method excels in applications requiring:

- Explicit functional forms
- Uncertainty quantification  
- Computational efficiency
- Physical interpretability

---
