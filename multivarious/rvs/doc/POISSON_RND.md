
# A sample of Poisson-distributed  random values


### How to relate this process to standard uniform distribution for correlated Poisson processes?

- **Knuth’s algorithm uses uniform random variables internally:**  
  The product of uniforms until it falls below \(L\) simulates the Poisson count.

- To **generate correlated Poisson processes**, you need correlated uniform random variables as the building block.

- **Approach:**

  1. **Generate correlated uniform random variables:**  
     Use a copula (e.g., Gaussian copula) to generate correlated uniform samples across your processes.

  2. **Apply the multiplicative algorithm per process:**  
     Use the correlated uniforms instead of independent uniforms inside the multiplicative loop.

- **Challenges:**

  - The multiplicative algorithm needs a sequence of independent uniforms per sample; correlation between processes means uniforms at the same iteration across processes should be correlated, but uniforms across iterations should remain independent.

  - This requires generating **correlated uniform vectors per iteration**, respecting the correlation matrix \(R\).

---

### How this works:

- `p` starts as ones and is multiplied by uniform random numbers until it falls below `exp_tT`.
- `X` counts how many uniforms were multiplied before the product dropped below the threshold, effectively sampling from a Poisson distribution with mean \(t / T\).
- `active` is a boolean mask indicating which elements are still in the loop.

---

### To incorporate correlation:

To introduce correlation among the Poisson processes, you want the uniforms generated at each iteration to be correlated **across the \(n\) processes**, but independent **across iterations**.

---

### How to do this:

1. **Generate correlated uniforms for each iteration:**

   - For each iteration, generate one vector of correlated uniforms of length \(n\).

2. **Apply these uniforms in the multiplicative step for all active processes.**

---

### Example sketch integrating correlated uniforms:

```python
from scipy.stats import norm

def correlated_poisson_samples(t, T, N, R, seed=None):
    rng = np.random.default_rng(seed)
    n = len(T)
    exp_tT = np.exp(-t / T)

    p = np.ones((n, N))
    X = np.zeros((n, N), dtype=int)

    # Pre-generate correlated uniforms for all iterations, U 
    _, _, U = correlated_rvs(n,N,R)

    active = p >= exp_tT
    iter_idx = 0
    while np.any(active) and iter_idx < N:
        # Apply correlated uniforms for active elements at iteration iter_idx
        p[active, iter_idx] *= U[active, iter_idx]
        X[active, iter_idx] += 1
        active = p >= exp_tT
        iter_idx += 1

    X = X - 1
    return X
```

---

### Notes:

- This sketch assumes \(R\) is a valid correlation matrix (positive semi-definite).
- The loop uses pre-generated correlated uniforms, one vector per iteration.
- The indexing ensures uniforms are applied only where `active` is True.
- You may need to adjust the shape and indexing depending on your exact use case.

---

You're on the right track with this approach! Generating the correlated uniforms **inside the loop over samples** is indeed a practical and memory-efficient way to produce independent samples of correlated Poisson processes.

---

### Let's clarify and refine your idea:

Your code snippet suggests:

- Loop over each sample `i` (n,) in `N`.
- For each sample, generate a fresh set of correlated uniforms `U` of shape `(n, K)` where `K` is the max iterations.
- Run the multiplicative algorithm for that sample using `U`.
- Store the result in `X[:, i]`.

---

### This approach:

- **Ensures independence across samples** because each sample gets its own fresh set of correlated uniforms.
- **Preserves correlation across processes** within each sample because uniforms at each iteration are correlated via the covariance matrix `R`.
- **Controls memory usage** by only generating `(n, K)` uniforms per sample, not the full `(n, K, M)` 3D array.
- **Is straightforward to implement and understand.**

---

### A few notes and a polished version:

```python
def correlated_poisson_samples(t, T, M, K, R, seed=None):

    t = np.atleast_1d(t).reshape(-1, 1).astype(float)
    T = np.atleast_1d(T).reshape(-1, 1).astype(float)
    n = len(T)
    
    exp_tT = np.exp(-t / T).flatten()  # shape (n,)
    
    X = np.zeros((n, N), dtype=int)
    
    rng = np.random.default_rng(seed)
    
    for i in range(N):
        p = np.ones(n)
        x = np.zeros(n, dtype=int)
        
        # Generate correlated uniforms for this sample
        _, _, U = correlated_rvs(n, K, R)

        active = p >= exp_tT
        iter = 0
        
        while np.any(active) and iter < K:
            p[active] *= U[active, iter_idx]
            x[active] += 1
            active = p >= exp_tT
            iter += 1
        
        X[:, i] = x - 1  # Adjust per Knuth's algorithm

    return X
```

---

### Why this is a good balance:

- **Memory:** Only `(n, K)` uniforms per sample are in memory at once.
- **Computation:** Loop over `M` samples is unavoidable unless you generate huge 3D arrays.
- **Correctness:** Each sample is independent; correlation is preserved across processes per iteration.

---

### When to consider other approaches?

- If `K` is extremely large and performance is critical, you might want to:

  - Use JIT compilation (e.g., with Numba) to speed up the loop.
  - Generate samples in batches.
  - Explore approximate or alternative sampling methods.

---
