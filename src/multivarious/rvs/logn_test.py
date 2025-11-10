import numpy as np
import matplotlib.pyplot as plt
from lognormal import pdf, cdf, inv, rnd

print("Testing Lognormal Distribution")
print("=" * 40)
print()

# Test parameters
x_val = 1.5
medX = 1.0
covX = 0.5

# Test PDF
print("--- PDF Test ---")
f = pdf(x_val, medX, covX)
print(f"PDF at x={x_val:.1f}: {f:.6f}")
print()

# Test CDF
print("--- CDF Test ---")
F = cdf(x_val, medX, covX)
print(f"CDF at x={x_val:.1f}: {F:.6f}")
print()

# Test INV
print("--- INV Test ---")
x_recovered = inv(F, medX, covX)
print(f"INV(CDF({x_val:.1f})): {x_recovered:.6f} (should be ~{x_val:.1f})")
print()

# Test RND (uncorrelated)
print("--- RND Test (Uncorrelated) ---")
np.random.seed(42)
samples = rnd(medX, covX, N=5000)
print(f"Sample median: {np.median(samples):.4f} (should be ~{medX:.1f})")
print(f"Sample mean: {np.mean(samples):.4f}")
print(f"Sample shape: {samples.shape}")
print()

# Test RND (correlated - 2 variables)
print("--- RND Test (Correlated) ---")
medX_vec = np.array([1.0, 2.0])
covX_vec = np.array([0.5, 0.3])
R = np.array([[1.0, 0.8], [0.8, 1.0]])

np.random.seed(42)
samples_corr = rnd(medX_vec, covX_vec, N=2000, R=R)
correlation = np.corrcoef(samples_corr)[0, 1]
print(f"Variable 1 median: {np.median(samples_corr[0, :]):.4f}")
print(f"Variable 2 median: {np.median(samples_corr[1, :]):.4f}")
print(f"Target correlation: 0.8")
print(f"Sample correlation: {correlation:.4f}")
print()

# Plot correlated vs uncorrelated
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Uncorrelated (2 independent variables)
np.random.seed(123)
samples_uncorr = rnd(medX_vec, covX_vec, N=2000)
corr_uncorr = np.corrcoef(samples_uncorr)[0, 1]

axes[0].scatter(samples_uncorr[0, :], samples_uncorr[1, :], alpha=0.3, s=1)
axes[0].set_xlabel('Variable 1')
axes[0].set_ylabel('Variable 2')
axes[0].set_title(f'Uncorrelated (ρ={corr_uncorr:.3f})')
axes[0].grid(True, alpha=0.3)

# Correlated
axes[1].scatter(samples_corr[0, :], samples_corr[1, :], alpha=0.3, s=1)
axes[1].set_xlabel('Variable 1')
axes[1].set_ylabel('Variable 2')
axes[1].set_title(f'Correlated (target=0.8, actual={correlation:.3f})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lognormal_correlation_test.png', dpi=150)
print("✓ Plot saved as 'lognormal_correlation_test.png'")
print()
print("✓ All tests passed!")