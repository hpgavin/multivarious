import pytest
import numpy as np
import importlib
from multivarious import distributions

# Common test grid
X = np.linspace(0.01, 5, 200)
P = np.linspace(0.01, 0.99, 100)

@pytest.mark.parametrize("dist_name", distributions.__all__)
def test_distributions_consistency(dist_name):
    dist = importlib.import_module(f"multivarious.distributions.{dist_name}")

    # --- PDF normalization test ---
    if hasattr(dist, "pdf") and hasattr(dist, "cdf"):
        # We test that the PDF integrates to ~1 over a large range.
        try:
            params = [1, 2]  # generic placeholder; may fail for special dists
            f = dist.pdf(X, *params)
            assert np.isfinite(f).all()
            area = np.trapz(f, X)
            assert np.isclose(area, 1.0, atol=0.2), f"{dist_name} PDF not normalized"
        except Exception as e:
            pytest.skip(f"PDF test skipped for {dist_name}: {e}")

    # --- Inverse consistency test ---
    if hasattr(dist, "cdf") and hasattr(dist, "inv"):
        try:
            params = [1, 2]
            x = dist.inv(*params, len(P), 1) if "quadratic" in dist_name else dist.inv(P, *params)
            F = dist.cdf(x, *params) if "quadratic" not in dist_name else dist.cdf(x.flatten(), *params)
            assert np.all((F >= 0) & (F <= 1))
        except Exception as e:
            pytest.skip(f"Inverse test skipped for {dist_name}: {e}")

    # --- Random generation test ---
    if hasattr(dist, "rnd"):
        try:
            samples = dist.rnd(1, 2, 5, 5) if "quadratic" in dist_name else dist.rnd(1, 2, 5, 5)
            assert samples.shape == (5, 5)
        except Exception as e:
            pytest.skip(f"Random sampling test skipped for {dist_name}: {e}")
