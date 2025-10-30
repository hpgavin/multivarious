"""
multivarious.distributions
--------------------------
Collection of probability distributions used in Multivarious.
Each module defines pdf, cdf, inv, and rnd functions.
"""

__all__ = [
    "beta",
    "chi2",
    "exp",
    "ext_I",
    "ext_II",
    "gamma",
    "gev",
    "laplace",
    "lognormal",
    "normal",
    "poisson",
    "quadratic",
    "rayleigh",
    "students_t",
    "triangular",
    "uniform",
]

# Lazy import pattern like SciPy uses
import importlib

def __getattr__(name):
    """Dynamically import distributions on demand."""
    if name in __all__:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
