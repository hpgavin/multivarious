"""
multivarious.rvs - a multivarious subpackage for random variables

Each distribution is a module that exposes the usual functions:
    pdf(), cdf(), inv(), rnd()

Usage patterns forces module-style calls:

    import multivarious as m
    x = m.rvs.beta.rnd(a, b, p, q, N)

or:

    from multivarious.rvs import beta
    x = beta.rnd(a, b, p, q, N)

Do **not** re-export the individual functions here keep them in their module
so callers must use the module namespace (helpful for clarity and avoiding
name collisions).
"""

# multivarious/rvs/__init__.py

from typing import TYPE_CHECKING

from .plot_CDF_ci import plot_CDF_ci
from .quantile_ci import quantile_ci

__all__ = [
    "plot_CDF_ci", 
    "quantile_ci", 
    "beta",
    "chi2",
    "exponential",
    "extreme_value_I",
    "extreme_value_II",
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

# For type checkers / IDEs: import module names only during static analysis.
# This helps editors show completions like `m.rvs.beta` without causing runtime imports.
if TYPE_CHECKING: # pragma: no cover
    from . import (
        beta,
        chi2,
        exponential,
        extreme_value_I,
        extreme_value_II,
        gamma,
        gev,
        laplace,
        lognormal,
        normal,
        poisson,
        quadratic,
        rayleigh,
        students_t,
        triangular,
        uniform,
    )


