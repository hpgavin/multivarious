"""
multivarious.rvs - a multivarious subpackage for random variables

Each module defines a probability distribution that exposes the usual functions:
    pdf(), cdf(), inv(), rnd()

Usage patterns forces module-style calls:

  import line                                  function call 
  ---------------------------------            -------------------
  import multivarious as m                     m.rvs.beta.rnd() 
  from multivarious.rvs import beta            beta.rnd()
  from multivarious.rvs import *               beta.rnd()

Do **not** re-export the individual functions here keep them in their module
so callers must use the module namespace (helpful for clarity and avoiding
name collisions).
"""

# multivarious/rvs/__init__.py

from typing import TYPE_CHECKING

# quantile.py contains a single function, quantile_ci.py
from .quantile_ci import quantile_ci

from . import beta, binomial, chi2, exponential, extreme_value_I, extreme_value_II, gamma, gev, laplace, lognormal, normal, poisson, quadratic, quantile_ci, rayleigh, students_t, triangular, uniform

__all__ = [
    "quantile_ci", 
    "beta",
    "binomial", 
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
        binomial,
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


