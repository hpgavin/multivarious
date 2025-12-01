"""
multivarious: a package of multivariable tools for computations in: 

* digital signal processing (dsp),
* linear time invariant systems (lti), 
* ordinary differential equations (ode),
* numerical optimization (opt), 
* random variables (rvs), 

Top-level package. This file exposes the main subpackages as attributes so
users can either call the function via the package directly :

    import multivarious as m

    m.lti.lsym(A, B, C, D, t, u)

... or import the function callable directly from the package-level namespace
    import multivarious as m
    from m.lti import lsym

    y = lsym(A, B, C, D, t, u)

This top-level import will bind the subpackage modules (dsp, lti, ode, opt, rvs, utl)
onto the package object so the `m.<subpkg>.<callable>` style works.
"""
# multivarious/__init.py__

from importlib.metadata import version, PackageNotFoundError

try:
    # When installed normally, this will return the package version from metadata
    __version__ = version("multivarious")
except PackageNotFoundError:
    # Fallback for editable/dev installs or when metadata isn't available
    __version__ = "0.0.0"

# Expose subpackages as attributes for method-style access:
# (these imports are lightweight - they only bind the subpackage packages,
#  not necessarily every function inside them unless those subpackages import
#  their modules eagerly in their own __init__.py)

from . import dsp, lti, ode, opt, rvs, utl  # noqa: F401

__all__ = [" dsp", "lti", "ode", "opt", "rvs", "utl", "__version__" ]
