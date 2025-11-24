# multivarious/utils/__init__.py
"""
Utility functions for optimization: box constraints, options, and averaging.
"""

from .opt_options import optim_options
from .box_constraint import box_constraint
from .avg_cov_func import avg_cov_func
from .plot_opt_surface import plot_opt_surface

__all__ = ["optim_options", "box_constraint", "avg_cov_func"]

