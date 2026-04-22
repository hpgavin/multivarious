"""
multivarious.utl: a multivarious subpackage for general utility functions

Each module in this package defines one function whose name matches the
module filename (case-sensitive). Re-export all of them here so users may call:

  import line                                  function call 
  ---------------------------------            -------------------
  import multivarious as m                     m.utl.plot_cvg_hst() 
  from multivarious.utl import plot_cvg_hst    plot_cvg_hst()
  from multivarious.utl import *               plot_cvg_hst()

"""

# multivarious/utl/__init__.py

from .avg_cov_func import avg_cov_func
from .box_constraint import box_constraint
from .correlated_rvs import correlated_rvs
from .correlation_analysis import correlation_analysis
from .format_bank import format_bank
from .format_plot import format_plot
from .opt_options import opt_options
from .opt_report import opt_report
from .plot_ECDF_ci import plot_ECDF_ci
from .plot_cvg_hst import plot_cvg_hst
from .plot_ensemble import plot_ensemble
from .plot_lm import plot_lm
from .plot_opt_surface import plot_opt_surface
from .plot_scatter_dist import plot_scatter_dist
from .plot_spectra import plot_spectra 
from .L1_plots import L1_plots
from .StableNamespace import StableNamespace


__all__ = [ "avg_cov_func", 
            "box_constraint", 
            "correlated_rvs",
            "correlation_analysis",
            "format_bank", 
            "format_plot", 
            "opt_options", 
            "opt_report", 
            "plot_ECDF_ci", 
            "plot_cvg_hst", 
            "plot_ensemble", 
            "plot_lm", 
            "plot_opt_surface", 
            "plot_spectra", 
            "plot_scatter_dist", 
            "L1_plots", 
            "StableNamespace", 
          ]

