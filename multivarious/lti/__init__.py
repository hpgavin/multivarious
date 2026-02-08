"""
multivarious.lti: a multivarious subpackage for linear time invariant systems

Each module in this package defines one function whose name matches the
module filename (case-sensitive). Re-export all of them here so users may call:

  import line                                  function call 
  ---------------------------------            -------------------
  import multivarious as m                     m.lti.lsym() 
  from multivarious.lti import lsym            lsym()
  from multivarious.lti import *               lsym()

"""

# multivarious/lti/__init__.py

# Re-export functions from submodules
from .abcd_dim import abcd_dim
from .blk_hankel import blk_hankel
from .blk_toeplitz import blk_toeplitz
from .con2dis import con2dis
from .ctrb import ctrb
from .dis2con import dis2con
from .dliap import dliap
from .dlsym import dlsym
from .damp import damp
from .kalman_dcmp import kalman_dcmp
from .liap import liap
from .lsym import lsym
from .mimo_bode import mimo_bode
from .mimo_tfe import mimo_tfe
from .obsv import obsv
from .pz_plot import pz_plot
from .sys_zero import sys_zero
from .wiener_filter import wiener_filter

__all__ = [
    "abcd_dim",
    "blk_hankel",
    "blk_toeplitz",
    "ctrb"
    "con2dis",
    "dis2con",
    "dliap",
    "dlsym",
    "damp",
    "kalman_dcmp", 
    "liap", 
    "lsym",
    "mimo_bode",
    "mimo_tfe",
    "obsv"
    "pz_plot",
    "sys_zero",
    "wiener_filter", 
]

