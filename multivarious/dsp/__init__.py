"""
multivarious.dsp: a multivarious subpackage for digital signal processing
"""

# multivarious/dsp/__init__.py

from multivarious.dsp.accel2displ import accel2displ
from multivarious.dsp.butter_synth_ss import butter_synth_ss
from multivarious.dsp.cdiff import cdiff
from multivarious.dsp.chirp import chirp
from multivarious.dsp.ftdsp import ftdsp
from multivarious.dsp.taper import taper

__all__ = [ "accel2displ" , "butter_synth_ss" , "cdiff" , "chirp" , "ftdsp' , "taper" ]
