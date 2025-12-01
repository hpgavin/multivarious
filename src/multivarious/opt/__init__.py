# multivarious/opt/__init__.py
# multivarious/opt/__init__.py

"""
Optimization algorithms: 
- ORS  (Optimized Random Search)
- NMS  (Nelder–Mead Simplex)
- SQP  (Sequential Quadratic Programming)
"""

# multivarious/opt/__init__.py
from .nms import nms
from .ors import ors  
from .sqp import sqp

__all__ = ["nms", "ors", "sqp"]



