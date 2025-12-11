# ~/Code/multivarious/examples/verify_path_import.py
# verify that:  
#  (1) the PYTHONPATH has been set
#  (2) multivarious can be imported 

import os, sys

print("hello")
print("verifying that PYTHONPATH has been set ... ")
print("PYTHONPATH env:", os.environ.get("PYTHONPATH"))
print("sys.path (first 5):", sys.path[:5])
print("... and yes, yes it has. Great!  ")

print("verifying that multivarious can be imported ... ")
import multivarious  
print("... and yes, yes it can. Great!  ")

