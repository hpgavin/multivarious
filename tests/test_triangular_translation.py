import numpy as np

from multivarious.distributions.triangular import (
    triangular_pdf, triangular_cdf, triangular_inv
)

def load_matlab_array(file):
    return np.loadtxt(file)

# Load saved data from MATLAB
x_vals = load_matlab_array("x_vals.txt")
pdf_matlab = load_matlab_array("pdf_vals.txt")
cdf_matlab = load_matlab_array("cdf_vals.txt")
inv_input = load_matlab_array("p_vals.txt")
inv_matlab = load_matlab_array("inv_vals.txt")

# Parameters
a, b, c = 1.0, 5.0, 3.0

# Compute with Python
pdf_python = triangular_pdf(x_vals, a, b, c)
cdf_python = triangular_cdf(x_vals, a, b, c)
inv_python = triangular_inv(inv_input, a, b, c)

# Compare and print results
def compare(name, py, matlab):
    match = np.allclose(py, matlab, atol=1e-6)
    print(f"{name} match:", "**success**!" if match else "**fail**!")
    if not match:
        print("First mismatches:")
        print("Python:", py[:5])
        print("MATLAB:", matlab[:5])

compare("PDF", pdf_python, pdf_matlab)
compare("CDF", cdf_python, cdf_matlab)
compare("Inverse CDF", inv_python, inv_matlab)
