import numpy as np
from multivarious.distributions import (
    uniform_pdf,
    uniform_cdf,
    uniform_inv
)

def load(file):
    return np.loadtxt(file)

# Load MATLAB reference outputs
x_vals = load("x_vals_uniform.txt")
pdf_matlab = load("pdf_vals_uniform.txt")
cdf_matlab = load("cdf_vals_uniform.txt")
p_vals = load("p_vals_uniform.txt")
inv_matlab = load("inv_vals_uniform.txt")

# Parameters used in MATLAB: a = 2, b = 8
a, b = 2.0, 8.0

# Compute Python equivalents
pdf_py = uniform_pdf(x_vals, a, b)
cdf_py = uniform_cdf(x_vals, a, b)
inv_py = uniform_inv(p_vals, a, b)

# Compare and print
def compare(name, py, matlab):
    match = np.allclose(py, matlab, atol=1e-6)
    print(f"{name} match:", "**success**!" if match else "**fail**")
    if not match:
        print("Python:", py[:5])
        print("MATLAB:", matlab[:5])

compare("PDF", pdf_py, pdf_matlab)
compare("CDF", cdf_py, cdf_matlab)
compare("Inverse CDF", inv_py, inv_matlab)
