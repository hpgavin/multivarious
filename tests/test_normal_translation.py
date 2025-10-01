import numpy as np
from multivarious.distributions import (
    normal_pdf,
    normal_cdf,
    normal_inv
)

def load(file):
    return np.loadtxt(file)

# Load MATLAB reference outputs
x_vals = load("x_vals_normal.txt")
pdf_matlab = load("pdf_vals_normal.txt")
cdf_matlab = load("cdf_vals_normal.txt")
p_vals = load("p_vals_normal.txt")
inv_matlab = load("inv_vals_normal.txt")

# Parameters used in MATLAB
mu = 0.0
sigma = 1.5

# Compute Python outputs
pdf_py = normal_pdf(x_vals, mu, sigma)
cdf_py = normal_cdf(x_vals, mu, sigma)
inv_py = normal_inv(p_vals, mu, sigma)

# Compare function
def compare(name, py, matlab):
    match = np.allclose(py, matlab, atol=1e-4)
    print(f"{name} match:", "**success**!" if match else "**fail**")
    if not match:
        print("Python:", py[:5])
        print("MATLAB:", matlab[:5])

compare("PDF", pdf_py, pdf_matlab)
compare("CDF", cdf_py, cdf_matlab)
compare("Inverse CDF", inv_py, inv_matlab)
