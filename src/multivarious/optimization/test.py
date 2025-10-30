import numpy as np
from multivarious.utils.optim_options import optim_options
from multivarious.utils.box_constraint import box_constraint
from multivarious.utils.avg_cov_func import avg_cov_func


from multivarious.optimization.ORSopt import orsopt

# toy constrained quadratic: f = ||x||^2, g1 = x0 + x1 - 0.5  (want g<0)
def toy_func(x, consts):
    f = float(np.dot(x, x))
    g = np.array([x[0] + x[1] - 0.5])
    return f, g

x0 = np.array([0.8, 0.8])
lb = np.array([-1.0, -1.0])
ub = np.array([+1.0, +1.0])
opts = optim_options([1, 1e-4, 1e-4, 1e-4, 400])  # display=1, tighter tol, fewer evals

x_opt, f_opt, g_opt, hist = orsopt(toy_func, x0, lb, ub, opts)
print("x_opt:", x_opt, "f_opt:", f_opt, "g_max:", g_opt.max())
