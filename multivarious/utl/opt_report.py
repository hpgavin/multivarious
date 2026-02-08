import numpy as np
import time 
from datetime import datetime, timedelta


def opt_report(v_init, v_opt, f_opt, g_opt, v_lb, v_ub, tol_v, tol_f, tol_g, lambda_qp, start_time, function_evals, max_evals, find_feas, feasible, converged, stalled ):
    '''
    print a final report of the result of optimization from ors, nms, sqp

    Parameters
    ----------

    Returns
    -------
       (nothing)
    '''

    n = len(v_opt)
    m = len(g_opt)

    # final report
    if feasible and find_feas:
        print('\n')
        print(f' * Woo Hoo! Feasible solution found in {function_evals} function evaluations!')
        print(' *          ... and that is all we are asking for.')
    if converged:
        print(f'\n * Woo-Hoo! Convergence in variables and objective in {function_evals} function evaluations!')
        if np.max(g_opt) < tol_g:
            print(' * Woo-Hoo! Converged solution is feasible!')
        else:
            print(' * Boo-Hoo! Converged solution is NOT feasible!')
            print(' *   ... Increase opts[5] (penalty) and try, try again ...')
    else:
        print('\n * Boo-Hoo! Solution NOT converged!')

    if stalled:
        print(f' * Hmmm ... no improvement in the last {0.2*max_evals:.0f} function evaluations')
        print(' *      ... Increase tol_v (opts[1]), tol_f (opts[2]) or max_evals (opts[4]) ...')
        print(' *      ... and try try again.')
    
    # check if the maximum function evaluation limit was exceeded
    if function_evals >= max_evals:
        print(f" * Enough! max evaluations ({max_evals}) exceeded.")
        print(" *   ... Increase tol_v (opts[1]) or tol_f (opts[2]) "
                  "or max_evals (opts[4]) and try again")

    dur = time.time() - start_time
    print(" * ----------------------------------------------------------------------------")
    print(" * Variables: ") 
    if np.any(lambda_qp == None):
        print(' *               v_init      v_lb          v_opt          v_ub')
    else:
        print(" *               v_init      v_lb          v_opt          v_ub      lambda")
    print(" * ----------------------------------------------------------------------------")

    for i in range(n):
        eqlb = ' = ' if v_opt[i] < v_lb[i] + tol_g + 1e-6 else ' < '
        equb = ' = ' if v_opt[i] > v_ub[i] - tol_g - 1e-6 else ' < '
        lulb = '   '
        if eqlb == ' = ' and np.all(lambda_qp != None):
            lulb = f'{lambda_qp[m + i]:12.5f}'
        elif equb == ' = ' and np.all(lambda_qp != None):
            lulb = f'{lambda_qp[m + n + i]:12.5f}'
        print(f" *  v[{i+1:3d}] {v_init[i]:11.4f} "
              f"{v_lb[i]:11.4f}{eqlb}{v_opt[i]:12.5f}{equb}{v_ub[i]:11.4f}{lulb}")
    print(" * ----------------------------------------------------------------------------")
    print(f' * Objective:  {f_opt:11.3e} ')
    print(" * ----------------------------------------------------------------------------")
    print(' * Constraints: ')
    for j in range(m):
        binding = ' '
        lulb = ' '
        if g_opt[j] > -tol_g:
            binding = ' ** binding ** '
        if g_opt[j] > tol_g:
            binding = ' ** not ok  ** '
        if np.any(lambda_qp == None): 
            print(f" *  g[{j+1:3d}] = {g_opt[j]:12.5f}   {binding}") 
        else: 
            print(f" *  g[{j+1:3d}] = {g_opt[j]:12.5f}      "
                  f"lambda[{j+1:3d}] = {lambda_qp[j]:12.5f}   {binding}")
    print(" * ----------------------------------------------------------------------------")

    elapsed = time.time() - start_time
    completion_time = datetime.now().strftime('%H:%M:%S')
    elapsed_str = str(timedelta(seconds=int(elapsed)))
    print(f' * Completion  : {completion_time} ({elapsed_str}) ({dur:.2f} s)\n')

