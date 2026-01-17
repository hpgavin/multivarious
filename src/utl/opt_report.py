import numpy as np
import time 
from datetime import datetime, timedelta


def opt_report(v_init, v_opt, f_opt, g_opt, v_lb, v_ub, tol_v, tol_f, tol_g, start_time, function_evals, max_evals, feasible, converged, stalled ):
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
    if feasible:
        print('\n')
        print(f' * Woo Hoo! Feasible solution found in {function_evals} function evaluations!')
        print(' *          ... and that is all we are asking for.')
    if converged:
        print(f'\n * Woo-Hoo! Convergence in variables and objective in {function_evals} function evaluations!')
        if np.max(g_opt) < tol_g:
            print(' * Woo-Hoo! Converged solution is feasible!')
        else:
            print(' * Boo-Hoo! Converged solution is NOT feasible!')
            print(' *   ... Increase options[5] (penalty) and try, try again ...')
    else:
        print('\n * Boo-Hoo! Solution NOT converged!')

    if stalled:
        print(f' * Hmmm ... no improvement in the last {0.2*max_evals:.0f} function evaluations ...')
        print(' *      ... Increase tol_v (options[1]), tol_f (options[2]) or max_evals (options[4]) ...')
        print(' *      ... and try try again.')
    
    # check if the maximum function evaluation limit was exceeded
    if function_evals >= max_evals:
        print(f" * Enough! max evaluations ({max_evals}) exceeded.")
        print(" *   ... Increase tol_v (options[1]) or tol_f (options[2]) "
                  "or max_evals (options[4]) and try again")

    dur = time.time() - start_time
    print(f' * objective = {f_opt:11.3e} ')
    print(' * --------------------------------------------------------------')
    print(' *             v_init         v_lb     <     v_opt    <     v_ub')
    print(' * --------------------------------------------------------------')
    
    for i in range(n):
        eqlb = ' '
        equb = ' '
        if v_opt[i] < v_lb[i] + tol_g + 100*np.finfo(float).eps:
            eqlb = '='
        elif v_opt[i] > v_ub[i] - tol_g - 100*np.finfo(float).eps:
            equb = '='

        print(f' * v[{i:3d}] {v_init[i]:11.4f}    {v_lb[i]:11.4f} {eqlb} '
              f' {v_opt[i]:12.5f} {equb} {v_ub[i]:11.4f}')

    print(' * Constraints :')
    for j in range(m):
        binding = ' '
        if g_opt[j] > -tol_g:
            binding = ' ** binding ** '
        if g_opt[j] > tol_g:
            binding = ' ** not ok  ** '
        print(f' *     g({j:3d}) = {g_opt[j]:12.5f}  {binding}')
    print(' *\n * --------------------------------------------------------------')

    elapsed = time.time() - start_time
    completion_time = datetime.now().strftime('%H:%M:%S')
    elapsed_str = str(timedelta(seconds=int(elapsed)))
    print(f' * Completion  : {completion_time} ({elapsed_str}) ({dur:.2f} s)\n')
