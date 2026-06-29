import numpy as np

def correlation_analysis(v, r, g, NSC):
    """
    pg_correl, rg_correl = correlation_analysis(v, r, g, NSC)
    Perform a correlation analysis for the safety analysis of
    a design optimization problem.

    INPUT VARIABLES
    ===============
     v    (NP x NS)  matrix of design variable values
     r    (NR x NS)  matrix of random non-design values
     g    (NC x NS)  matrix of constraint values
     NSC             the first NSC rows of g are safety constraints

    OUTPUT VARIABLES
    ================
     pg_correl   matrix of correlation values between v and g
     rg_correl   matrix of correlation values between r and g
    """

    g = g[:NSC, :]        # keep only the safety constraints

    NP, NS = v.shape
    NR, _  = r.shape
    NC, _  = g.shape

    # ---- which constraints are in violation?
    max_gt = np.max(g, axis=1)                        # max over simulations, per constraint

    constraints_in_violation = np.where(max_gt > 0)[0]
    bad_constraint_values    = max_gt[constraints_in_violation]

    print('--------------------------')
    print('Constraints In Violation :')
    print('--------------------------')
    for i, ci in enumerate(constraints_in_violation):
        print(f' max[ g({ci+1:3d}) ] = {bad_constraint_values[i]:f}')   # 1-based label

    # ---- full correlation matrix of [v ; r ; g] stacked row-wise
    # np.corrcoef expects (variables x observations), i.e. each row is one variable
    combined     = np.vstack([v, r, g])               # (NP+NR+NC) x NS
    correlations = np.corrcoef(combined)              # (NP+NR+NC) x (NP+NR+NC)

    # correlations is symmetric; diagonal entries are 1.0
    # rows/cols  0      : NP          → design variable uncertainty   v
    # rows/cols  NP     : NP+NR       → random non-design uncertainty r
    # rows/cols  NP+NR  : NP+NR+NC   → constraint uncertainty        g

    pg_correl = correlations[     0:NP    ,  NP+NR:NP+NR+NC]   # v vs g
    rg_correl = correlations[    NP:NP+NR ,  NP+NR:NP+NR+NC]   # r vs g

    # retain only the cross-correlations between (v & r) and g
    correlations = correlations[0:NP+NR,  NP+NR:NP+NR+NC]      # (NP+NR) x NC

    # now:
    # rows  0    : NP    of correlations → design variable uncertainty  v
    # rows  NP   : NP+NR of correlations → random non-design uncertainty r
    # cols  0    : NC    of correlations → constraint uncertainty        g

    significant_correlation = 0.8

    # ---- significant positive correlations
    ii_pos, jj_pos = np.where(correlations > significant_correlation)
    order_pos      = np.argsort(ii_pos)

    print('-----------------------')
    print('Positive Correlations :')
    print('-----------------------')
    for k in order_pos:
        val = correlations[ii_pos[k], jj_pos[k]]
        if ii_pos[k] < NP:
            print(f' correlation between v({ii_pos[k]+1:3d}) and g({jj_pos[k]+1:3d}) is  {val:f}')
        else:
            print(f' correlation between r({ii_pos[k]-NP+1:3d}) and g({jj_pos[k]+1:3d}) is  {val:f}')

    # ---- significant negative correlations
    ii_neg, jj_neg = np.where(correlations < -significant_correlation)
    order_neg      = np.argsort(ii_neg)

    print('-----------------------')
    print('Negative Correlations :')
    print('-----------------------')
    for k in order_neg:
        val = correlations[ii_neg[k], jj_neg[k]]
        if ii_neg[k] < NP:
            print(f' correlation between v({ii_neg[k]+1:3d}) and g({jj_neg[k]+1:3d}) is  {val:f}')
        else:
            print(f' correlation between r({ii_neg[k]-NP+1:3d}) and g({jj_neg[k]+1:3d}) is  {val:f}')

    return pg_correl, rg_correl

# ------------------------------------------------- correlation_analysis
