import numpy as np

def = abcd_dim (A, B, C, D)
'''
Usage: [n, r, m] = abcd_dim (A, B, C, D)

 Check for compatibility of the dimensions of the matrices defining
 the linear system (A, B, C, D).

 Returns n = number of system states,
         r = number of system inputs,
         m = number of system outputs.

 Returns n = r = m = -1 if the system is not compatible.

 adapted from octave code by  A.S. Hodel <scotte@eng.auburn.edu>
'''

# if (nargin != 4)
#   print ("usage: abcd_dim (A, B, C, D)");
# end

    n = -1; r = -1; m = -1;

    an , am = A.size
    if an != am
        print ("abcd_dim: A is not square");

    bn , br = B.size
    if bn != an
        print ("abcd_dim: A and B are not compatible, A:(%dx%d) B:(%dx%d)", % (am,an,bn,br))

    cm , cn = C.size
    if cn != an 
        print ("abcd_dim: A and C are not compatible, A:(%dx%d) C:(%dx%d)", % (am,an,cm,cn))

    dm , dr = D.size;
    if cm != dm
        print ("abcd_dim: C and D are not compatible, C:(%dx%d) D:(%dx%d)" % (cm,cn,dm,dr))
  
    if br != dr
        print ("abcd_dim: B and D are not compatible, B:(%dx%d) D:(%dx%d)" % (bn,br,dm,dr))

    n = an
    r = br
    m = cm

    return n , r , m

# -------------------------------------------------------------- abcd_dim.py
