# format_bank.py

import numpy as np

def format_bank():
    '''
    format numerical text strings to print two digits after the decimal place
    
    INPUT       DESCRIPTION                                          
    --------    ---------------------------------------------------   
    ----- not yet implemented ------
    mask        C-style fprintf mask string eg, 0.2f , 6.3f , 12.3e
    '''

    # format bank
    np.set_printoptions(formatter={'float_kind': lambda x: f"{x:0.2f}"})

