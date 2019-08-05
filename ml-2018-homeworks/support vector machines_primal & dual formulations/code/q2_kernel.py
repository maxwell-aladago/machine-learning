import numpy as np

def q2_kernel(x1, x2, mode):
    # Calculates the kernel distance between two given examples.
    #
    # INPUT:
    #  x1: a numpy.ndarray vector of size [n x 1] and type 'float', containing
    #      an n-dimensional input example
    #  x2: a numpy.ndarray vector of size [n x 1] and type 'float', containing
    #      an n-dimensional input example
    #  mode: a 'str' indicating the type of kernel; it can be either 'linear' or 'polynomial'
    #
    # OUTPUT:
    #  d: 'float' value containing the kernel distance (either linear of polynomial)
    #             between the two examples.

    if mode == 'linear':

        # insert your code here
        d = x1.T.dot(x2)

    elif mode == 'polynomial':

        # insert your code here
        d = (x1.T.dot(x2))**3

    else:
        d = []
        print('parameter mode not recognized')

    d = float(d)
    return d