import numpy as np


def q1_dist2(X1, X2):
    # Calculates the *squared* Euclidean distance between two sets of points.
    # You should be able to efficiently implement this function *without* using for-loops.
    #
    # INPUT:
    # X1 : a numpy.ndarray matrix of size [m1 x n] and type 'float', where each row
    #      is an n-dimensional input example
    # X2 : a numpy.ndarray matrix of size [m2 x n] and type 'float', where each row
    #      is an n-dimensional input example
    #
    # OUTPUT:
    #  D: a numpy.ndarray matrix of size [m1 x m2] and type 'float',
    #     where the element D[i,j] represent the squared Euclidean distance between
    #     the i-th example of X1, and the j-th example of X2.

    # insert your code here
    D = np.square(X1[:, np.newaxis] - X2).sum(axis=2)
    return D
