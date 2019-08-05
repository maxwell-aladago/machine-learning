import numpy as np

def q2_error(Y, pred_Y):
    # Calculates the misclassification rate by comparing the predicted labels pred_Y to
    # the true labels Y

    # INPUT
    #  Y  : a numpy.ndarray vector of size [m x 1] and type 'float', where the
    #       i-th element is the correct labelfor the i-th input example
    #  pred_Y  : a numpy.ndarray vector of size [m x 1] and type 'float', where the
    #            i-th element is the predicted labelfor the i-th input example
    # OUTPUT
    #  error : float, misclassification rate, i.e. the number of
    #          examples misclassified over the total number of examples

    error = len(Y[Y != pred_Y])/len(Y)
    return error
