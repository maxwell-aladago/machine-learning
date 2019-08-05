import numpy as np

# Do not modify this function, it has already been coded for you
def q1_error(Y, pred_Y):
    # Calculates the misclassification rate by comparing the predicted labels pred_Y to
    # the true labels Y

    # INPUT
    #  Y         : a numpy.ndarray vector of size [m x 1] and type 'uint8' containing
    #              the true label for each example. Each entry value is  either 0 or 1.
    #  pred_Y    : a numpy.ndarray vector of size [m x 1] and type 'uint8' containing
    #              the predicted label for each example. Each entry value is  either 0 or 1.

    # OUTPUT
    #  error : float, misclassification rate, i.e. the number of
    #          examples misclassified over the total number of examples

    # do not modify the lines below: this function has been already coded for you
    m = Y.size
    error = float(np.sum(Y != pred_Y) / m)

    return error
