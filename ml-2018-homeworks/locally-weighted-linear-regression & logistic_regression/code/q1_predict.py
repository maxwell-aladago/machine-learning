import numpy as np
from q1_train import q1_train

def q1_predict(X, Y, xtest, tau):
    # Predicts the output value of the input example xtest, given the training
    # set X, Y, parameter tau, and the test example

    #
    # INPUT
    #  X  : a numpy.ndarray matrix of size [m x n] and type 'float' where each row
    #       is a n-dimensional input training example
    #  Y  : a numpy.ndarray vector of size [m x 1] and type 'float', where the
    #       i-th element is the correct labelfor the i-th input training example
    #  xtest : a numpy.ndarray vector of size [d x 1] and type 'float',
    #          it  is the input vector of a *single* test example
    #  tau   : float, a *single* value for the regularization hyperparameter
    #
    # OUTPUT
    #  pred_y: float, the predicted output value.
    #

    theta = q1_train(X, Y, xtest, tau)
    xtest = np.hstack((1, xtest))
    pred_y = theta.dot(xtest)
    return pred_y
