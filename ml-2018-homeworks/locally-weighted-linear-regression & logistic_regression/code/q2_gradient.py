import numpy as np
from q2_predict import q2_predict

def q2_gradient(Xtrain, Ytrain, theta):
    # Compute the gradient of the log likelihood at theta

    # INPUT
    #  Xtrain  : a numpy.ndarray matrix of size [m x n] and type 'float' where each row
    #            is a n-dimensional input example (assume it already contains the constant feature set to 1)
    #  Ytrain  : a numpy.ndarray vector of size [m x 1] and type 'float', where the
    #            i-th element is the correct labelfor the i-th input example.
    #  theta   : a numpy.ndarray vector of size [n x 1] and type 'float'  containing
    #            the current model parameters
    # OUTPUT
    #  grad    : a numpy.ndarray vector of size [n x 1] and type 'float'  containing
    #            the gradient of the log likelihood at theta

    diff = Ytrain - q2_predict(Xtrain, theta)[1]
    grad = diff.dot(Xtrain)
    return grad
