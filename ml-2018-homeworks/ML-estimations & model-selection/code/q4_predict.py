import numpy as np
from q4_features import q4_features

def q4_predict(theta, X, mode):
    # Predicts the output values of the input examples X, given the learned parameter vector theta.
    #
    # INPUT
    #  theta: a numpy.ndarray vector of size [n x 1] and type 'float'
    #         containing the learned model parameters.
    #  X: a numpy.ndarray matrix of size [m x d] and type 'float' where each row
    #     is a d-dimensional input example
    #  mode: specifies the type of features;
    #        it is a 'str' that can be either 'linear' or 'quadratic'.
    #
    # OUTPUT
    #  pred_Y: a numpy.ndarray vector of size [m x 1] and type 'float' containing
    #          the m predicted values
    #

    # insert your code here
    B = q4_features(X, mode)
    pred_Y =  B.dot(theta) 
    return pred_Y
