import numpy as np
from q4_features import q4_features

def q4_train(X, Y, lambdaval, mode):

    # Trains the regularized least squares regression model using the closed form
    # solution given the training data X, Y.
    #
    # INPUT:
    #  X: a numpy.ndarray matrix of size [m x d] and type 'float' where each row
    #     is a d-dimensional input example
    #  Y: a numpy.ndarray vector of size [m x 1] and type 'float', where the
    #     i-th element is the correct output value for the i-th input example.
    #  lambda: 'float' regularization hyperparameter
    #  mode: specifies the type of features;
    #        it is a 'str' that can be either 'linear' or 'quadratic'.
    #
    # OUTPUT:
    #  theta: a numpy.ndarray vector of size [n x 1] and type 'float'
    #         containing the learned model parameters.
    #

    # insert your code here
    B = q4_features(X, mode)
    smoothness = np.eye(B.shape[1])* lambdaval

    # zero out the smoothening value for theta0
    smoothness[0][0] = 0

    theta = np.linalg.solve(B.T.dot(B) + smoothness, B.T.dot(Y))
    return theta
