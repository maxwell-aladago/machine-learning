import numpy as np
from q1_features import q1_features
from q1_W import q1_W

def q1_train(X, Y, xtest, tau):
    # Trains the locally weighted linear regression (LWLR) model using the
    # closed form solution given the training data X, Y, the test
    # input vector xtest and the parameter tau.
    #
    # INPUT:
    #  X  : a numpy.ndarray matrix of size [m x n] and type 'float' where each row
    #       is a n-dimensional input training example
    #  Y  : a numpy.ndarray vector of size [m x 1] and type 'float', where the
    #       i-th element is the correct labelfor the i-th input training example
    #  xtest : a numpy.ndarray vector of size [d x 1] and type 'float',
    #          it  is the input vector of a *single* test example
    #  tau   : float, a *single* value for the regularization hyperparameter
    #
    # OUTPUT
    #  theta   : a numpy.ndarray vector of size [n x 1] and type 'float'  containing
    #            the learned model parameters
    #
    B = q1_features(X, 'linear')
    xtest = np.hstack((1, xtest))
    W = q1_W(B, xtest, tau)
    theta = np.linalg.solve(np.linalg.multi_dot([B.T, W, B]), np.linalg.multi_dot([B.T, W, Y]))
    return theta
