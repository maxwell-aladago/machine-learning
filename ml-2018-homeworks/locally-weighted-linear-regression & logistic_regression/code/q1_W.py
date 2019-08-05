import numpy as np
def q1_W(X, xtest, tau):
# Constructs the matrix W described in problem 1(a) of the homework
#
# INPUT
#  X     : a numpy.ndarray matrix of size [m x d] and type 'float' where each row 
#          is a d-dimensional input training example
#  xtest : a numpy.ndarray vector of size [d x 1] and type 'float',
#          it  is the input vector of a *single* test example 
#  tau   : float, a *single* value for the regularization hyperparameter
#
# OUTPUT
# W: [m x m] matrix
    l2_norm_squa = np.sum(np.square(X - xtest), axis=1)
    W = np.exp(l2_norm_squa/(-2 * tau * tau))
    total = W.sum()
    W = np.diag(W/total)
    return W
