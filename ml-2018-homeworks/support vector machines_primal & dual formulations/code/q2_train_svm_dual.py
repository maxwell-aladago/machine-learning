import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from q2_kernel import q2_kernel

def q2_train_svm_dual(X, Y, C, mode):
    # Trains a linear SVM (non-separable dual formulation).
    #
    # INPUT:
    #  X      : a numpy.ndarray matrix of size [m x n] and type 'float' where each row
    #           is a n-dimensional input training example
    #  Y      : a numpy.ndarray vector of size [m x 1] and type 'int16', where the
    #           i-th element is the correct label for the i-th input example (either -1 or 1)
    #  C      : 'float', indicating the hyperparameter C to use in the SVM formulation
    #  mode   : a 'str' indicating the type of kernel; it can be either 'linear' or 'polynomial'
    #
    #
    # OUTPUT:
    #  svs: a numpy.ndarray vector of size [nsv x 1] and type 'int64',
    #       containing the indices of the nsv training examples that are support vectors;
    #       for example, svs[0] will give the index of the first support vector
    #       and will be an integer number between 0 and m-1
    #  alphas : a numpy.ndarray vector of size [nsv x 1] and type 'float',
    #           containing the  the alpha coefficients associated to the support vectors

    # insert your code here
    m, _ = X.shape

    K = np.array([[q2_kernel(X[i], X[j], mode) for i in range(m)] for j in range(m)])
    y_outer = np.outer(Y, Y)

    P = matrix(K * y_outer)
    q = matrix(-1 * np.ones((m, )))

    slack_const_0 = -1 * np.eye(m)
    slack_const_c = np.eye(m)
    G = matrix(np.vstack((slack_const_0, slack_const_c)))
    h = matrix(np.hstack((np.zeros((m,)), np.ones((m,)) * C)))

    # flatten out y
    Y = np.array(Y, dtype='float').reshape(1, m)
    A = matrix(Y)
    b = matrix([0.0])

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)

    alphas = np.array(sol['x']).squeeze()
    svs = np.nonzero(alphas[0:m] >= 1e-5)[0]
    alphas = alphas[svs]
    return (svs, alphas)
