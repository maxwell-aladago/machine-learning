import numpy as np
from cvxopt import matrix
from cvxopt import solvers

def q2_train_svm_primal_separable(X, Y):
    # Trains a linear SVM (primal formulation, separable case), given the training set.
    #
    # INPUT:
    #  X      : a numpy.ndarray matrix of size [m x n] and type 'float' where each row
    #           is a n-dimensional input training example
    #  Y      : a numpy.ndarray vector of size [m x 1] and type 'int16', where the
    #           i-th element is the correct label for the i-th input example (either -1 or 1)
    #
    # OUTPUT:
    #  w : a numpy.ndarray vector of size [n x 1] and type 'float',
    #      containing the learned model parameters (coefficients of the hyperplane).
    #  b : 'float', the bias of the hyperplane.
    #  svs: a numpy.ndarray vector of size [nsv x 1] and type 'int64',
    #       containing the indices of the nsv training examples that are support vectors;
    #       for example, svs[0] will give the index of the first support vector
    #       and will be an integer number between 0 and m-1.

    # insert your code here
    m, n = X.shape
    Y = Y.reshape(m, -1)
    q = matrix([0.0]*(n + 1))
    P = matrix(np.eye(n + 1))
    P[n, n] = 0
    h = matrix(-np.ones((m, )))
    G = matrix(-1 * np.hstack((Y * X, Y)))

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)
    params = np.array(sol['x']).squeeze()
    alphas = np.array(sol['z'])
    w = params[:-1]

    # last value is the bias
    b = float(params[-1])
    svs = np.nonzero(alphas[0:m] >= 1e-5)[0]
    return (w, b, svs)