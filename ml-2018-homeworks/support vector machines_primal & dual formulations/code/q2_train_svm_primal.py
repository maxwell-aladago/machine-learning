import numpy as np
from cvxopt import matrix
from cvxopt import solvers

def q2_train_svm_primal(X, Y, C):
    # Trains a linear SVM (primal formulation, non-separable case), given a training set.
    #
    # INPUT:
    #  X      : a numpy.ndarray matrix of size [m x n] and type 'float' where each row
    #           is a n-dimensional input training example
    #  Y      : a numpy.ndarray vector of size [m x 1] and type 'int16', where the
    #           i-th element is the correct label for the i-th input example (either -1 or 1)
    #  C      : 'float', indicating the hyperparameter C to use in the SVM formulation.
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

    quad_diag = np.hstack((np.ones((n, )), np.zeros((m + 1, ))))
    q = np.hstack((np.zeros((n + 1, )), np.ones((m, )))) * C

    P = matrix(np.diag(quad_diag))
    q = matrix(q)

    # G
    g_w = np.hstack((Y * X, Y, np.eye(m)))
    g_slack = np.hstack((np.zeros((m, n + 1)), np.eye(m)))
    G = matrix(-1 * np.vstack((g_w, g_slack)))

    h = matrix(-1 * np.hstack((np.ones((m, )), np.zeros((m,)))))

    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)
    params = np.array(sol['x']).squeeze()
    w = params[0:n]
    b = float(params[n])

    alphas = np.array(sol['z'])
    svs = np.nonzero(alphas[0:m] >= 1e-5)[0]
    return (w, b, svs)
