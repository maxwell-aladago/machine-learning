import numpy as np
from q2_loglik import q2_loglik
from q2_gradient import q2_gradient

def q2_train_line_search(Xtrain, Ytrain, theta_init, alpha0, tol):
    # Train logistic regression using gradient ascent via line search
    #
    #  Xtrain  : a numpy.ndarray matrix of size [m x n] and type 'float' where each row
    #            is a n-dimensional input example (assume it already contains the constant feature set to 1)
    #  Ytrain  : a numpy.ndarray vector of size [m x 1] and type 'float', where the
    #            i-th element is the correct labelfor the i-th input example.
    #  theta_init   : a numpy.ndarray vector of size [n x 1] and type 'float' containing
    #                 the initial model parameters
    #  alpha0  : float, the initial (large) step size used for line search
    #  tol     : float, tolerance value used in the stopping condition

    # OUTPUT
    #  theta   : a numpy.ndarray vector of size [n x 1] and type 'float'  containing
    #            the learned model parameters
    #  n_iter  : int, the number of gradient ascent iterations until convergence
    #  loglik  : a numpy.ndarray vector of size [n_iter x 1] and type 'float'  containing
    #            the log likelihood value at each iteration

    # HINT
    #  your program should use the following stopping criterion:
    #        while (np.linalg.norm(grad)>tol) and (n_iter < 100000)
    #
    #  where grad is the gradient at the current iteration



    n_iter = 1
    n = Xtrain.shape[1]
    grad = np.ones(n) * 1000 # grad set to a large value just to enter the while loop
    theta = theta_init
    loglik_i = q2_loglik(Xtrain, Ytrain, theta)
    loglik = np.array([loglik_i])

    while (np.linalg.norm(grad)>tol) and (n_iter < 100000):
        # insert your code here
        alpha = alpha0
        grad = q2_gradient(Xtrain, Ytrain, theta)

        while q2_loglik(Xtrain, Ytrain, theta + (alpha * grad)) <= loglik_i:
            alpha = alpha/2

        theta = theta + (alpha * grad)
        loglik_i = q2_loglik(Xtrain, Ytrain, theta)
        loglik = np.hstack((loglik, loglik_i))
        n_iter = n_iter + 1
    return (theta, n_iter, loglik)
