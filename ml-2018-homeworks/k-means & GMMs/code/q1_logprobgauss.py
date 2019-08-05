import numpy as np


def q1_logprobgauss(x, mu, sigma):
    # Calculates the log-probability density value for the input example under a
    # given multivariate Gaussian, i.e., log(P(x;  mu, sigma))
    #
    # INPUT:
    #  x     : a numpy.ndarray vector of size [n x 1] and type 'float'
    #          representing a single input example
    #  mu    : a numpy.ndarray vector of size [n x 1] and type 'float'
    #          representing the mean of the Gaussian
    #  sigma : a numpy.ndarray matrix of size [n x n] and type 'float'
    #          containing the covariance matrix of the Gaussian
    #
    # OUTPUT:
    #  logprob: 'float',  representing the log of the probability density value

    # insert your code here
    n = x.shape[0]
    # compute the term independent from the mean
    const = n * (np.log(np.pi) + np.log(2)) + np.log(np.linalg.det(sigma))
    const = const * -0.5
    var = x - mu
    # compute the exponential term of the gaussian
    exp = -0.5 * np.linalg.multi_dot((var.T, np.linalg.inv(sigma), var))
    logprob = const + exp
    return logprob 