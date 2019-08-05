import numpy as np

def q1_gmminit(X, K, labels):
    # Initializes a GMM model, given an initial clustering.

    # INPUT:
    #  X : a numpy.ndarray matrix of size [m x n] and type 'float', where each row
    #      is an n-dimensional input example
    #  K: 'int', indicating the number of gaussians for the GMM
    #  labels: a numpy.ndarray vector of size [m x 1] and type 'int',
    #          containing the labels that the K-means algorithm assigned to the examples.
    #          labels[i] is an element of {0, ..., K-1}, and it indicates the
    #          cluster initially associated to the i-th example
    #
    # OUTPUT:
    #  mus: a numpy.ndarray matrix of size [K x n] and type 'float', where the i-th row
    #       contains the n-dimensional mean of the i-th Gaussian
    #  sigmas: a numpy.ndarray matrix of size [K x n x n] and type 'float',
    #          where sigmas[i,:,:] is the [n x n] covariance matrix of the i-th Gaussian
    #  priors: a numpy.ndarray vector of size [K x 1] and type 'float',
    #          containing the mixture priors of the K Gaussians.

    # insert your code here
    m, n = X.shape
    priors = np.zeros((K, 1))
    mus = np.zeros((K, n))
    sigmas = np.zeros((K, n, n))

    for i in range(K):
        sum_i = labels[labels == i].shape[0]
        i_s = np.array(labels == i, dtype='int')
        priors[i] = sum_i/m
        mus[i] = np.sum(i_s[:, np.newaxis] * X, axis=0)/sum_i
        exp_m = i_s[:, np.newaxis] * (X - mus[i])
        sigmas[i] = exp_m.T.dot(exp_m)/sum_i
    return (mus, sigmas, priors)