import numpy as np
from q1_GM_Expectation import q1_GM_Expectation
from q1_GM_Maximization import q1_GM_Maximization

def q1_GaussianMixture(X, mus_init, sigmas_init, priors_init, num_iterations):
    # Learn a Gaussian mixture model (GMM) by using the Expectation-Maximization (EM) algorithm.
    #
    # INPUT:
    #  X : a numpy.ndarray matrix of size [m x n] and type 'float', where each row
    #      is an n-dimensional input example
    #  mus_init: a numpy.ndarray matrix of size [K x n] and type 'float', where the i-th row
    #            contains the n-dimensional mean of the i-th Gaussian
    #  sigmas_init: a numpy.ndarray matrix of size [K x n x n] and type 'float',
    #               where sigmas[i,:,:] is the [n x n] covariance matrix of the i-th Gaussian
    #  priors_init: a numpy.ndarray vector of size [K x 1] and type 'float',
    #               containing the mixture priors of the K Gaussians.
    #  num_iterations: 'int', indicating the number of EM iterations.
    #
    # OUTPUT:
    #  mus: a numpy.ndarray matrix of size [K x n] and type 'float', where the i-th row
    #       contains the n-dimensional mean of the i-th Gaussian
    #  sigmas: a numpy.ndarray matrix of size [K x n x n] and type 'float',
    #          where sigmas[i,:,:] is the [n x n] covariance matrix of the i-th Gaussian
    #  priors: a numpy.ndarray vector of size [K x 1] and type 'float',
    #          containing the mixture priors of the K Gaussians.
    #  likelihood_e: a numpy.ndarray vector of size [num_iterations x 1] and type 'float',
    #                containing the likelihood after each E-step.
    #  free_energy_e: a numpy.ndarray vector of size [num_iterations x 1] and type 'float',
    #                 containing the free energy after each E-step.
    #  likelihood_m: a numpy.ndarray vector of size [num_iterations x 1] and type 'float',
    #                containing the likelihood after each M-step.
    #  free_energy_m: a numpy.ndarray vector of size [num_iterations x 1] and type 'float',
    #                 containing the free energy after each M-step.

    # insert your code here
    mus = np.copy(mus_init)
    sigmas = np.copy(sigmas_init)
    priors = np.copy(priors_init)
    free_energy_m = np.zeros((num_iterations, 1))
    free_energy_e = np.zeros((num_iterations, 1))
    likelihood_m = np.zeros((num_iterations, 1))
    likelihood_e = np.zeros((num_iterations, 1))

    for i in range(num_iterations):
        prob_c, free_energy_e[i], likelihood_e[i] = q1_GM_Expectation(X, mus, sigmas, priors)
        mus, sigmas, priors, free_energy_m[i], likelihood_m[i] = q1_GM_Maximization(X, prob_c)

    return (mus, sigmas, priors, likelihood_e, free_energy_e, likelihood_m, free_energy_m)
