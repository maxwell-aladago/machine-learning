import numpy as np
from q1_logprobgauss import q1_logprobgauss

def q1_GM_Expectation(X, mus, sigmas, priors):
    # Executes the Expectation-step for the learning of a GMM.
    #
    # INPUT:
    #  X : a numpy.ndarray matrix of size [m x n] and type 'float', where each row
    #      is an n-dimensional input example
    #  mus: a numpy.ndarray matrix of size [K x n] and type 'float', where the i-th row
    #       contains the n-dimensional mean of the i-th Gaussian
    #  sigmas: a numpy.ndarray matrix of size [K x n x n] and type 'float',
    #          where sigmas[i,:,:] is the [n x n] covariance matrix of the i-th Gaussian
    #  priors: a numpy.ndarray vector of size [K x 1] and type 'float',
    #          containing the mixture priors of the K Gaussians.
    #
    # OUTPUT:
    #  prob_c: a numpy.ndarray matrix of size [m x K] and type 'float',
    #          containing the posterior probabilities over the K Gaussians for the m examples.
    #          Specifically, prob_c[i,j] represents the probability that the
    #          i-th example belongs to the j-th Gaussian, i.e., P(z^(i) = j | X^(i))
    #  free_energy_e: 'float' representing the free energy value
    #  likelihood_e: 'float' representing the log-likelihood value

    # insert your code here

    m, n = X.shape
    K, _ = mus.shape

    # evaluate P(x^(i) | z^(i); pi, mus, sigmas)
    den_x = np.zeros((m, K))
    for i in range(m):
        for j in range(K):
            den_x[i, j] = q1_logprobgauss(X[i], mus[j], sigmas[j])

    joint_prob = np.exp(den_x) * priors.reshape(-1, K)

    # compute marginal probabilities
    prob_x = np.sum(joint_prob, axis=1)

    # compute actual posterior probability
    prob_c = joint_prob/prob_x[:, np.newaxis]
    likelihood_e = np.sum(np.log(prob_x))
    free_energy_e = np.sum(prob_c * np.log(prob_x).reshape(-1, 1))
    return (prob_c, free_energy_e, likelihood_e)