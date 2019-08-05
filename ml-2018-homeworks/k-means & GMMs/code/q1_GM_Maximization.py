import numpy as np
from q1_logprobgauss import q1_logprobgauss

def q1_GM_Maximization(X, prob_c):
    # Executes Maximization-step for the learning of a GMM.
    #
    # INPUT:
    #  X : a numpy.ndarray matrix of size [m x n] and type 'float', where each row
    #      is an n-dimensional input example
    #  prob_c: a numpy.ndarray matrix of size [m x K] and type 'float',
    #          containing the posterior probabilities over the K Gaussians for the m examples.
    #          Specifically, prob_c[i,j] represents the probability that the
    #          i-th example belongs to the j-th Gaussian, i.e., P(z^(i) = j | X^(i))
    #
    # OUTPUT:
    #  mus: a numpy.ndarray matrix of size [K x n] and type 'float', where the i-th row
    #       contains the n-dimensional mean of the i-th Gaussian
    #  sigmas: a numpy.ndarray matrix of size [K x n x n] and type 'float',
    #          where sigmas[i,:,:] is the [n x n] covariance matrix of the i-th Gaussian
    #  priors: a numpy.ndarray vector of size [K x 1] and type 'float',
    #          containing the mixture priors of the K Gaussians.
    #  free_energy_m: 'float' representing the free energy value
    #  likelihood_m: 'float' representing the log-likelihood value


    # insert your code here

    K = prob_c.shape[1]
    m, n = X.shape

    # maximization of step: update priors
    pK = np.sum(prob_c, axis=0)
    priors = pK/m

    # compute the new means fo the gaussians
    mus = prob_c.T.dot(X)
    mus = mus/pK[:, np.newaxis]

    # compute the new the covariances
    sigmas = np.zeros((K, n, n))
    for i in range(K):
        diff = X - mus[i]
        sigmas[i] = np.linalg.multi_dot((diff.T, np.diag(prob_c[:,  i]), diff))

    sigmas = sigmas/pK[:, np.newaxis, np.newaxis]

    # valuate the joint probabilities again
    den_x = np.empty_like(prob_c)
    for i in range(m):
        for j in range(K):
            den_x[i, j] = q1_logprobgauss(X[i], mus[j], sigmas[j])

    joint_prob = np.exp(den_x) * priors
    prob_x = np.sum(joint_prob, axis=1)
    likelihood_m = np.sum(np.log(prob_x))
    # use previous posterior probabilities with new parameters
    free_energy_m = np.sum(prob_c * np.log(prob_x).reshape(-1, 1))

    return (mus, sigmas, priors, free_energy_m, likelihood_m)