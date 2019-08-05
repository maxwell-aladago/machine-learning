import numpy as np
def q3_likelihood(mu, m, H):
    # Returns the likelihood for different values of mu, given the scalar parameters m and H.
    #
    # INPUT
    #  mu: N-dimensional numpy.ndarray vector of type 'float' containing N different values for mu
    #  m: int
    #  H: int
    #
    # OUTPUT
    #  lik: N-dimensional numpy.ndarray vector  of type 'float' containing the likelihood values associated with the entries of mu
    
    # insert your code here
    lik = np.power(mu, H) * np.power(1 - mu, m-H)
    return lik
