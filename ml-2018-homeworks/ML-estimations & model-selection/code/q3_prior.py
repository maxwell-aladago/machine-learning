import numpy as np
def q3_prior(mu, a, Z):
    # Returns the prior for multiple values of mu, given the parameters a and Z.
    #
    # INPUT
    #  mu: N-dimensional numpy.ndarray vector of type 'float' containing N different values for mu
    #  a: int
    #  Z: float
    #
    # OUTPUT
    #  prob: N-dimensional numpy.ndarray vector of type 'float' containing the prior probabilities associated with the entries of mu


    # insert your code here

    prob = (np.power(mu, a-1) * np.power((1 - mu), a -1))/Z
    return prob
