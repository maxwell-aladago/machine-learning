import numpy as np
def q3_posterior(mu, m, H, a, Z):
    # Returns the posterior for multiple values of mu, given the parameters m, H, a, and Z.
    #
    # INPUT
    #  mu: N-dimensional numpy.ndarray vector of type 'float' containing N different values for mu
    #  m: scalar
    #  H: scalar
    #  a: scalar
    #  Z: scalar
    #
    # OUTPUT
    #  prob: N-dimensional numpy.ndarray vector of type 'float' containing he posterior values associated with the entries of mu

    # insert your code here

    prob = (np.power(mu, H + a - 1) * np.power(1-mu, m + a - H - 1))/Z
    return prob
