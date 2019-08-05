import numpy as np

def q1_top_words(phi_y0, phi_y1, phi_prior, k):
    # For each class, finds the words that are most indicative of a message
    # belonging to that class

    # INPUT
    #  phi_y0    : a numpy.ndarray vector of size [n x 1] and type 'float' containing
    #              the  class conditional probabilities for y=0
    #              where phi_y0[j] = p(x_j = 1 | y = 0)
    #  phi_y1    : a numpy.ndarray vector of size [n x 1] and type 'float' containing
    #              the  class conditional probabilities for y=1
    #              where phi_y1[j] = p(x_j = 1 | y = 1)
    #  phi_prior : 'float' representing the prior probability of y being 1, i.e., phi_prior = p(y = 1)
    #  k         : 'int', the number of words to output

    # OUTPUT
    #  word_idx  : a numpy.ndarray matrix of size [2 x k] and type 'int',
    #              the first row contains the indices of the k most indicative
    #              words for class y=0, the  second row the ones for y=1.
    #              Indices should be integer values between 0 and n-1

    # insert your code here
    y1 = np.log(phi_prior) + np.log(phi_y1)
    y0 = np.log(1 - phi_prior) + np.log(phi_y0)

    word_idx = np.zeros((2, k))

    word_idx[0] = y0.argsort()[-k:]
    word_idx[1] = y1.argsort()[-k:]
    word_idx = np.array(word_idx, dtype='int')
    return word_idx