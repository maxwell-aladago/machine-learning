import numpy as np
def q1_nb_predict(X, phi_y0, phi_y1, phi_prior):
    # Predicts the labels of examples X, given the trained model

    # INPUT
    #  X : a numpy.ndarray matrix of size [m x n] and type 'uint8' where each row
    #      is a n-dimensional input training example  and each entry is a binary feature
    #  phi_y0    : a numpy.ndarray vector of size [n x 1] and type 'float' containing
    #              the  class conditional probabilities for y=0
    #              where phi_y0[j] = p(x_j = 1 | y = 0)
    #  phi_y1    : a numpy.ndarray vector of size [n x 1] and type 'float' containing
    #              the  class conditional probabilities for y=1
    #              where phi_y1[j] = p(x_j = 1 | y = 1)
    #  phi_prior : 'float' representing the prior probability of y being 1, i.e., phi_prior = p(y = 1)

    # OUTPUT
    #  pred_Y    : a numpy.ndarray vector of size [m x 1] and type 'uint8' containing
    #              the predicted label for each example.
    #              Each entry value should be either 0 or 1.

    # HINTS
    #  1. for each example compute pred_y = argmax_k p(y=k) \prod_{j=1}^n p(x_j|y=k)
    #  2. use the log function to avoid numerical problems:
    #       pred_y = argmax_k { \log{p(y=k)} + \sum_{j=1}^n \log{p(x_j|y=k)} }

    # insert your code here

    y1 = np.log(phi_prior) + np.sum(np.log(phi_y1) * X, axis=1)
    y0 = np.log(1 - phi_prior) + np.sum(np.log(phi_y0) * X, axis=1)
    pred_Y = np.argmax([y0, y1], axis=0)
    return pred_Y
