import numpy as np

def q1_nb_train(Xtrain, Ytrain):
    # Train a Naive Bayes model using Laplacian smoothing

    # INPUT
    #  Xtrain : a numpy.ndarray matrix of size [m x n] and type 'uint8' where each row
    #           is a n-dimensional input training example and each entry is a binary feature
    #  Ytrain : a numpy.ndarray vector of size [m x 1] and type 'uint8', where the
    #           i-th element is the correct label for the i-th input example (either 0 or 1)
    #
    # OUTPUT
    #  phi_y0    : a numpy.ndarray vector of size [n x 1] and type 'float' containing
    #              the  class conditional probabilities for y=0
    #              where phi_y0[j] = p(x_j = 1 | y = 0)
    #  phi_y1    : a numpy.ndarray vector of size [n x 1] and type 'float' containing
    #              the  class conditional probabilities for y=1
    #              where phi_y1[j] = p(x_j = 1 | y = 1)
    #  phi_prior : 'float' representing the prior probability of y being 1, i.e., phi_prior = p(y = 1)

    # insert your code here

    m, _ = Xtrain.shape
    num_y = len(Ytrain[Ytrain == 1])
    y1 = np.array(np.logical_and(Xtrain == 1, (Ytrain == 1).reshape(m, -1)), dtype='int16')
    y0 = np.array(np.logical_and(Xtrain == 1, (Ytrain == 0).reshape(m, -1)), dtype='int16')

    # sum for each  and add laplacian smoothing term
    # class 1
    y1 = np.sum(y1, axis=0) + 1


    # class 0
    y0 = np.sum(y0, axis=0) + 1

    phi_y0 = y0/(2 + m - num_y)
    phi_y1 = y1/(2 + num_y)
    phi_prior =  num_y/ m

    return (phi_y0, phi_y1, phi_prior)


