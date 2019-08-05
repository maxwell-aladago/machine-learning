import numpy as np

def q2_predict(X, theta):
    # Predict the labels and probabilities for the set of examples X using the
    # model theta

    # INPUT
    #  X  : a numpy.ndarray matrix of size [m x n] and type 'float' where each row is a n-dimensional
    #       input example (please assume it already contains the constant feature set to 1)
    #  theta   : a numpy.ndarray vector of size [n x 1] and type 'float'
    #            containing the model parameters used to make predictions
    # OUTPUT
    #  pred_Y : a numpy.ndarray vector of size [m x 1] and type 'float', containing
    #           the predicted labels for the examples in X. Please note that pred_Y
    #           has binary values {0,1} in this case
    #  prob_Y : a numpy.ndarray vector of size [m x 1] and type 'float', containing
    #           the posterior probabilities produced by the logistic function

    prob_Y = 1/(1 + np.exp(-1 * X.dot(theta)))
    pred_Y = np.ones(prob_Y.shape)
    pred_Y[prob_Y<= 0.5] = 0.0
    return (pred_Y, prob_Y)
