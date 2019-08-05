import numpy as np
from q1_predict import q1_predict
from q1_mse import q1_mse

def q1_test_error(Xtrain, Ytrain, Xtest, Ytest, tau_vec):
    # Given training and test set, it trains the model and calculates the test error.
    #
    # INPUT
    #  Xtrain  : a numpy.ndarray matrix of size [m x n] and type 'float' where each row
    #            is a n-dimensional input training example
    #  Ytrain  : a numpy.ndarray vector of size [m x 1] and type 'float', where the
    #            i-th element is the correct label for the i-th input training example
    #  Xtest   : a numpy.ndarray matrix of size [M x n] and type 'float',
    #            where each row is a n-dimensional input *test* example
    #  Ytest   : a numpy.ndarray vector of size [M x 1] and type 'float', where the
    #            i-th element is the correct label for the i-th input *test* example
    #  tau_vec : a numpy.ndarray vector of size [K x 1] containing K distinct values
    #            of regularization hyperparameter
    #
    # OUTPUT
    #  error : a numpy.ndarray vector of size [K x 1] containing
    #          MSE test errors, one for each value of hyperparameter tau_vec
    #
    k = tau_vec.shape[0]
    m = Ytest.shape[0]
    error = np.zeros(k)
    for i in range(k):
        pred_Y = np.zeros(m)
        for j in range(m):
            pred_Y[j] = q1_predict(Xtrain, Ytrain, Xtest[j], tau_vec[i])
        error[i] = q1_mse(Ytest, pred_Y)
    return error

