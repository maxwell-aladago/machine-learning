import numpy as np
from q3_predict import q3_predict
from q1_error import q1_error


def q3_leave_one_out_error(Xtrain, Ytrain, k):
    # Evaluate the leave-one-out validation error of kNN with
    # different values of k

    # INPUT
    #  Xtrain : a numpy.ndarray matrix of size [m x n] and type 'float' where each row
    #           is a n-dimensional input training example
    #  Ytrain : a numpy.ndarray vector of size [m x 1] and type 'uint8', where the
    #           i-th element is the correct label for the i-th input training example (either 0 or 1)
    #  k      : a numpy.ndarray vector of size [L x 1] and type 'uint8' containing
    #           the different values of parameter k to be used by kNN
    #
    # OUTPUT
    #  error  : a numpy.ndarray vector of size [L x 1] and type 'float' containing
    #           the leave one out validation error of k-NN for the L different choices
    #           of neighborhood size (i.e., the values in k)

    # insert your code here
    error = np.zeros(k.shape)
    N = Xtrain.shape[0]
    K = k.shape[0]
    for i in range(N):
        trainX = np.vstack((Xtrain[0:i], Xtrain[i + 1: N]))
        trainY = np.hstack((Ytrain[0:i], Ytrain[i + 1: N]))
        validateX = Xtrain[i]
        validateY = Ytrain[i]

        for j in range(K):
            pred_Y = q3_predict(trainX, trainY, validateX, k[j])
            error[j] += q1_error(validateY, pred_Y)

    error = error / N
    return error
