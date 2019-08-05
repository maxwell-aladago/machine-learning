import numpy as np
from q3_predict import q3_predict
from q1_error import q1_error

def q3_test_error(Xtrain, Ytrain, Xtest, Ytest, k):
    # Computes the test error of kNN for different values of k
    #
    # INPUT
    #  Xtrain : a numpy.ndarray matrix of size [m x n] and type 'float' where each row
    #           is a n-dimensional input training example
    #  Ytrain : a numpy.ndarray vector of size [m x 1] and type 'uint8', where the
    #           i-th element is the correct label for the i-th input training example (either 0 or 1)
    #  Xtest  : a numpy.ndarray matrix of size [mtest x n] and type 'float' where each row
    #           is the input feature vector of a test example
    #  Ytest : a numpy.ndarray vector of size [mtest x 1] and type 'uint8', where the
    #           i-th element is the correct label for the i-th input test example (either 0 or 1)
    #  k      : a numpy.ndarray matrix of size [L x 1] and type 'uint8'
    #           containing the different values of parameter k to be used by kNN
    #
    # OUTPUT
    #  error       : a numpy.ndarray matrix of size [mtest x 1] and type 'float'
    #                containing the misclassification errors of kNN on the
    #                test set for the L different choices of neighborhood size (i.e., the
    #                values in k)


    # insert your code here
    L = k.shape[0]
    mtest = Xtest.shape[0]
    error = np.zeros((L,))

    for i in range(L):
        pred_Y = np.zeros((mtest, ))
        k_i = k[i]
        for j in range(mtest):
            pred_Y[j] = q3_predict(Xtrain, Ytrain, Xtest[j], k_i)

        error[i] = q1_error(Ytest, pred_Y)

    return error
