import numpy as np
from q2_kernel import q2_kernel

def q2_predict_svm_dual(xtest, Xtrain, Ytrain, svs, alphas, bias, mode):
    # Predict the labels of the given examples, given the learned parameters for the
    # primal SVN formulation.
    #
    # INPUT:
    #  xtest: a numpy.ndarray vector of size [n x 1] and type 'float', containing
    #         an n-dimensional test example
    #  Xtrain : a numpy.ndarray matrix of size [m x n] and type 'float' where each row
    #           is a n-dimensional input training example
    #  Ytrain : a numpy.ndarray vector of size [m x 1] and type 'int16', where the
    #           i-th element is the correct label for the i-th input example (either -1 or 1)
    #  svs: a numpy.ndarray vector of size [nsv x 1] and type 'int64',
    #       containing the indices of the nsv training examples that are support vectors;
    #       for example, svs[0] will give the index of the first support vector
    #       and will be an integer number between 0 and m-1
    #  alphas : a numpy.ndarray vector of size [nsv x 1] and type 'float',
    #           containing the  the alpha coefficients associated to the support vectors
    #  bias: 'float' value containing the bias term of the model
    #  mode   : a 'str' indicating the type of kernel; it can be either 'linear' or 'polynomial'

    # OUTPUT:
    #  label: 'int16' containing the predicted label for the test example (+1 or -1)
    #  s: 'float' the SVM score for the test example

    # insert your code here
    nsv = svs.shape[0]

    x_svs = Xtrain[svs]
    y_svs = Ytrain[svs]
    k = np.array([q2_kernel(x_svs[i], xtest, mode) for i in range(nsv)])
    k = np.diag(k)

    s = float(np.sum(alphas * y_svs * k) + bias)
    label = np.array([1], dtype='int16')[0]
    if s <= 0:
        label = np.array([1], dtype='int16')[0]
    return (label, s)