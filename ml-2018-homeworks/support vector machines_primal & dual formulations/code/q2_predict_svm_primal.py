import numpy as np

def q2_predict_svm_primal(X, w, b):
    # Predict the labels of the given examples, given the learned parameters for the
    # primal SVM formulation.
    #
    # INPUT:
    #  X      : a numpy.ndarray matrix of size [m x n] and type 'float' where each row
    #           is a n-dimensional input training example
    #  w : a numpy.ndarray vector of size [n x 1] and type 'float',
    #      containing the learned model parameters (coefficients of the hyperplane).
    #  b : 'float', the bias of the hyperplane.
    #
    # OUTPUT:
    #  labels: : a numpy.ndarray vector of size [m x 1] and type 'int16', where the
    #           i-th element is the predicted label for the i-th input example (either -1 or 1)
    #  S       : a numpy.ndarray vector of size [m x 1] and type 'float', where the i-th
    #            element is the SVM score for the i-th input example, i.e., (w'*x + b) in case of a single example x

    # insert your code here
    labels = np.zeros(X.shape[0])
    S = np.array(X.dot(w) + b, dtype='float')
    labels[S > 0] = 1
    labels[S <= 0] = -1
    return (labels, S)