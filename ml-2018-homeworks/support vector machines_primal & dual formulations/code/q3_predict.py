import numpy as np
def q3_predict(Xtrain, Ytrain, xtest, k):
    # Uses kNN to predict the label of the input example given the training set
    # (Xtrain, Ytrain) and the neighborhood size k.

    # INPUT
    #  Xtrain : a numpy.ndarray matrix of size [m x n] and type 'float' where each row
    #           is a n-dimensional input training example
    #  Ytrain : a numpy.ndarray vector of size [m x 1] and type 'uint8', where the
    #           i-th element is the correct label for the i-th input training example (either 0 or 1)
    #  xtest  : a numpy.ndarray vector of size [n x 1], the input feature vector of test example
    #  k      : 'float', the neighborhood size used by kNN
    #
    # OUTPUT
    #  pred_y : 'uint8', the predicted label for xtest

    # HINT
    #  It is possible to implement this function without using for or while
    #  loops. This can be achieved via vectorization. This will make your code much faster.

    # insert your code here
    L2_squared = np.linalg.norm(Xtrain - xtest, axis=1)

    top_k = Ytrain[L2_squared.argsort()[:k]]
    labels_uniq, freq = np.unique(top_k, return_counts=True)
    pred_y = labels_uniq[freq.argsort()[-1]]

    pred_y = np.array([pred_y], dtype='uint8')[0]

    return pred_y