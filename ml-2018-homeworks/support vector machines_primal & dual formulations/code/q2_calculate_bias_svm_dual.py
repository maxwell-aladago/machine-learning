from q2_kernel import q2_kernel

def  q2_calculate_bias_svm_dual(X, Y, svs, alphas, mode):
    # Calculate the bias given the learned parameters of a SVM dual formulation, and its training set.
    #
    # INPUT:
    #  X      : a numpy.ndarray matrix of size [m x n] and type 'float' where each row
    #           is a n-dimensional input training example
    #  Y      : a numpy.ndarray vector of size [m x 1] and type 'int16', where the
    #           i-th element is the correct label for the i-th input example (either -1 or 1)
    #  svs: a numpy.ndarray vector of size [nsv x 1] and type 'int64',
    #       containing the indices of the nsv training examples that are support vectors;
    #       for example, svs[0] will give the index of the first support vector
    #       and will be an integer number between 0 and m-1
    #  alphas : a numpy.ndarray vector of size [nsv x 1] and type 'float',
    #           containing the  the alpha coefficients associated to the support vectors
    #  mode   : a 'str' indicating the type of kernel; it can be either 'linear' or 'polynomial'
    #
    #
    # OUTPUT:
    #  bias: 'float' value containing the bias term of the model

    # insert your code here

    nsv = svs.shape[0]
    svs_Y = Y[svs]
    svs_X = X[svs]
    k = [[q2_kernel(svs_X[i], svs_X[j], mode) for j in range(nsv)] for i in range(nsv)]
    inner_sum = alphas * svs_Y * k
    inner_sum.sum(axis=1)
    bias = float((svs_Y - inner_sum).sum()/nsv)
    return bias