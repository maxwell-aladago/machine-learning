import numpy as np


def q4_features(X, mode):
    # Given the data matrix X (where each row X[i,:] is an example), the function
    # computes the feature matrix B, where row B[i,:] represents the feature vector
    # associated to example X[i,:]. The features should be either linear or quadratic
    # functions of the inputs, depending on the value of the input argument 'mode'.
    # Please make sure to implement the features according to the *exact* order
    # specified in the text of the homework assignment.
    #
    # INPUT:
    #  X: a numpy.ndarray matrix of size [m x d] and type 'float' where each row
    #     is a d-dimensional input example
    #  mode: specifies the type of features;
    #        it is a 'str' that can be either 'linear' or 'quadratic'.
    #
    # OUTPUT:
    #  B: a numpy.ndarray matrix of size [m x n] and type 'float', with each row
    #     containing the feature vector of an example

    # X_linear = np.pad(X, ((0, 0), (1, 0)), 'constant', pad_values=1)[1:] Will do the same thing as the code below but
    # it's much slower than the concatenate() function. 

    X_linear = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)


    if mode == 'linear':
        # insert your code here
        B = X_linear
    elif mode == 'quadratic':
        # insert your code here
        m = X_linear.shape[0]
        n = X_linear.shape[1]
        # only values of interest are those of the upper triangular matrix formed
        # from the outer product of X^(i) by X^(i)
        uptriang_indices = np.triu_indices(n)
        B = np.vstack([np.outer(X_linear[i], X_linear[i])[uptriang_indices] for i in range(m)])
    else:
        print('Error, only linear and quadratic forms are supported')
        return np.array([])

    return B
