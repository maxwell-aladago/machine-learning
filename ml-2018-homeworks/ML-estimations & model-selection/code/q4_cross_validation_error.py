import numpy as np
import math
from q4_train import q4_train
from q4_predict import q4_predict
from q4_mse import q4_mse


def q4_cross_validation_error(X, Y, lambdavec, mode, N):
    # Calculates the cross-validation errors for different values of lambdavec, given the
    # training set X, Y.
    #
    # ** Implementation notes **
    # - As discussed in class, you should first randomly permute the examples, before starting the
    #   cross-validation stage. Here we did it for you: we created Xr and Yr which are obtained from
    #   X and Y by permuting examples. You should use Xr and Yr in your code (not X and Y)
    # - Do not change/initialize/reset the Python pseudo-number generator.
    #
    # INPUT
    #  X: a numpy.ndarray matrix of size [m x d] and type 'float' where each row
    #     is a d-dimensional input example
    #  Y: a numpy.ndarray vector of size [m x 1] and type 'float', where the
    #     i-th element is the correct output value for the i-th input example.
    #  lambdavec: a numpy.ndarray vector of size [k x 1] and type 'float'
    #             containing the set of regularization hyperparameter values
    #  mode: specifies the type of features;
    #        it is a 'str' that can be either 'linear' or 'quadratic'.
    #  N: `int' representing the number of folds for the cross-validation stage
    #
    # OUTPUT
    #  error: a numpy.ndarray vector of size [k x 1] and type 'float'
    #         containing the cross-validation error (i.e., the average of the mean
    #         squared errors over the N validation sets) for each value in lambdavec.
    #


    # ********  DO NOT TOUCH THE FOLLOWING 5 LINES  ********************
    np.random.seed(0)
    m = X.shape[0]
    idxperm = np.random.permutation(m)-1
    Xr = X[idxperm,:]
    Yr = Y[idxperm]
    # ******************************************************************

    # insert your code here
    # make sure to use Xr and Yr in your code, NOT X and Y (read Implemantiation notes in the header)

    # important local variables
    error = np.zeros(lambdavec.shape[0])
    fold_size = math.ceil(Xr.shape[0]/N)
    m = Xr.shape[0]

    """In the code below, slicing is used to aggregate training and validation
         folds instead of first diving the datasets upfront using array_split().
          The reason is because of the high time cost concatenating many sub-arrays returned
          by array_copy for the training fold. Slicing is much faster.  
    """
    for run in range(N):
        end_index = (run + 1) * fold_size
        start_index = run * fold_size

        # check for possible out of index error due to the above operation
        if end_index > m: end_index = m

        # divide data into training and validation folds
        Xr_train = np.concatenate((Xr[0:start_index], Xr[end_index:m]), axis = 0)
        Xr_validate = Xr[start_index: end_index]
        Yr_train = np.concatenate((Yr[0:start_index], Yr[end_index:m]), axis = 0)
        Yr_validate = Yr[start_index: end_index]

        for lmindex in range(lambdavec.shape[0]):
            theta = q4_train(Xr_train,Yr_train,lambdavec[lmindex], mode)
            pred_Y = q4_predict(theta, Xr_validate, mode)
            error[lmindex] += q4_mse(pred_Y, Yr_validate)

    error = error/N
    return error
